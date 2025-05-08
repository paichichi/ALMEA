import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from models.mmd import MMD
from models.mmi import MMI
from src.utils import pairwise_distances, csls_sim

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import random

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class CVAE(nn.Module):

    def __init__(self, args, in_dim, hidden_dims, latent_dim=None, **kwargs):
        super(CVAE, self).__init__()
        self.args = args

        if latent_dim:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = hidden_dims[-1]

        modules = []

        # encoder hidden_dims[300,300,300]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        x = self.encoder(x)
        mu_x = self.fc_mu(x)
        log_var_x = self.fc_var(x)

        return (mu_x, log_var_x)

    def decode(self, z, reparameterize=False):
        if reparameterize:
            z = self.reparameterize(*z)
        z = self.decoder_input(z)
        # 通过潜在变量重建模态嵌入
        x = self.decoder(z)

        return x

    def masked_decode(self, z, reparameterize=False):
        if reparameterize:
            z = self.reparameterize(*z)
        z = self.masked_decoder_input(z)
        # 通过潜在变量重建模态嵌入
        x = self.masked_decoder(z)

        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, args, embs, train_links):
        extra_latent_x, extra_zz_x, extra_r_x, extra_x = None, None, None, None
        extra_latent_y, extra_zz_y, extra_r_y, extra_y = None, None, None, None

        x_index = train_links[:, 0]
        y_index = train_links[:, 1]

        x = embs[x_index]
        y = embs[y_index]

        masked_x, x_mask = Mask(self.args, x).apply_mask()
        masked_y, y_mask = Mask(self.args, y).apply_mask()

        zz_x, zz_y = self.encode(masked_x), self.encode(masked_y)
        full_zz_x, full_zz_y = self.encode(x), self.encode(y)

        latent_x = self.reparameterize(*zz_x)
        latent_y = self.reparameterize(*zz_y)

        unmasked_latent_x = self.reparameterize(*full_zz_x)
        unmasked_latent_y = self.reparameterize(*full_zz_y)

        r_x, r_y = self.decode(latent_x, reparameterize=False), self.decode(
            latent_y, reparameterize=False)
        full_r_x, full_r_y = self.decode(unmasked_latent_x, reparameterize=False), self.decode(
            unmasked_latent_y, reparameterize=False)

        """注释在test里"""
        flows = {'x': (
            x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x),
                 'y': (
            y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y)}

        return flows


class NeighborDecoder(nn.Module):
    def __init__(self, sub_dim, ent_embs) -> None:
        super().__init__()

        self.ent_embs = None
        self.subdecoder = nn.Sequential(nn.Linear(sub_dim, sub_dim),
                                        nn.Tanh(),
                                        nn.Dropout(0.5),
                                        nn.BatchNorm1d(sub_dim),
                                        nn.Linear(sub_dim, sub_dim),
                                        nn.Tanh(),
                                        nn.Dropout(0.5),
                                        nn.BatchNorm1d(sub_dim),
                                        )
        self.register_parameter('bias', nn.Parameter(torch.zeros(ent_embs.shape[0])))

    def forward(self, x):
        output = self.subdecoder(x)
        output = x @ self.ent_embs.T + self.bias
        return F.tanh(output)


class Mask:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.mask_probability = self.args.mask
        self.masks = self.create_masks()

    def create_masks(self):
        data_len = self.data.shape[0]
        masks = (torch.rand(data_len, device=self.args.device) > self.mask_probability).float().view(-1, 1)
        return masks

    def apply_mask(self):
        masked_data = self.data * self.masks
        return masked_data, self.masks


class GEEA(nn.Module):

    def __init__(self, args, kgs, concrete_features, sub_dims, joint_dim, ent_embs, fusion_layer):
        self.num_epochs = 0
        self.latent_z_first_iter = None
        self.latent_z_last_iter = None
        self.labels_first_iter = None
        self.labels_last_iter = None
        super().__init__()
        self.args = args
        self.kgs = kgs

        self.latent_dim = sub_dims[0]  # 预设为300

        self.subgenerators = []
        self.subdecoders = []
        self.none_linear_transforms = []

        self.ModalWeightingMLP = []

        self.num_none_concrete_feature = 0

        for i, sub_dim, concrete_feature in zip(range(len(sub_dims)), sub_dims, concrete_features):
            if concrete_feature is not None:
                # in_dim是300
                # hidden_dims是[300, 300, 300]
                # latent_dim是[300, 300, 300]
                subgenerator = CVAE(args=args,
                                    in_dim=sub_dim,
                                    hidden_dims=[sub_dim, ] * args.num_layers,
                                    latent_dim=sub_dim)
                self.subgenerators.append(subgenerator)

                # none_linear_transform = nn.Sequential(
                #     nn.Linear(sub_dim, 300),
                #     nn.Tanh(),
                #     nn.Dropout(0.5),
                #     nn.BatchNorm1d(300),
                #     nn.Linear(300, sub_dim),
                #     nn.ReLU(),
                # )
                none_linear_transform = nn.Sequential(
                    nn.Linear(sub_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.Tanh(),
                    nn.Dropout(0.3),
                    nn.Linear(300, sub_dim),
                    nn.Softplus(),
                    nn.BatchNorm1d(sub_dim)
                )
                self.none_linear_transforms.append(none_linear_transform)

                # mlp = nn.Sequential(
                #     nn.Linear(sub_dim, 300),
                #     nn.ReLU(),
                #     nn.Linear(300, 1)
                # )
                mlp = nn.Sequential(
                    nn.Linear(sub_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.ReLU(),
                    nn.Linear(300, 1)
                )
                self.ModalWeightingMLP.append(mlp)

                if i == -1:
                    # 这里的内容不会触发
                    subdecoder = NeighborDecoder(sub_dim, ent_embs)
                else:

                    subdecoder = nn.Sequential(nn.Linear(sub_dim, 1000),
                                               nn.Tanh(),
                                               nn.Dropout(0.5),
                                               nn.BatchNorm1d(1000),
                                               nn.Linear(1000, concrete_feature.shape[-1]),
                                               )
                self.subdecoders.append(subdecoder)
            else:
                # 记录name和char为空，我们只有四个模态
                self.num_none_concrete_feature += 1

        self.masked_decoder_input = nn.Linear(300, 300)
        # self.masked_decoder = nn.Sequential(nn.Linear(300, 300),
        #                                     nn.BatchNorm1d(300),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.5),
        #                                     nn.Linear(300, 300), )

        # # self.re_fusion_layer = nn.Sequential(nn.Linear(1200, 300),
        #                                      )
        self.masked_decoder = nn.Sequential(nn.Linear(300, 300),
                                            nn.Tanh(),
                                            nn.Dropout(0.5),
                                            nn.BatchNorm1d(300),
                                            nn.Linear(300, 300),)
        self.subgenerators = nn.ModuleList(self.subgenerators)
        self.subdecoders = nn.ModuleList(self.subdecoders)
        self.none_linear_transforms = nn.ModuleList(self.none_linear_transforms)
        self.ModalWeightingMLP = nn.ModuleList(self.ModalWeightingMLP)

        self.prior_reconstruction_loss_func = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss()
        self.post_reconstruction_loss_func = nn.MSELoss()
        self.concrete_features = concrete_features

        self.fusion_layer = fusion_layer

    # flows = {'x': (
    #         x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x, extra_latent_x, extra_zz_x,
    #         extra_r_x),
    #     'y': (
    #         y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y, extra_latent_y, extra_zz_y,
    #         extra_r_y)}
    def distribution_match_loss(self, outputs):

        def kld_loss(mu, logvar):
            return torch.mean(-.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        # output = (x, z=(mu, var), reconstrctued_x)
        x_distribution_match_loss = [
            kld_loss(*output['x'][4]) for output in outputs]
        y_distribution_match_loss = [
            kld_loss(*output['y'][4]) for output in outputs]

        unmasked_x_distribution_match_loss = [
            kld_loss(*output['x'][6]) for output in outputs]
        unmasked_y_distribution_match_loss = [
            kld_loss(*output['y'][6]) for output in outputs]

        return (sum(x_distribution_match_loss) + sum(y_distribution_match_loss) +
                sum(unmasked_x_distribution_match_loss) + sum(unmasked_y_distribution_match_loss))

    # def sampled_bce_loss(self, predicted, label, neg_ratio=5):
    #     pos_mask = torch.where(label > 0)
    #
    #     neg = torch.randint(high=label.shape[-1], size=(len(pos_mask[0]) * neg_ratio,))
    #     neg_mask = [pos_mask[0].repeat(neg_ratio), neg]
    #
    #     predicted_pos = predicted[pos_mask]
    #     label_pos = torch.ones_like(predicted_pos)
    #     predicted_neg = predicted[neg_mask]
    #     label_neg = torch.zeros_like(predicted_neg)
    #
    #     loss = self.prior_reconstruction_loss_func(predicted_pos, label_pos) + self.prior_reconstruction_loss_func(
    #         predicted_neg, label_neg) / neg_ratio
    #     return loss

    # def sampled_crossentropy_loss(self, predicted, label, neg_ratio=1):
    #     pos_mask, labels = torch.where(label > 0)
    #     sampled = torch.randperm(len(pos_mask))[:3500]
    #     pos_mask, labels = pos_mask[sampled], labels[sampled]
    #
    #     predicted_pos = predicted[pos_mask]
    #
    #     loss = F.cross_entropy(predicted_pos.to(self.args.device), labels.to(self.args.device))
    #     return loss

    # flows = {'x': (
    #         x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x, extra_latent_x, extra_zz_x,
    #         extra_r_x),
    #     'y': (
    #         y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y, extra_latent_y, extra_zz_y,
    #         extra_r_y)}
    def prior_reconstruction_loss(self, outputs, train_links):

        prior_reconstruction_loss = []

        for output, subdecoder, concrete_feature in zip(outputs, self.subdecoders, self.concrete_features):
            reconstructed_x = subdecoder(output['x'][2])
            reconstructed_y = subdecoder(output['y'][2])

            concrete_x = concrete_feature[train_links[:, 0]].cuda()
            concrete_y = concrete_feature[train_links[:, 1]].cuda()

            # 使用二进制交叉熵损失函数 - 相当于用还原的向量维度与原生进行对比
            loss_xy = self.prior_reconstruction_loss_func(
                reconstructed_x, concrete_y)
            loss_yx = self.prior_reconstruction_loss_func(
                reconstructed_y, concrete_x)

            loss_list = [loss_xy, loss_yx]

            flow_weights = [1, 1]
            prior_reconstruction_loss += [
                sum(loss * flow_weight for loss, flow_weight in zip(loss_list, flow_weights)), ]

        return sum(prior_reconstruction_loss)

    def re_fusion(self, sub_embs):
        sub_embs = sub_embs + [None, ] * self.num_none_concrete_feature
        return self.fusion_layer(*sub_embs)

    def post_reconstruction_loss(self, outputs, joint_emb, train_links):

        x, y = [], []

        for output, subdecoder in zip(outputs, self.subdecoders):
            x.append(output['x'][2])
            y.append(output['y'][2])

        # reconstructed
        reconstructed_xy = self.re_fusion(x)
        reconstructed_yx = self.re_fusion(y)

        # the targets
        x_index = train_links[:, 0]
        y_index = train_links[:, 1]

        joint_emb = joint_emb.detach()
        joint_x = joint_emb[x_index]
        joint_y = joint_emb[y_index]

        # loss
        loss_xy = self.post_reconstruction_loss_func(
            reconstructed_xy, joint_y)
        loss_yx = self.post_reconstruction_loss_func(
            reconstructed_yx, joint_x)

        return loss_xy + loss_yx

    # flows = {'x': (
    #         x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x),
    #     'y': (
    #         y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y)}
    def reconstruction_loss(self, outputs):
        loss = 0.
        for output in outputs:
            for key, data in output.items():
                input_, zz, output_, _, _, _, _, _, = data
                loss += self.post_reconstruction_loss_func(input_.detach(), output_)
        return loss

    # flow = [fused_masked_latent_x, fused_masked_latent_y, fused_non_masked_latent_x, fused_non_masked_latent_y]
    # def masked_decode_fuction_layer(self, outputs_second):
    #     fused_masked_latent_x = outputs_second[0]
    #     fused_masked_latent_y = outputs_second[1]
    #     fused_non_masked_latent_x = outputs_second[2]
    #     fused_non_masked_latent_y = outputs_second[3]
    #
    #     fused_masked_recon_x = self.masked_decoder(self.masked_decoder_input(fused_masked_latent_x))
    #     fused_masked_recon_y = self.masked_decoder(self.masked_decoder_input(fused_masked_latent_y))
    #     fused_unmasked_recon_x = self.masked_decoder(self.masked_decoder_input(fused_non_masked_latent_x))
    #     fused_unmasked_recon_y = self.masked_decoder(self.masked_decoder_input(fused_non_masked_latent_y))
    #
    #     # fused_unlabeled_recon_x = self.masked_decoder(self.masked_decoder_input(fused_unlabeled_latent_x))
    #     # fused_unlabeled_recon_y = self.masked_decoder(self.masked_decoder_input(fused_unlabeled_latent_y))
    #
    #     flow = [fused_masked_recon_x, fused_masked_recon_y, fused_unmasked_recon_x, fused_unmasked_recon_y]
    #     return flow

    # flows = {'x': (x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x, y_unlabeled,
    #                unlabeled_zz_x, unlabeled_latent_x, unlabeled_r_x),
    #          'y': (y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y, y_unlabeled,
    #                unlabeled_zz_x, unlabeled_latent_y, unlabeled_r_y)}
    def distribution_reconstruction_loss(self, outputs):
        loss = 0.
        fused_masked_recon_x, fused_masked_recon_y, fused_unmasked_recon_x, fused_unmasked_recon_y = [], [], [], []
        for output in outputs:
            fused_masked_recon_x.append(output['x'][2])
            fused_masked_recon_y.append(output['y'][2])
            fused_unmasked_recon_x.append(output['x'][7])
            fused_unmasked_recon_y.append(output['y'][7])

        reconstructed_fused_masked_x = self.re_fusion(fused_masked_recon_x)
        reconstructed_fused_masked_y = self.re_fusion(fused_masked_recon_y)
        reconstructed_fused_unmasked_x = self.re_fusion(fused_unmasked_recon_x)
        reconstructed_fused_unmasked_y = self.re_fusion(fused_unmasked_recon_y)

        loss += self.post_reconstruction_loss_func(reconstructed_fused_masked_x, reconstructed_fused_unmasked_x)
        loss += self.post_reconstruction_loss_func(reconstructed_fused_masked_y, reconstructed_fused_unmasked_y)

        loss += self.post_reconstruction_loss_func(reconstructed_fused_masked_x, reconstructed_fused_unmasked_y)
        loss += self.post_reconstruction_loss_func(reconstructed_fused_masked_y, reconstructed_fused_unmasked_x)
        return loss

    # def distribution_post_reconstruction_loss(self, outputs_third, joint_emb, train_links):
    #
    #     fused_masked_recon_x = outputs_third[0]
    #     fused_masked_recon_y = outputs_third[1]
    #     fused_unmasked_recon_x = outputs_third[2]
    #     fused_unmasked_recon_y = outputs_third[3]
    #
    #     x_index = train_links[:, 0]
    #     y_index = train_links[:, 1]
    #
    #     joint_emb = joint_emb.detach()
    #     joint_x = joint_emb[x_index]
    #     joint_y = joint_emb[y_index]
    #
    #     # 均方误差损失计算loss
    #     loss_masked = self.post_reconstruction_loss_func(
    #         fused_masked_recon_x, joint_y)
    #     loss_masked += self.post_reconstruction_loss_func(
    #         fused_masked_recon_y, joint_x)
    #
    #     loss_unmasked = self.post_reconstruction_loss_func(
    #         fused_unmasked_recon_x, joint_y)
    #     loss_unmasked += self.post_reconstruction_loss_func(
    #         fused_unmasked_recon_y, joint_x)
    #
    #     return loss_masked + loss_unmasked

    # def encode(self, xs, sub_embs):
    #     sub_embs = [embs for embs in sub_embs if embs is not None]
    #
    #     x_zs = [subgenerator.encode(embs[xs])
    #             for embs, subgenerator in zip(sub_embs, self.subgenerators)]
    #
    #     return x_zs

    # def decode(self, zs, reparameterize=False):
    #
    #     reconstructed_x = [subgenerator.decode(z, reparameterize=reparameterize)
    #                        for subgenerator, z in zip(self.subgenerators, zs)]
    #
    #     return reconstructed_x

    # def masked_decode(self, zs):
    #
    #     z = self.masked_decoder_input(zs)
    #
    #     x = self.masked_decoder(z)
    #
    #     return x

    # def sample(self, num):
    #     z = torch.randn(num, self.latent_dim).to(self.args.device)
    #
    #     samples = self.decode(z)
    #
    #     return samples
    #
    # def id2feature(self):
    #     pass
    #
    # def sample_from_x_to_y(self, xs, sub_embs):
    #     zs = self.encode(xs, sub_embs)
    #
    #     samples = self.decode(zs, reparameterize=True)
    #
    #     return samples

    # flows = {'x': (x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x, y_unlabeled,
    #                unlabeled_zz_x, unlabeled_latent_x, unlabeled_r_x),
    #          'y': (y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y, y_unlabeled,
    #                unlabeled_zz_x, unlabeled_latent_y, unlabeled_r_y)}
    def nonlinear_fusion_method(self, outputs):
        #这里 我想尝试一下 移出模组， 第一种移出非线性转换模组，第二种移出模态权重模组。
        masked_latent_list_x = []
        masked_latent_list_y = []

        non_masked_latent_list_x = []
        non_masked_latent_list_y = []

        weighted_masked_list_x = []
        weighted_masked_list_y = []

        weighted_unmasked_list_x = []
        weighted_unmasked_list_y = []

        for output, none_linear_transform, weighting in zip(outputs, self.none_linear_transforms,
                                                            self.ModalWeightingMLP):
            masked_latent_x = none_linear_transform(output['x'][1])
            masked_latent_y = none_linear_transform(output['y'][1])

            weight_x = weighting(masked_latent_x)
            weight_y = weighting(masked_latent_y)

            masked_latent_list_x.append(masked_latent_x)
            masked_latent_list_y.append(masked_latent_y)

            weighted_masked_list_x.append(weight_x)
            weighted_masked_list_y.append(weight_y)


            non_masked_latent_x = none_linear_transform(output['x'][5])
            non_masked_latent_y = none_linear_transform(output['y'][5])

            weight_unmasked_x = weighting(non_masked_latent_x)
            weight_unmasked_y = weighting(non_masked_latent_y)

            non_masked_latent_list_x.append(non_masked_latent_x)
            non_masked_latent_list_y.append(non_masked_latent_y)

            weighted_unmasked_list_x.append(weight_unmasked_x)
            weighted_unmasked_list_y.append(weight_unmasked_y)


        masked_latent_list_x = torch.stack(masked_latent_list_x, dim=1)
        masked_latent_list_y = torch.stack(masked_latent_list_y, dim=1)
        non_masked_latent_list_x = torch.stack(non_masked_latent_list_x, dim=1)
        non_masked_latent_list_y = torch.stack(non_masked_latent_list_y, dim=1)

        weighted_masked_list_x = torch.stack(weighted_masked_list_x, dim=1)
        weighted_masked_list_y = torch.stack(weighted_masked_list_y, dim=1)
        weighted_unmasked_list_x = torch.stack(weighted_unmasked_list_x, dim=1)
        weighted_unmasked_list_y = torch.stack(weighted_unmasked_list_y, dim=1)

        normalized_weighted_masked_list_x = F.softmax(weighted_masked_list_x, dim=1)
        normalized_weighted_masked_list_y = F.softmax(weighted_masked_list_y, dim=1)
        normalized_weighted_unmasked_list_x = F.softmax(weighted_unmasked_list_x, dim=1)
        normalized_weighted_unmasked_list_y = F.softmax(weighted_unmasked_list_y, dim=1)

        weighted_masked_latent_x = masked_latent_list_x * normalized_weighted_masked_list_x
        weighted_masked_latent_y = masked_latent_list_y * normalized_weighted_masked_list_y
        weighted_non_masked_latent_x = non_masked_latent_list_x * normalized_weighted_unmasked_list_x
        weighted_non_masked_latent_y = non_masked_latent_list_y * normalized_weighted_unmasked_list_y

        fused_masked_latent_x = torch.sum(weighted_masked_latent_x, dim=1)
        fused_masked_latent_y = torch.sum(weighted_masked_latent_y, dim=1)
        fused_non_masked_latent_x = torch.sum(weighted_non_masked_latent_x, dim=1)
        fused_non_masked_latent_y = torch.sum(weighted_non_masked_latent_y, dim=1)
        """ 第一方案，去除模态权重模组"""
        # for output, none_linear_transform in zip(outputs, self.none_linear_transforms):
        #     masked_latent_x = none_linear_transform(output['x'][1])
        #     masked_latent_y = none_linear_transform(output['y'][1])
        #
        #     masked_latent_list_x.append(masked_latent_x)
        #     masked_latent_list_y.append(masked_latent_y)
        #
        #
        #     non_masked_latent_x = none_linear_transform(output['x'][5])
        #     non_masked_latent_y = none_linear_transform(output['y'][5])
        #
        #     non_masked_latent_list_x.append(non_masked_latent_x)
        #     non_masked_latent_list_y.append(non_masked_latent_y)
        #
        #
        # fused_masked_latent_x = torch.cat(masked_latent_list_x, dim=1)
        # fused_masked_latent_y = torch.cat(masked_latent_list_y, dim=1)
        #
        # fused_non_masked_latent_x = torch.cat(non_masked_latent_list_x, dim=1)
        # fused_non_masked_latent_y = torch.cat(non_masked_latent_list_y, dim=1)

        """ 第二方案，去除非线性转换"""
        # for output, weighting in zip(outputs, self.ModalWeightingMLP):
        #     masked_latent_x = output['x'][1]
        #     masked_latent_y = output['y'][1]
        #
        #     weight_x = weighting(masked_latent_x)
        #     weight_y = weighting(masked_latent_y)
        #
        #     masked_latent_list_x.append(masked_latent_x)
        #     masked_latent_list_y.append(masked_latent_y)
        #
        #     weighted_masked_list_x.append(weight_x)
        #     weighted_masked_list_y.append(weight_y)
        #
        #
        #     non_masked_latent_x = output['x'][5]
        #     non_masked_latent_y = output['y'][5]
        #
        #     weight_unmasked_x = weighting(non_masked_latent_x)
        #     weight_unmasked_y = weighting(non_masked_latent_y)
        #
        #     non_masked_latent_list_x.append(non_masked_latent_x)
        #     non_masked_latent_list_y.append(non_masked_latent_y)
        #
        #     weighted_unmasked_list_x.append(weight_unmasked_x)
        #     weighted_unmasked_list_y.append(weight_unmasked_y)
        #
        #
        # masked_latent_list_x = torch.stack(masked_latent_list_x, dim=1)
        # masked_latent_list_y = torch.stack(masked_latent_list_y, dim=1)
        # non_masked_latent_list_x = torch.stack(non_masked_latent_list_x, dim=1)
        # non_masked_latent_list_y = torch.stack(non_masked_latent_list_y, dim=1)
        #
        # weighted_masked_list_x = torch.stack(weighted_masked_list_x, dim=1)
        # weighted_masked_list_y = torch.stack(weighted_masked_list_y, dim=1)
        # weighted_unmasked_list_x = torch.stack(weighted_unmasked_list_x, dim=1)
        # weighted_unmasked_list_y = torch.stack(weighted_unmasked_list_y, dim=1)
        #
        # normalized_weighted_masked_list_x = F.softmax(weighted_masked_list_x, dim=1)
        # normalized_weighted_masked_list_y = F.softmax(weighted_masked_list_y, dim=1)
        # normalized_weighted_unmasked_list_x = F.softmax(weighted_unmasked_list_x, dim=1)
        # normalized_weighted_unmasked_list_y = F.softmax(weighted_unmasked_list_y, dim=1)
        #
        # weighted_masked_latent_x = masked_latent_list_x * normalized_weighted_masked_list_x
        # weighted_masked_latent_y = masked_latent_list_y * normalized_weighted_masked_list_y
        # weighted_non_masked_latent_x = non_masked_latent_list_x * normalized_weighted_unmasked_list_x
        # weighted_non_masked_latent_y = non_masked_latent_list_y * normalized_weighted_unmasked_list_y
        #
        # fused_masked_latent_x = torch.sum(weighted_masked_latent_x, dim=1)
        # fused_masked_latent_y = torch.sum(weighted_masked_latent_y, dim=1)
        # fused_non_masked_latent_x = torch.sum(weighted_non_masked_latent_x, dim=1)
        # fused_non_masked_latent_y = torch.sum(weighted_non_masked_latent_y, dim=1)


        flow = [fused_masked_latent_x, fused_masked_latent_y, fused_non_masked_latent_x, fused_non_masked_latent_y]
        return flow


    # flow = [fused_masked_latent_x, fused_masked_latent_y, fused_non_masked_latent_x, fused_non_masked_latent_y]
    def semantic_calibration_loss(self, outputs_second):
        total_loss = 0

        fused_masked_latent_x = outputs_second[0]
        fused_masked_latent_y = outputs_second[1]
        fused_non_masked_latent_x = outputs_second[2]
        fused_non_masked_latent_y = outputs_second[3]

        total_loss += MMI(fused_masked_latent_x, fused_non_masked_latent_y)
        total_loss += MMI(fused_masked_latent_y, fused_non_masked_latent_x)

        return total_loss


    def active_learning_weighting(self, sub_embs):
        with torch.no_grad():
            full_latent_list = []
            for sub_emb, subgenerator in zip(sub_embs, self.subgenerators):
                full_zz = subgenerator.encode(sub_emb)
                full_latent = subgenerator.reparameterize(*full_zz)
                full_latent_list.append(full_latent)

            weighting_list = []

            for latent_emb, none_linear_transform, weighting in zip(full_latent_list, self.none_linear_transforms, self.ModalWeightingMLP):
                none_linear_latent = none_linear_transform(latent_emb)
                weight = weighting(none_linear_latent).squeeze(-1)
                # weight = weighting(none_linear_latent)
                weighting_list.append(weight)

            weighting_list = torch.stack(weighting_list, dim=1)
            weighting_list = F.softmax(weighting_list, dim=1)

        return weighting_list

    def forward(self, train_links, sub_embs, joint_emb):

        sub_embs = [embs for embs in sub_embs if embs is not None]
        self.subdecoders[0].ent_embs = sub_embs[0]

        outputs = [subgenerator(self.args, embs, train_links)
                   for embs, subgenerator in zip(sub_embs, self.subgenerators)]

        outputs_second = self.nonlinear_fusion_method(outputs)

        distribution_match_loss = self.distribution_match_loss(outputs)

        uni_reconstruction_loss = self.reconstruction_loss(outputs)
        # prior_reconstruction_loss = self.prior_reconstruction_loss(
        #     outputs, train_links)
        # post_reconstruction_loss = self.post_reconstruction_loss(outputs, joint_emb, train_links)

        semantic_calibration_loss = self.semantic_calibration_loss(outputs_second)
        # outputs_third = self.masked_decode_fuction_layer(outputs_second)

        joint_reconstruction_loss = self.distribution_reconstruction_loss(outputs)
        # distribution_post_reconstruction_loss = self.distribution_post_reconstruction_loss(outputs_third, joint_emb,
        #                                                                                    train_links)

        print("Dist Match Loss: %.3f; Semantic Calibrate: %.3f; Reconstruct Loss: %.3f" %
              (distribution_match_loss.item()*0.5, semantic_calibration_loss, (uni_reconstruction_loss.item()+uni_reconstruction_loss.item()+joint_reconstruction_loss.item())))

        return uni_reconstruction_loss + semantic_calibration_loss + distribution_match_loss*0.5 + joint_reconstruction_loss
