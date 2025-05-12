import torch
import torch.nn as nn
import torch.nn.functional as F

from .Semantic_Calibration_KL import semantic_calibration


class GenerativeVAE(nn.Module):

    def __init__(self, args, in_dim, hidden_dims, latent_dim=None, **kwargs):
        super(GenerativeVAE, self).__init__()
        self.args = args

        if latent_dim:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = hidden_dims[-1]

        modules = []

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
        x = self.decoder(z)

        return x

    # def masked_decode(self, z, reparameterize=False):
    #     if reparameterize:
    #         z = self.reparameterize(*z)
    #     z = self.masked_decoder_input(z)
    #     x = self.masked_decoder(z)
    #
    #     return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, args, embs, train_links):

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

        flows = {'x': (
            x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x),
                 'y': (
            y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y)}

        return flows

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


class ALMEA(nn.Module):

    def __init__(self, args, kgs, concrete_features, sub_dims, joint_dim, ent_embs, fusion_layer):
        self.num_epochs = 0
        self.latent_z_first_iter = None
        self.latent_z_last_iter = None
        self.labels_first_iter = None
        self.labels_last_iter = None
        super().__init__()
        self.args = args
        self.kgs = kgs

        self.latent_dim = sub_dims[0]

        self.subgenerators = []
        self.none_linear_transforms = []
        self.ModalWeightingMLP = []

        self.num_none_concrete_feature = 0

        for i, sub_dim, concrete_feature in zip(range(len(sub_dims)), sub_dims, concrete_features):
            if concrete_feature is not None:
                subgenerator = GenerativeVAE(args=args,
                                    in_dim=sub_dim,
                                    hidden_dims=[sub_dim, ] * args.num_layers,
                                    latent_dim=sub_dim)
                self.subgenerators.append(subgenerator)

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

                mlp = nn.Sequential(
                    nn.Linear(sub_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.ReLU(),
                    nn.Linear(300, 1)
                )
                self.ModalWeightingMLP.append(mlp)

            else:
                self.num_none_concrete_feature += 1

        self.masked_decoder_input = nn.Linear(300, 300)
        self.masked_decoder = nn.Sequential(nn.Linear(300, 300),
                                            nn.Tanh(),
                                            nn.Dropout(0.5),
                                            nn.BatchNorm1d(300),
                                            nn.Linear(300, 300),)
        self.subgenerators = nn.ModuleList(self.subgenerators)
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

    def re_fusion(self, sub_embs):
        sub_embs = sub_embs + [None, ] * self.num_none_concrete_feature
        return self.fusion_layer(*sub_embs)

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

    # flows = {'x': (x, latent_x, r_x, x_mask, zz_x, unmasked_latent_x, full_zz_y, full_r_x, y_unlabeled,
    #                unlabeled_zz_x, unlabeled_latent_x, unlabeled_r_x),
    #          'y': (y, latent_y, r_y, y_mask, zz_y, unmasked_latent_y, full_zz_y, full_r_y, y_unlabeled,
    #                unlabeled_zz_x, unlabeled_latent_y, unlabeled_r_y)}
    def nonlinear_fusion_method(self, outputs):
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

        flow = [fused_masked_latent_x, fused_masked_latent_y, fused_non_masked_latent_x, fused_non_masked_latent_y]
        return flow


    # flow = [fused_masked_latent_x, fused_masked_latent_y, fused_non_masked_latent_x, fused_non_masked_latent_y]
    def semantic_calibration_loss(self, outputs_second):
        total_loss = 0

        fused_masked_latent_x = outputs_second[0]
        fused_masked_latent_y = outputs_second[1]
        fused_non_masked_latent_x = outputs_second[2]
        fused_non_masked_latent_y = outputs_second[3]

        total_loss += semantic_calibration(fused_masked_latent_x, fused_non_masked_latent_y)
        total_loss += semantic_calibration(fused_masked_latent_y, fused_non_masked_latent_x)

        return total_loss


    # def active_learning_weighting(self, sub_embs):
    #     with torch.no_grad():
    #         full_latent_list = []
    #         for sub_emb, subgenerator in zip(sub_embs, self.subgenerators):
    #             full_zz = subgenerator.encode(sub_emb)
    #             full_latent = subgenerator.reparameterize(*full_zz)
    #             full_latent_list.append(full_latent)
    #
    #         weighting_list = []
    #
    #         for latent_emb, none_linear_transform, weighting in zip(full_latent_list, self.none_linear_transforms, self.ModalWeightingMLP):
    #             none_linear_latent = none_linear_transform(latent_emb)
    #             weight = weighting(none_linear_latent).squeeze(-1)
    #             weighting_list.append(weight)
    #
    #         weighting_list = torch.stack(weighting_list, dim=1)
    #         weighting_list = F.softmax(weighting_list, dim=1)
    #
    #     return weighting_list

    def forward(self, train_links, sub_embs, joint_emb):

        sub_embs = [embs for embs in sub_embs if embs is not None]

        outputs = [subgenerator(self.args, embs, train_links)
                   for embs, subgenerator in zip(sub_embs, self.subgenerators)]
        outputs_second = self.nonlinear_fusion_method(outputs)

        distribution_match_loss = self.distribution_match_loss(outputs)
        uni_reconstruction_loss = self.reconstruction_loss(outputs)
        semantic_calibration_loss = self.semantic_calibration_loss(outputs_second)
        joint_reconstruction_loss = self.distribution_reconstruction_loss(outputs)

        print("Dist Match Loss: %.3f; Semantic Calibrate: %.3f; Reconstruct Loss: %.3f" %
              (distribution_match_loss.item()*0.5, semantic_calibration_loss, (uni_reconstruction_loss.item()+joint_reconstruction_loss.item())))

        return uni_reconstruction_loss + semantic_calibration_loss + distribution_match_loss*0.5 + joint_reconstruction_loss
