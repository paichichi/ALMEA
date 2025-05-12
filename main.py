import math
import os
import os.path as osp
import pickle

import random
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
from collections import defaultdict

from transformers import get_cosine_schedule_with_warmup

from ACS_ADMM import soft_threshold_L21, optimize_ds3_regularized, find_representatives_fast
from config import config
from models.RANKER import RANKER, Discriminator
from src.tensorBoardManager import TensorBoardManager
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
# add model here
from models import MCLEA

from src.distributed_utils import init_distributed_mode, dist_pdb, is_main_process, reduce_value, cleanup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import scipy
import gc
import copy


class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.ranker = None
        self.datapath = edict()
        self.args = args
        self.writer = writer
        self.logger = logger
        self.rank = rank
        self.scaler = GradScaler()

        self.model_list = []
        set_seed(args.random_seed)

        self.right_ents = []
        self.left_ents = []
        self.total_ills = None
        self.data_init()
        self.model_choice()

        self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
        self.model_list = [self.model]

        train_epoch_1_stage = self.args.epoch
        self.optim_init(self.args, total_epoch=train_epoch_1_stage)
        self.add_count = 0


        self.selected_pairs = []
        self.train_losses = []


    def model_choice(self):
        self.model = MCLEA(self.KGs, self.args)
        self.model = self._load_model(self.model)


    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger,
                                                                                                           self.args)

        self.total_ills = len(self.train_set) + len(self.test_ill_)
        self.left_ents = self.KGs['left_ents']
        self.right_ents = self.KGs['right_ents']
        
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).cuda()

        self.eval_sampler = None

    def optim_init(self, opt, accumulation_step=None):

        freeze_part = []
        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, accumulation_step)


    def dataloader_init(self, train_set=None, eval_set=None, test_set=None):
        bs = self.args.batch_size
        # 将张量转换成numpy数组
        collator = Collator_base(self.args)
        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        if train_set is not None:
            self.train_dataloader = self._dataloader(train_set, bs, collator)
        if test_set is not None:
            self.test_dataloader = self._dataloader(test_set, bs, collator)
        if eval_set is not None:
            self.eval_dataloader = self._dataloader(eval_set, bs, collator)

    def _load_model(self, model, model_name=None):
        model.to(self.args.device)
        return model

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.weight = [1, 1, 1, 1, 1, 1]
        self.loss_weight = [1, 1]
        self.loss_item = 99999.
        self.step = 1
        self.epoch = 0
        self.new_links = []
        self.best_model_wts = None

        self.best_mrr = 0
        self.early_stop_init = 500
        self.early_stop_count = self.early_stop_init
        self.stage = 0

        # budgets = math.ceil(len(self.test_ill_) * (0.3 - self.args.data_rate))
        # logger.info(f'The budget sizes is {budgets}')
        # self.budget_per_round = math.ceil(budgets / 5)
        # logger.info(f"The budget sizes per cycles is {self.budget_per_round}")

        # total_epoch = self.args.epoch + 300 * (self.args.CYCLES - 1)
        with tqdm(total=self.args.epoch + self.args.epoch_per_CYCLES*self.args.CYCLES) as _tqdm:
            for cycle in range(self.args.CYCLES+1):
                if cycle != 0:
                    self.logger.info("-----------------------------------------------------------------------")
                    self.logger.info(f'Incremental Training Start {cycle}')
                else:
                    self.logger.info("-----------------------------------------------------------------------")
                    self.logger.info("Training Start")

                for i in range(self.args.epoch):
                    self.epoch = i

                    curr_loss, sub_embs = self.train(_tqdm)
                    self.train_losses.append(curr_loss)

                    self.loss_log.update(self.curr_loss)
                    self.loss_item = self.loss_log.get_loss()
                    _tqdm.set_description(
                        f'Train |  Ep [{self.epoch}/{self.args.epoch}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                    self.update_loss_log()
                    if (i + 1) % self.args.eval_epoch == 0:
                        self.eval()
                    _tqdm.update(1)

                self.logger.info("-----------------------------------------------------------------------")

                self.eval()
                if cycle != self.args.CYCLES:
                    self.logger.info(f"Active Learning Phase {cycle+1}")
                    self.active_learning_sampling(_tqdm, sub_embs, cycle)

                    self.logger.info('END')
                    self.args.epoch = self.args.epoch_per_CYCLES
                else:
                    break
                break

        name = self._save_name_define()
        # if self.best_model_wts is not None:
        #     self.logger.info("load from the best model before final testing ... ")
        #     self.model.load_state_dict(self.best_model_wts)
        self.test(save_name=f"{name}_test_ep{self.args.epoch}")

        if self.rank == 0:
            self.logger.info(f"min loss {self.loss_log.get_min_loss()}")

    def _save_name_define(self):
        prefix = ""
        if self.args.dist:
            prefix = f"dist_{prefix}"
        if self.args.il:
            prefix = f"il{self.args.epoch - self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name

    def _dataloader(self, train_set, batch_size, collator):

        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def pairwise_kendall_tau_distance(self, x, y):
        def kendall(x, y):
            n = x.shape[-1]
            def sub_pairs(z):
                z_diff = z.unsqueeze(-1) - z.unsqueeze(-2)
                return z_diff.sign()
            concordant_pairs = (sub_pairs(x) * sub_pairs(y)).sum(dim=(-2, -1))
            return 1 - concordant_pairs.div(n * (n - 1))
        N, M = x.shape[0], y.shape[0]
        x_expanded = x.unsqueeze(1).expand(N, M, -1)
        y_expanded = y.unsqueeze(0).expand(N, M, -1)
        distance_matrix = kendall(x_expanded, y_expanded)
        tau_distance = 1 + distance_matrix

        return tau_distance

    def dissimilarity_loss_score(self, dissimilarity_matrix, sigma):

        dissimilarity_matrix = dissimilarity_matrix.clone()
        dissimilarity_matrix.fill_diagonal_(float('inf'))

        min_dissimilarity_ji = dissimilarity_matrix.min(dim=1).values
        max_min_dissimilarity_jk = min_dissimilarity_ji.max()

        dissimilarity_scores = 1 + (sigma - 1) * (min_dissimilarity_ji / max_min_dissimilarity_jk)

        return dissimilarity_scores

    def diversity_loss_score(self, KT_matrix, sigma):

        min_KT_ji = KT_matrix.min(dim=1).values
        max_min_KT_jk = min_KT_ji.max()

        c_diversity = sigma - (sigma - 1) * (min_KT_ji / max_min_KT_jk)
        return c_diversity

    def euclidean_distance(self, x, y):

        diff = x[:, None, :] - y[None, :, :]

        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=2))

        return dist_matrix

    def min_max_Normalization(self, matrix):
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)

        matrix_min = matrix.min()
        matrix_max = matrix.max()
        matrix_normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        return matrix_normalized

    def active_learning_sampling(self, _tqdm, sub_embs, cycle):
        max_iteration = 1000
        sigma = 3
        with torch.no_grad():
            weighting_list = self.model.geea.active_learning_weighting(sub_embs)
            final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
            target_distance = pairwise_distances(final_emb[self.left_ents], final_emb[self.right_ents])
            target_distance = 1 - csls_sim(1 - target_distance, self.args.csls_k)

            sorted_distances_x, sorted_indices_x = torch.sort(target_distance, descending=False)
            sorted_distances_y, sorted_indices_y = torch.sort(target_distance.t(), descending=False)

            self.selected_pairs = []
            min_distances = []
            left_ents = []
            right_ents = []
            for idx in range(len(self.left_ents)):
                x_to_y_idx = sorted_indices_x[idx, 0]
                y_to_x_idx = sorted_indices_y[x_to_y_idx, 0]
                if y_to_x_idx == idx:
                    x = self.left_ents[idx]
                    y = self.right_ents[x_to_y_idx]

                    min_distance = sorted_distances_x[idx, 0].item()

                    self.selected_pairs.append((x, y, min_distance))
                    left_ents.append(x)
                    right_ents.append(y)
                    min_distances.append(min_distance)
            min_distances = np.array(min_distances)
            x_index = self.train_set[:, 0]
            y_index = self.train_set[:, 1]

            A_matrix_x = pairwise_distances(final_emb[left_ents], final_emb[left_ents])
            A_matrix_x = 1 - csls_sim(1 - A_matrix_x, self.args.csls_k)
            A_matrix_x = self.min_max_Normalization(A_matrix_x)

            A_matrix_y = pairwise_distances(final_emb[right_ents], final_emb[right_ents])
            A_matrix_y = 1 - csls_sim(1 - A_matrix_y, self.args.csls_k)
            A_matrix_y = self.min_max_Normalization(A_matrix_y)

            A_matrix_xy_pair = (A_matrix_x + A_matrix_y) / 2

            space_distance_x = pairwise_distances(final_emb[left_ents], final_emb[x_index])
            space_distance_x = 1 - csls_sim(1 - space_distance_x, self.args.csls_k)
            space_distance_x = self.min_max_Normalization(space_distance_x)
            KT_score_x = space_distance_x

            space_distance_y = pairwise_distances(final_emb[right_ents], final_emb[y_index])
            space_distance_y = 1 - csls_sim(1 - space_distance_y, self.args.csls_k)
            space_distance_y = self.min_max_Normalization(space_distance_y)
            KT_score_y = space_distance_y

            KT_score_pair = (KT_score_x + KT_score_y) / 2

            C_diversity_score = self.diversity_loss_score(KT_score_pair, sigma)
            C_diversity_score = torch.diag(C_diversity_score)

            Z_C = optimize_ds3_regularized(A_matrix_xy_pair, self.args.lambda_, self.args.rho, C_diversity_score, max_iteration,
                                        self.args.early_stop_threshold)
            Z_C_Ind = find_representatives_fast(Z_C)

            list_result = min_distances[Z_C_Ind.cpu()]
            sorted_indices = np.argsort(list_result)

            count = math.floor(self.total_ills * self.args.tau3)

            Z_C_Ind_list = Z_C_Ind[sorted_indices[:count]].cpu().numpy().tolist()
            selected_pairs_subset = [self.selected_pairs[i] for i in Z_C_Ind_list]
            for pair in selected_pairs_subset:
                self.train_set = np.vstack([self.train_set, np.array(pair[:2])])

            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)

            self.left_ents = [x for x in self.left_ents if x not in [pair[0] for pair in selected_pairs_subset]]
            self.right_ents = [y for y in self.right_ents if y not in [pair[1] for pair in selected_pairs_subset]]

            print(f'Found: {count} out of {len(Z_C_Ind)} pairs that are most informative and representative. '
                  f'currently left: {len(self.left_ents)}, currently right: {len(self.right_ents)}')

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps

        for batch in self.train_dataloader:
            loss, output, sub_embs = self.model(batch)
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()

            self.step += 1

            if not self.args.dist or is_main_process():
                curr_loss += loss.item()
                self.output_statistic(loss, output)

            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()

                if not self.args.dist or is_main_process():
                    self.lr = self.scheduler.get_last_lr()[-1]
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)
                return curr_loss, sub_embs

        return curr_loss, sub_embs

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'weight' in output and output['weight'] is not None:
            self.weight = output['weight']

    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)

        if self.weight is not None:
            weight_dic = {}
            weight_dic["img"] = self.weight[0]
            weight_dic["attr"] = self.weight[1]
            weight_dic["rel"] = self.weight[2]
            weight_dic["graph"] = self.weight[3]
            self.writer.add_scalars("modal_weight", weight_dic, self.step)

        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    def test(self, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right

        self.model.eval()
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(test_left, test_right, last_epoch=True, save_name=save_name)

    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None):
        with torch.no_grad():
            w_normalized = F.softmax(self.model.multimodal_encoder.fusion.weight.reshape(-1), dim=0)
            if self.rank == 0:
                appdx = ""
                self.logger.info(
                    f"weight_raw:[img_{w_normalized[0]:.3f}]-[attr_{w_normalized[1]:.3f}]-[rel_{w_normalized[2]:.3f}]-[graph_{w_normalized[3]:.3f}]{appdx}")

            final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
        top_k = [1, 10, 50]

        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
        if self.args.distance == 2:
            distance = pairwise_distances(final_emb[test_left], final_emb[test_right])

        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        if last_epoch:
            to_write = []
            test_left_np = test_left.cpu().numpy()
            test_right_np = test_right.cpu().numpy()
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if last_epoch:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                                 test_right_np[indices[1]], test_right_np[indices[2]]])

        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1

        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 6)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 6)
        avg_acc = [round((l + r) / 2, 6) for l, r in zip(acc_l2r, acc_r2l)]
        avg_mr = round((mean_l2r + mean_r2l) / 2, 6)
        avg_mrr = round((mrr_l2r + mrr_r2l) / 2, 6)
        gc.collect()

        Loss_out = f", Loss = {self.loss_item:.4f}"
        if self.rank == 0:
            self.logger.info(
                f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
            self.logger.info(
                f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
            self.logger.info(
                f"Ep {self.epoch} | Average: acc of top {top_k} = {avg_acc}, avg mr = {avg_mr:.3f}, avg mrr = {avg_mrr:.3f}{Loss_out}")
            self.early_stop_count -= 1

        if mrr_l2r > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(f"Best model update in Ep {self.epoch}: MRR from [{max(self.loss_log.acc)}] --> [{mrr_l2r}] ... ")
            self.loss_log.update_acc(mrr_l2r)

if __name__ == '__main__':
    # 加载预设参数

    gc.collect()
    torch.cuda.empty_cache()
    cfg = config()
    cfg.get_args()
    cfgs = cfg.update_configs()

    set_seed(cfgs.random_seed)

    writer, logger = None, None
    rank = cfgs.rank

    if rank == 0:
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    runner.run()

    logger.info("done!")
