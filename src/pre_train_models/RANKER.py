import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from src.data_processing.utils import csls_sim

class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r,z):
        z = torch.cat([z, r], 1)
        return self.net(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)

        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.fc4 = nn.Linear(512, output_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        return x


class RANKER(nn.Module):
    def __init__(self, KGs, args):
        super().__init__()
        self.KGs = KGs
        self.args = args
        # self.target_distance = target_distance
        # self.train_set = data_set['train_set']
        # self.final_emb = data_set['final_emb']
        # self.test_left = data_set['test_left']
        # self.test_right = data_set['test_right']
        self.mlp_left = MLP(300, 5000).to(self.args.device) # totall 12846, test_set 10277
        self.mlp_right = MLP(300, 5000).to(self.args.device) # totall 12846, test_set 10277

    def forward(self, target_distance, data_set):

        train_set = data_set['train_set']
        final_emb = data_set['final_emb'].to(self.args.device)
        test_left = data_set['test_left']
        test_right = data_set['test_right']
        target_distance = target_distance.to(self.args.device)

        pred_distance_y = self.mlp_left(final_emb[test_left])
        pred_distance_x = self.mlp_right(final_emb[test_right])
        pred_distance = pred_distance_y + pred_distance_x.T
        pred_distance = 1 - csls_sim(1 - pred_distance, self.args.csls_k)

        rankloss = self.rankLoss(pred_distance, target_distance, test_left,test_right, 2)
        print('Rank loss: {}'.format(rankloss))

        return rankloss

    def rankLoss(self, pred_distance, target_distance, test_left,test_right, weight=2):
        mse_loss =.0
        mse_loss += F.mse_loss(pred_distance,target_distance)
        relative_loss = .0

        for idx in range(len(test_left)):
            values, indices = torch.sort(pred_distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            relative_loss += rank / len(test_left)
        for idx in range(len(test_right)):
            values, indices = torch.sort(pred_distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            relative_loss += rank / len(test_right)

        total_samples = len(test_left) + len(test_right)
        average_relative_loss = relative_loss / total_samples

        return mse_loss + weight * average_relative_loss
