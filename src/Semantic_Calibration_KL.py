import torch
import torch.nn as nn
import sys

EPS = sys.float_info.epsilon


def joint_probability_matrix(modality):
    relu = nn.ReLU()
    modality =  relu(modality)
    tensor_sum = modality.sum(dim=-1, keepdim=True) + EPS  # 避免除以零
    prob = modality / tensor_sum
    return prob

def semantic_calibration(source, target, scale_factor=100):
    source = joint_probability_matrix(source)
    target = joint_probability_matrix(target)

    kld_loss = source * (torch.log(source + EPS) - torch.log(target + EPS))
    kld_loss += target * (torch.log(target + EPS) - torch.log(source + EPS))

    return kld_loss.mean() * scale_factor
