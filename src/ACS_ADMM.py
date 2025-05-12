import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


# def compute_lambda_l21(D, strategy='mean', factor=0.3):
#     row_l2_norms = torch.norm(D, p=2, dim=1)
#
#     if strategy == 'max':
#         lambda_value = torch.max(row_l2_norms).item() * factor
#     elif strategy == 'mean':
#         lambda_value = torch.mean(row_l2_norms).item() * factor
#     elif strategy == 'median':
#         lambda_value = torch.median(row_l2_norms).item() * factor
#     elif strategy == 'quantile':
#         quantile_value = 0.75
#         lambda_value = torch.quantile(row_l2_norms, quantile_value).item() * factor
#     else:
#         raise ValueError("Invalid strategy. Use 'max', 'mean', 'median', or 'quantile'.")
#
#     return lambda_value


def soft_threshold_L21(Z_old, lambda_, D_score):
    norm_Z_old = torch.linalg.norm(Z_old, dim=1)

    adjusted_lambda = lambda_ * (D_score.diagonal())
    factor = torch.clamp(norm_Z_old - adjusted_lambda, min=0)

    C_new = Z_old * (factor / (norm_Z_old + 1e-8)).unsqueeze(1)

    return C_new


def project_onto_simplex(C_new):
    C_new_T = C_new.T

    sorted_C_new, _ = torch.sort(C_new_T, descending=True, dim=1)
    cumsum_C_new = torch.cumsum(sorted_C_new, dim=1)

    rho = torch.arange(1, C_new_T.size(1) + 1, device=C_new.device)
    rho_term = (sorted_C_new - (cumsum_C_new - 1) / rho.unsqueeze(0)) > 0
    rho_sum = torch.sum(rho_term, dim=1)

    theta = (cumsum_C_new.gather(1, rho_sum.unsqueeze(1) - 1) - 1) / rho_sum.float().unsqueeze(1)

    projected_C_new_T = torch.clamp(C_new_T - theta, min=0)

    col_sums = projected_C_new_T.sum(dim=1, keepdim=True)
    projected_C_new_T = projected_C_new_T / col_sums

    projected_C_new = projected_C_new_T.T

    return projected_C_new


def compute_convergence_error(matrix_current, matrix_reference):
    return torch.mean(torch.abs(matrix_current - matrix_reference))


def find_representatives_fast(C, ratio=0.1):
    r = torch.max(C.abs(), dim=1)[0]

    threshold = ratio * torch.max(r)

    sInd = torch.nonzero(r >= threshold, as_tuple=False).squeeze()

    v = torch.norm(C[sInd], p=2, dim=1)
    sInd = sInd[torch.argsort(v, descending=True)]

    return sInd


def optimize_ds3_regularized(A, lambda_, rho, D_score, max_iteration, early_stop_threshold):
    CFD = torch.ones(A.shape[0], device=A.device)
    num_R, num_C = A.shape

    P_old = torch.ones((num_R, num_C), device=A.device)
    mu = torch.ones((num_R, num_C), device=A.device)

    start_time_admm = time.time()

    for k in range(1, max_iteration + 1):

        C_new = soft_threshold_L21(P_old - (mu + A) / rho, lambda_ / rho * CFD, D_score)

        P_new = project_onto_simplex(C_new + mu / rho)

        mu.add_(rho * (C_new - P_new))

        err1 = compute_convergence_error(C_new, P_new)
        err2 = compute_convergence_error(P_old, P_new)

        if k % 100 == 0:
            end_time_admm = time.time()
            time_admm = end_time_admm - start_time_admm
            print(f" update Z and C time usage: {round(time_admm, 2)} seconds")
            print(
                f'||Z-C||= {err1:.2e}, ||C1-C2||= {err2:.2e}, repNum = {len(find_representatives_fast(P_new))}, iteration = {k}')
            start_time_admm = time.time()

        if err1 <= early_stop_threshold and err2 <= early_stop_threshold:
            break

        P_old = P_new

    return P_new