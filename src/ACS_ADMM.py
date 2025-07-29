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

    Z_new = Z_old * (factor / (norm_Z_old + 1e-8)).unsqueeze(1)

    return Z_new


def project_onto_simplex(P_old):
    P_new_T = P_old.T

    sorted_P_new, _ = torch.sort(P_new_T, descending=True, dim=1)
    cumsum_P_new = torch.cumsum(sorted_P_new, dim=1)

    rho = torch.arange(1, P_new_T.size(1) + 1, device=P_old.device)
    rho_term = (sorted_P_new - (cumsum_P_new - 1) / rho.unsqueeze(0)) > 0
    rho_sum = torch.sum(rho_term, dim=1)

    theta = (cumsum_P_new.gather(1, rho_sum.unsqueeze(1) - 1) - 1) / rho_sum.float().unsqueeze(1)

    projected_P_new_T = torch.clamp(P_new_T - theta, min=0)

    col_sums = projected_P_new_T.sum(dim=1, keepdim=True)
    projected_P_new_T = projected_P_new_T / col_sums

    projected_P_new = projected_P_new_T.T

    return projected_P_new

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

    total_time_Z = 0
    total_time_P = 0
    total_time_mu = 0
    total_time_err1 = 0
    total_time_err2 = 0

    for k in range(1, max_iteration + 1):
        start_total_time_iteration = time.time()

        start_time_Z = time.time()
        Z_new = soft_threshold_L21(P_old - (mu + A) / rho, lambda_ / rho * CFD, D_score)
        end_time_Z = time.time()
        total_time_Z += end_time_Z - start_time_Z

        start_time_P = time.time()
        P_new = project_onto_simplex(Z_new + mu / rho)
        end_time_P = time.time()
        total_time_P += end_time_P - start_time_P

        start_time_mu = time.time()
        mu.add_(rho * (Z_new - P_new))
        end_time_mu = time.time()
        total_time_mu += end_time_mu - start_time_mu

        start_time_err1 = time.time()
        err1 = compute_convergence_error(Z_new, P_new)
        end_time_err1 = time.time()
        total_time_err1 += end_time_err1 - start_time_err1

        start_time_err2 = time.time()
        err2 = compute_convergence_error(P_old, P_new)
        end_time_err2 = time.time()
        total_time_err2 += end_time_err2 - start_time_err2

        if k % 100 == 0:
            end_time_admm = time.time()
            time_admm = end_time_admm - start_time_admm
            print(f"Iteration {k}:")
            print(f"  Accumulated Update Z time: {total_time_Z:.4f} seconds")
            print(f"  Accumulated Update P time: {total_time_P:.4f} seconds")
            print(f"  Accumulated Update mu time: {total_time_mu:.4f} seconds")
            print(f"  Accumulated Update err 1: {err1:.4f} seconds")
            print(f"  Accumulated Update err 2: {err2:.4f} seconds")
            print(f"  Total time for Z and C update: {time_admm:.4f} seconds")
            print(f"  ||Z-P||= {err1:.2e}, ||Z_new-Z_old||= {err2:.2e}, repNum = {len(find_representatives_fast(P_new))}")
            start_time_admm = time.time()

        if err1 <= early_stop_threshold and err2 <= early_stop_threshold:
            break

        P_old = P_new

    return P_new