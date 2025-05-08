import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def plot_results(data, sInd):
    data = data.cpu()
    sInd = sInd.cpu()
    # 使用T-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    colors = np.array(['blue'] * len(data))  # 初始化所有点为蓝色
    colors[list(sInd)] = 'red'  # 将被选中的索引改为红色

    # 绘制结果
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=colors,  # 根据颜色数组进行着色
        palette={'blue': 'blue', 'red': 'red'},  # 定义颜色调色板
        legend=None,  # 如果不需要图例可以设置为 None
        alpha=0.8
    )
    plt.title(f"T-SNE projection of {len(data)} points with selected {len(sInd)} entities in red")
    plt.show()


def compute_lambda_l21(D, strategy='mean', factor=0.3):
    # 输入：D 是一个矩阵
    # 输出：根据 l21 范数策略计算的正则化参数 λ

    # 计算每行的 l2 范数
    row_l2_norms = torch.norm(D, p=2, dim=1)

    if strategy == 'max':
        # 使用行 l2 范数的最大值
        lambda_value = torch.max(row_l2_norms).item() * factor
    elif strategy == 'mean':
        # 使用行 l2 范数的均值
        lambda_value = torch.mean(row_l2_norms).item() * factor
    elif strategy == 'median':
        # 使用行 l2 范数的中位数
        lambda_value = torch.median(row_l2_norms).item() * factor
    elif strategy == 'quantile':
        # 使用行 l2 范数的特定分位数
        quantile_value = 0.75  # 比如，使用 75% 分位数
        lambda_value = torch.quantile(row_l2_norms, quantile_value).item() * factor
    else:
        raise ValueError("Invalid strategy. Use 'max', 'mean', 'median', or 'quantile'.")

    return lambda_value


# 收缩函数
# Z = shrink_L1Lp(C1 - (Lambda + D) / mu, rho / mu * CFD)
# def shrink_L1Lp(C1, lambda_):
#     # 计算矩阵中每行的 L2 范数，所以 norm_C1是每一行的L2范数
#     norm_C1 = torch.linalg.norm(C1, dim=1)
#
#     # 对每一行的范数进行 缩放，缩放 lambda_个单位， 并截断负值为0
#     factor = torch.clamp(norm_C1 - lambda_, min=0)
#
#     # 将缩放比例归一化，每行的缩放比例控制在[0,1]之间， 所以C2会是C1缩放了lambda_个单位的结果
#     C2 = C1 * (factor/norm_C1).unsqueeze(1)
#     return C2
def shrink_L1Lp(C1, lambda_, C):
    """
    对 C1 进行 L1-Lp 范数收缩，结合对角矩阵 C 的额外惩罚项。

    参数:
    - C1: 待收缩的矩阵。
    - lambda_: 正则化参数。
    - C: 对角矩阵，用于对每行的惩罚进行缩放。

    返回:
    - C2: 收缩后的矩阵。
    """
    # 计算每行的 L2 范数
    norm_C1 = torch.linalg.norm(C1, dim=1)

    # 计算缩放因子，结合对角矩阵 C 的影响
    adjusted_lambda = lambda_ * (C.diagonal())  # 结合C的对角线值，调整lambda_
    factor = torch.clamp(norm_C1 - adjusted_lambda, min=0)

    # 对每行进行缩放
    C2 = C1 * (factor / (norm_C1 + 1e-8)).unsqueeze(1)  # 加入1e-8防止除以零

    return C2


# 闭式解求解器 C2 = solver_BCLS_closedForm(Z + Lambda / mu)
def solver_BCLS_closedForm(V):
    # 输入：V 是一个矩阵
    # 输出：将 V 的每一列投影到概率单纯形上，确保每一列的和为 1，且所有元素非负

    # 转置 V，以便将列视为行进行操作
    V_T = V.T

    # 投影每一行（原来是每一列）
    sorted_V, _ = torch.sort(V_T, descending=True, dim=1)
    cumsum_V = torch.cumsum(sorted_V, dim=1)

    # 高效地计算 rho 值
    rho = torch.arange(1, V_T.size(1) + 1, device=V.device)
    rho_term = (sorted_V - (cumsum_V - 1) / rho.unsqueeze(0)) > 0
    rho_sum = torch.sum(rho_term, dim=1)

    # 计算 theta
    theta = (cumsum_V.gather(1, rho_sum.unsqueeze(1) - 1) - 1) / rho_sum.float().unsqueeze(1)

    # 执行投影操作
    projected_V_T = torch.clamp(V_T - theta, min=0)

    # 归一化以确保每一行（原来的列）的和为 1
    col_sums = projected_V_T.sum(dim=1, keepdim=True)
    projected_V_T = projected_V_T / col_sums

    # 转置回原始方向
    projected_V = projected_V_T.T

    return projected_V


# 误差系数计算
def error_coef(Z, C):
    return torch.mean(torch.abs(Z - C))


# 查找代表点
def find_representatives_fast(C, ratio=0.1):
    # 计算每一行的无穷范数,找出每一行中的最大指示值
    r = torch.max(C.abs(), dim=1)[0]

    # 计算阈值
    threshold = ratio * torch.max(r)

    # 快速选出满足条件的行索引
    sInd = torch.nonzero(r >= threshold, as_tuple=False).squeeze()

    v = torch.norm(C[sInd], p=2, dim=1)
    sInd = sInd[torch.argsort(v, descending=True)]

    return sInd


# DS3求解器
def ds3solver_regularized(A, lambda_, rho, penalty_matrix, max_iteration, early_stop_threshold):
    with torch.no_grad():

        CFD = torch.ones(A.shape[0], device=A.device)
        num_R, num_C = A.shape

        P_old = torch.ones((num_R, num_C), device=A.device)
        mu = torch.ones((num_R, num_C), device=A.device)

        start_time_admm = time.time()

        for k in range(1, max_iteration + 1):

            Z_new = shrink_L1Lp(P_old - (mu + A) / rho, lambda_ / rho * CFD, penalty_matrix)

            P_new = solver_BCLS_closedForm(Z_new + mu / rho)

            mu.add_(rho * (Z_new - P_new))

            err1 = error_coef(Z_new, P_new)
            err2 = error_coef(P_old, P_new)

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
