"""
clustering.py — K-Means 聚类工具（手动实现，无需 sklearn）
"""
import numpy as np

def kmeans_manual(X, n_clusters=4, max_iter=100, random_state=42):
    """
    手写 K-Means 聚类（Lloyd算法），无需 sklearn

    参数
    ----------
    X : np.ndarray shape (n_samples, n_features)
        输入数据矩阵
    n_clusters : int
        聚类数量
    max_iter : int
        最大迭代次数
    random_state : int
        随机种子

    返回
    ----------
    centroids : np.ndarray shape (n_clusters, n_features)
        聚类中心
    labels : np.ndarray shape (n_samples,)
        每个样本的聚类标签
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    # 随机初始化聚类中心
    init_idx = rng.choice(n_samples, n_clusters, replace=False)
    centroids = X[init_idx].copy().astype(np.float64)

    for iteration in range(max_iter):
        # E-step: 分配样本到最近的中心
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        labels = np.argmin(distances, axis=1)

        # M-step: 更新聚类中心
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]  # 空簇保持原中心

        # 检查收敛
        shift = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids
        if shift < 1e-8:
            break

    # 最后一次分配
    distances = np.zeros((n_samples, n_clusters))
    for k in range(n_clusters):
        distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
    labels = np.argmin(distances, axis=1)

    return centroids, labels
