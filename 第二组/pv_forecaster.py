"""
pv_forecaster.py — 技术一升级版：分布式绿电出力日前预测

================================================================
基于 Khayat et al. (2025) Scientific African 29, e02884
"A novel hybrid GRU-XGBoost model for day-ahead photovoltaic
 generation forecasting in microgrids"

架构：GRU(时序特征) → XGBoost(残差精炼) → 最终PV预测
超参：PSO粒子群优化（两阶段各跑一次）
================================================================

功能：
  1. 从NASA POWER CSV加载气象数据（含RH2M湿度）
  2. 物理模型生成PV训练标签
  3. 06:00-20:00日照时段过滤 + Min-Max归一化
  4. PSO优化GRU超参 → 训练GRU → 初步PV预测
  5. PSO优化XGBoost超参 → 训练XGBoost → 精炼PV预测
  6. 多基线对比（Persistence/纯GRU/纯XGBoost/GRU-XGBoost）
  7. 分晴天/阴天评估
  8. 对标论文的五项指标（RMSE/nRMSE/MAE/MAPE/R^2）

使用方法：
  python pv_forecaster.py              # 使用论文最优超参（推荐）
  python pv_forecaster.py --pso        # 运行完整PSO超参优化（较慢）
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_params, load_nasa_power_full
from green_power_predictor import compute_pv_power
from utils.plot_utils import (
    set_chinese_label, clean_axes, set_time_xticks,
    CHINESE_FONT, COLORS, save_and_show,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False
    print("[WARN] xgboost 未安装，请执行: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch 未安装，请执行: pip install torch")


# ══════════════════════════════════════════
# 1. PSO 粒子群优化（手写，无第三方依赖）
# ══════════════════════════════════════════
class Particle:
    """PSO单个粒子"""
    def __init__(self, bounds, rng):
        self.position = np.array([rng.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(len(bounds))
        self.best_position = self.position.copy()
        self.best_score = float('inf')
        self.score = float('inf')


class PSO:
    """
    粒子群优化算法

    参数
    ----------
    bounds : list of (low, high)
        每个维度的搜索边界
    n_particles : int
        粒子数量
    n_iterations : int
        迭代轮数
    w, c1, c2 : float
        惯性权重、个体学习因子、全局学习因子
    discrete_indices : list of int or None
        需要取整的维度索引（如GRU units, batch_size）
    """
    def __init__(self, bounds, n_particles=10, n_iterations=15,
                 w=0.7, c1=1.5, c2=1.5, discrete_indices=None,
                 random_state=42):
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.discrete_indices = discrete_indices or []
        self.rng = np.random.RandomState(random_state)
        self.best_score_history = []

    def _discretize(self, pos):
        """对需要整数的维度取整"""
        p = pos.copy()
        for idx in self.discrete_indices:
            p[idx] = round(p[idx])
        return p

    def optimize(self, fitness_fn, verbose=True):
        """
        运行PSO优化

        参数
        ----------
        fitness_fn : callable
            接收参数向量，返回 (score, extra_info) 元组
            score越小越好

        返回
        ----------
        best_position, best_score
        """
        particles = [Particle(self.bounds, self.rng) for _ in range(self.n_particles)]
        global_best_pos = None
        global_best_score = float('inf')

        for iteration in range(self.n_iterations):
            for i, p in enumerate(particles):
                # 离散化
                pos = self._discretize(p.position)
                # 评估
                score, _ = fitness_fn(pos)
                p.score = score

                if score < p.best_score:
                    p.best_score = score
                    p.best_position = pos.copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_pos = pos.copy()

            # 更新速度和位置
            for p in particles:
                r1, r2 = self.rng.uniform(0, 1, 2)
                p.velocity = (self.w * p.velocity
                              + self.c1 * r1 * (p.best_position - p.position)
                              + self.c2 * r2 * (global_best_pos - p.position))
                p.position = p.position + p.velocity
                # 边界钳制
                for d in range(len(self.bounds)):
                    p.position[d] = np.clip(p.position[d], self.bounds[d][0], self.bounds[d][1])

            self.best_score_history.append(global_best_score)
            if verbose:
                print(f"   PSO iter {iteration+1}/{self.n_iterations}, best_score={global_best_score:.4f}")

        return self._discretize(global_best_pos), global_best_score


# ══════════════════════════════════════════
# 2. GRU 预测模型（PyTorch）
# ══════════════════════════════════════════
class GRUForecaster(nn.Module):
    """
    GRU序列到序列日前PV预测

    Input:  (batch, 15, input_dim)  — Day n 的15h特征（2层GRU + inter-layer dropout）
    Output: (batch, 15)             — Day n+1 的15h PV预测
    """
    def __init__(self, input_dim=6, hidden_dim=128, dropout=0.2, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)        # (B, 15, hidden)
        out = self.dropout(out)
        out = self.fc(out)          # (B, 15, 1)
        return out.squeeze(-1)      # (B, 15)


def train_gru_model(model, train_loader, val_loader, epochs=200,
                    lr=0.01, patience=30, device='cpu', verbose=True):
    """
    训练GRU模型，使用early stopping

    返回
    ----------
    model, best_val_loss, training_history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train:.6f}, val_loss={avg_val:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"   Early stopping at epoch {epoch+1}, best_val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)
    return model, best_val_loss, history


# ══════════════════════════════════════════
# 3. 数据准备
# ══════════════════════════════════════════
def prepare_forecast_data(params, balanced_test=False):
    """
    加载数据，生成PV标签，预处理为监督学习样本

    参数
    ----------
    params : dict
    balanced_test : bool
        True = 测试集跨四季均匀采样；False = 时序尾部分割（默认）

    返回
    ----------
    X_gru, y, X_xgb_features, norm_params, daily_masks
    """
    print("\n" + "=" * 60)
    print("【数据准备：日前PV预测】")
    print("=" * 60)

    # 加载
    df = load_nasa_power_full(params)
    ghi = df["GHI"].values
    dhi = df["DHI"].values
    t2m = df["T2M"].values
    rh2m = df["RH2M"].values

    # 生成PV标签
    pv = compute_pv_power(ghi, t2m, params)
    print(f"   PV标签: 均值={pv.mean():.1f}MW, 峰值={pv.max():.1f}MW")

    # 日照时段: 06:00-20:00 (15小时/天)
    s_start = params["forecast"]["sunlight_start_hour"]  # 6
    s_end = params["forecast"]["sunlight_end_hour"] + 1  # 20→21 (exclusive, gives 6..=20 = 15h)
    sun_hours = s_end - s_start  # = 15

    # 提取每天日照时段的索引
    total_hours = len(df)
    n_days = total_hours // 24
    total_hours_used = n_days * 24
    hourly = np.arange(total_hours_used).reshape(n_days, 24)

    # 构造特征矩阵 (n_days, sun_hours, feature_dim)
    features = np.zeros((n_days, sun_hours, 6))
    for d in range(n_days):
        sun_idx = hourly[d, s_start:s_end]
        hour_of_day = np.arange(s_start, s_end)
        features[d, :, 0] = hour_of_day                          # Hour
        features[d, :, 1] = t2m[sun_idx]                         # T2M
        features[d, :, 2] = rh2m[sun_idx]                        # RH2M
        features[d, :, 3] = dhi[sun_idx]                         # DHI
        features[d, :, 4] = ghi[sun_idx]                         # GHI
        features[d, :, 5] = pv[sun_idx]                          # PV_hist

    # XGBoost额外特征: Day n+1 的 Hour, T2M, RH2M
    xgb_extra = np.zeros((n_days, sun_hours, 3))
    for d in range(n_days):
        xgb_extra[d, :, 0] = np.arange(s_start, s_end)           # Hour (same every day)
        xgb_extra[d, :, 1] = t2m[hourly[d, s_start:s_end]]      # T2M
        xgb_extra[d, :, 2] = rh2m[hourly[d, s_start:s_end]]     # RH2M

    # 构造监督样本: Day n → Day n+1
    X_gru = features[:-1]          # (N-1, 15, 6) — Day n 的特征
    y = features[1:, :, 5]         # (N-1, 15)    — Day n+1 的 PV（预测目标）
    X_xgb_features = xgb_extra[1:] # (N-1, 15, 3) — Day n+1 的天气（XGBoost精炼用）

    n_samples = len(y)
    print(f"   有效样本数: {n_samples} (来自 {n_days} 天)")
    print(f"   GRU输入形状: {X_gru.shape}")
    print(f"   目标形状: {y.shape}")

    # 时序分割
    train_r = params["forecast"]["train_ratio"]
    val_r = params["forecast"]["val_ratio"]
    n_train = int(n_samples * train_r)
    n_val = int(n_samples * val_r)

    # 默认：尾部分割（时序标准做法）
    if balanced_test:
        daily_masks = make_balanced_test_mask(n_samples, n_train, n_val, features)
    else:
        daily_masks = {
            "train": np.arange(n_train),
            "val": np.arange(n_train, n_train + n_val),
            "test": np.arange(n_train + n_val, n_samples),
        }
        print(f"   训练集: {n_train}样本, 验证集: {n_val}样本, 测试集: {n_samples - n_train - n_val}样本")

    # 归一化（只在训练集上计算min/max，防止数据泄露）
    norm_params_dict = {}
    X_gru_norm = X_gru.copy()
    y_norm = y.copy()
    X_xgb_norm = X_xgb_features.copy()

    for feat_idx, feat_name in enumerate(["Hour", "T2M", "RH2M", "DHI", "GHI", "PV"]):
        fmin = X_gru[daily_masks["train"], :, feat_idx].min()
        fmax = X_gru[daily_masks["train"], :, feat_idx].max()
        norm_params_dict[feat_name] = {"min": float(fmin), "max": float(fmax)}
        X_gru_norm[:, :, feat_idx] = (X_gru[:, :, feat_idx] - fmin) / (fmax - fmin + 1e-10)

    y_min = y[daily_masks["train"]].min()
    y_max = y[daily_masks["train"]].max()
    norm_params_dict["PV_target"] = {"min": float(y_min), "max": float(y_max)}
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)

    # XGBoost特征也做归一化
    for feat_idx, feat_name in enumerate(["Hour_xgb", "T2M_xgb", "RH2M_xgb"]):
        fmin = X_xgb_features[daily_masks["train"], :, feat_idx].min()
        fmax = X_xgb_features[daily_masks["train"], :, feat_idx].max()
        norm_params_dict[feat_name] = {"min": float(fmin), "max": float(fmax)}
        X_xgb_norm[:, :, feat_idx] = (X_xgb_features[:, :, feat_idx] - fmin) / (fmax - fmin + 1e-10)

    print(f"   PV归一化: y_min={y_min:.1f}, y_max={y_max:.1f}")

    return X_gru_norm, y_norm, X_xgb_norm, norm_params_dict, daily_masks


def make_balanced_test_mask(n_samples, n_train, n_val, features, n_test=None):
    """
    构造跨四季均匀分布的测试集索引

    策略：全年每季取末尾约9天作为测试集（秋/冬/春/夏各约9天），
    其余归训练+验证。测试日在各季末端，不违反时序原则。

    返回
    ----------
    daily_masks : dict
    """
    ghi_daily = features[:-1, :, 4].mean(axis=1)
    month_of = np.array([min(11, (i % 365) // 30) for i in range(n_samples)])
    smap = {0:"冬",1:"冬",2:"春",3:"春",4:"春",5:"夏",6:"夏",7:"夏",
            8:"秋",9:"秋",10:"秋",11:"冬"}

    season_ranges = {"春": [2,3,4], "夏": [5,6,7], "秋": [8,9,10], "冬": [0,1,11]}
    test_indices = []
    n_per = 9  # 每季固定9天

    for s_label in ["春","夏","秋","冬"]:
        months = season_ranges[s_label]
        candidates = [i for i in range(n_samples) if month_of[i] in months]
        tail = candidates[-min(len(candidates)//5, 25):] if candidates else []
        sampled = list(np.array(tail)[np.linspace(0, len(tail)-1,
                     min(n_per, len(tail)), dtype=int)]) if tail else []
        test_indices.extend(sampled)
        print(f"   {s_label}季: {len(candidates)}天, 取末尾{len(tail)}天采样{len(sampled)}天, "
              f"均GHI={ghi_daily[sampled].mean():.0f} W/m2" if sampled else f"   {s_label}季: 无候选")

    test_set = set(sorted(test_indices))
    # 其余归训练（不含测试日），验证从训练末尾取12%
    train_all = np.array([i for i in range(n_samples) if i not in test_set])
    n_v = int(len(train_all) * 0.12)
    daily_masks = {
        "train": train_all[:len(train_all) - n_v],
        "val": train_all[len(train_all) - n_v:],
        "test": np.array(sorted(test_set)),
    }
    print(f"   训练: {len(daily_masks['train'])}, 验证: {len(daily_masks['val'])}, "
          f"测试(均衡): {len(daily_masks['test'])}天, "
          f"均GHI={ghi_daily[list(test_set)].mean():.0f} vs 全局={ghi_daily.mean():.0f} W/m2")
    return daily_masks


def to_tensor_loader(X, y, batch_size=32, shuffle=True):
    """转为PyTorch DataLoader"""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ══════════════════════════════════════════
# 4. GRU训练 + PSO优化
# ══════════════════════════════════════════
def gru_fitness_fn(params_vec, X_train, y_train, X_val, y_val, device):
    """GRU PSO适应度函数：训练GRU并返回验证RMSE"""
    units = int(params_vec[0])
    dropout = params_vec[1]
    lr = params_vec[2]
    batch_size = int(params_vec[3])

    input_dim = X_train.shape[2]
    model = GRUForecaster(input_dim=input_dim, hidden_dim=units, dropout=dropout)

    train_loader = to_tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = to_tensor_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    _, val_loss, _ = train_gru_model(
        model, train_loader, val_loader,
        epochs=80, lr=lr, patience=15,
        device=device, verbose=False,
    )
    rmse = np.sqrt(val_loss)
    return rmse, {"units": units, "dropout": dropout, "lr": lr, "batch_size": batch_size}


def xgb_fitness_fn(params_vec, X_train, y_train, X_val, y_val):
    """XGBoost PSO适应度函数：训练XGBoost并返回验证RMSE"""
    lr = params_vec[0]
    max_depth = int(params_vec[1])
    subsample = params_vec[2]
    colsample = params_vec[3]

    # X_train: (N, 165) = GRU预测(15) + Day_n原始特征(90) + Day_n+1天气(45) + GRU偏差(15)
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=lr,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample,
        objective='reg:squarederror',
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    rmse = np.sqrt(np.mean((pred - y_val) ** 2))
    return rmse, {"learning_rate": lr, "max_depth": max_depth,
                  "subsample": subsample, "colsample_bytree": colsample}


# ══════════════════════════════════════════
# 5. 评估函数
# ══════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    """计算五项评估指标（对标论文）"""
    # y_true, y_pred 都是15h展平的1D数组
    mask = y_true > 0.01  # 避免除以极小的PV值
    y_t = y_true[mask]
    y_p = y_pred[mask]

    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    nrmse = rmse / (y_t.max() - y_t.min()) if y_t.max() > y_t.min() else 0
    mae = np.mean(np.abs(y_t - y_p))
    mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-10))) * 100
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return {"RMSE_MW": round(float(rmse), 2), "nRMSE": round(float(nrmse), 4),
            "MAE_MW": round(float(mae), 2), "MAPE_pct": round(float(mape), 2),
            "R2": round(float(r2), 4)}


def evaluate_models(X_gru, y, X_xgb_extra, daily_masks, norm_params, gru_params, xgb_params, device='cpu'):
    """全面评估：Persistence / GRU / XGBoost / GRU-XGBoost"""
    print("\n" + "=" * 60)
    print("【模型评估】")
    print("=" * 60)

    train_idx = daily_masks["train"]
    val_idx = daily_masks["val"]
    test_idx = daily_masks["test"]
    y_min = norm_params["PV_target"]["min"]
    y_max = norm_params["PV_target"]["max"]

    results = {}

    # ── 基线1: Persistence（明天=今天）──
    y_test_raw = y[test_idx]
    # Persistence: Day n+1 预测 = Day n 的实际PV
    pv_min = norm_params["PV"]["min"]
    pv_max = norm_params["PV"]["max"]
    y_persist = X_gru[test_idx, :, 5] * (pv_max - pv_min) + pv_min  # 用PV列的归一化参数
    y_test = y_test_raw * (y_max - y_min) + y_min
    results["Persistence"] = compute_metrics(y_test.flatten(), y_persist.flatten())
    print(f"\n   Persistence: nRMSE={results['Persistence']['nRMSE']:.4f}, R2={results['Persistence']['R2']:.4f}")

    # ── 基线2: 纯XGBoost（不接入GRU预测）──
    print("\n--- 纯XGBoost ---")
    # 展平特征：(N, 15*6+15*3) = 展平GRU输入 + 展平额外特征
    X_xgb_train_flat = np.concatenate([
        X_gru[train_idx].reshape(len(train_idx), -1),      # (N, 90)
        X_xgb_extra[train_idx].reshape(len(train_idx), -1),  # (N, 45)
    ], axis=1)  # → (N, 135)
    X_xgb_val_flat = np.concatenate([
        X_gru[val_idx].reshape(len(val_idx), -1),
        X_xgb_extra[val_idx].reshape(len(val_idx), -1),
    ], axis=1)
    X_xgb_test_flat = np.concatenate([
        X_gru[test_idx].reshape(len(test_idx), -1),
        X_xgb_extra[test_idx].reshape(len(test_idx), -1),
    ], axis=1)

    model_xgb_only = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        objective='reg:squarederror', verbosity=0, n_jobs=-1,
    )
    model_xgb_only.fit(X_xgb_train_flat, y[train_idx].reshape(len(train_idx), -1))
    y_xgb_pred = model_xgb_only.predict(X_xgb_test_flat)
    y_xgb_pred = y_xgb_pred.reshape(len(test_idx), -1) * (y_max - y_min) + y_min
    results["XGBoost"] = compute_metrics(y_test.flatten(), y_xgb_pred.flatten())
    print(f"   XGBoost: nRMSE={results['XGBoost']['nRMSE']:.4f}, R2={results['XGBoost']['R2']:.4f}")

    # ── 基线3: 纯GRU ──
    print("\n--- 纯GRU ---")
    input_dim = X_gru.shape[2]
    model_gru = GRUForecaster(input_dim=input_dim, hidden_dim=gru_params['units'],
                               dropout=gru_params['dropout'])
    train_loader = to_tensor_loader(X_gru[train_idx], y[train_idx],
                                     batch_size=gru_params['batch_size'], shuffle=True)
    val_loader = to_tensor_loader(X_gru[val_idx], y[val_idx],
                                   batch_size=gru_params['batch_size'], shuffle=False)
    test_loader = to_tensor_loader(X_gru[test_idx], y[test_idx],
                                    batch_size=gru_params['batch_size'], shuffle=False)

    model_gru, _, history = train_gru_model(
        model_gru, train_loader, val_loader,
        epochs=300, lr=gru_params['lr'], patience=50, device=device, verbose=True,
    )
    model_gru.eval()
    with torch.no_grad():
        y_gru_pred_norm = model_gru(torch.tensor(X_gru[test_idx], dtype=torch.float32).to(device)).cpu().detach().numpy()
    y_gru_pred = y_gru_pred_norm * (y_max - y_min) + y_min
    results["GRU"] = compute_metrics(y_test.flatten(), y_gru_pred.flatten())
    print(f"   GRU: nRMSE={results['GRU']['nRMSE']:.4f}, R2={results['GRU']['R2']:.4f}")

    # ── 最终模型: GRU-XGBoost ──
    print("\n--- GRU-XGBoost (混合模型) ---")
    # 用已训练的GRU对训练集和测试集做预测
    gru_pred_train = model_gru(torch.tensor(X_gru[train_idx], dtype=torch.float32).to(device)).cpu().detach().numpy()
    gru_pred_val = model_gru(torch.tensor(X_gru[val_idx], dtype=torch.float32).to(device)).cpu().detach().numpy()
    gru_pred_test = model_gru(torch.tensor(X_gru[test_idx], dtype=torch.float32).to(device)).cpu().detach().numpy()

    # 计算GRU在验证集上的逐时系统性偏差（告诉XGBoost"GRU在每小时习惯偏高/偏低多少"）
    gru_residual_val = y[val_idx] - gru_pred_val           # (N_val, 15)
    hourly_bias = gru_residual_val.mean(axis=0)            # (15,) — 逐时均值残差
    hourly_bias_train = np.tile(hourly_bias, (len(train_idx), 1))
    hourly_bias_val = np.tile(hourly_bias, (len(val_idx), 1))
    hourly_bias_test = np.tile(hourly_bias, (len(test_idx), 1))

    # 构造XGBoost输入: GRU预测 + Day n+1天气 + Day n原始特征 + 逐时GRU偏差
    # → 15 + 45 + 90 + 15 = 165 维（纯XGBoost基线仅135维，混合多了GRU信号）
    def build_xgb_input(gru_pred, xgb_extra, X_gru_raw=None, bias=None):
        parts = [gru_pred, xgb_extra.reshape(len(gru_pred), -1)]
        if X_gru_raw is not None:
            parts.append(X_gru_raw.reshape(len(gru_pred), -1))   # Day n 原始90维
        if bias is not None:
            parts.append(bias)                                    # 逐时偏差15维
        return np.concatenate(parts, axis=1)

    model_xgb = xgb.XGBRegressor(
        n_estimators=800,                                # 500→800，适应特征维度增大
        learning_rate=xgb_params['learning_rate'],
        max_depth=xgb_params['max_depth'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        objective='reg:squarederror',
        verbosity=0,
        n_jobs=-1,
    )
    model_xgb.fit(build_xgb_input(gru_pred_train, X_xgb_extra[train_idx],
                                  X_gru[train_idx], hourly_bias_train),
                  y[train_idx].reshape(len(train_idx), -1),
                  eval_set=[(build_xgb_input(gru_pred_val, X_xgb_extra[val_idx],
                                             X_gru[val_idx], hourly_bias_val),
                             y[val_idx].reshape(len(val_idx), -1))],
                  verbose=False)

    y_hybrid_pred_norm = model_xgb.predict(
        build_xgb_input(gru_pred_test, X_xgb_extra[test_idx],
                        X_gru[test_idx], hourly_bias_test))
    y_hybrid_pred = y_hybrid_pred_norm.reshape(len(test_idx), -1) * (y_max - y_min) + y_min
    results["GRU-XGBoost"] = compute_metrics(y_test.flatten(), y_hybrid_pred.flatten())
    print(f"   GRU-XGBoost: nRMSE={results['GRU-XGBoost']['nRMSE']:.4f}, R2={results['GRU-XGBoost']['R2']:.4f}")

    return results, y_test, {
        "Persistence": y_persist,
        "GRU": y_gru_pred,
        "XGBoost": y_xgb_pred,
        "GRU-XGBoost": y_hybrid_pred,
    }, model_gru, history


# ══════════════════════════════════════════
# 6. 分天气评估
# ══════════════════════════════════════════
def evaluate_by_weather(X_gru, y, y_test, predictions, daily_masks, norm_params):
    """分晴天/阴天评估模型性能"""
    test_idx = daily_masks["test"]
    y_min = norm_params["PV_target"]["min"]
    y_max = norm_params["PV_target"]["max"]

    # 用GHI日均值判断天气
    ghi_test = X_gru[test_idx, :, 4] * (norm_params["GHI"]["max"] - norm_params["GHI"]["min"]) + norm_params["GHI"]["min"]
    daily_ghi_mean = ghi_test.mean(axis=1)

    sunny_mask = daily_ghi_mean > 400    # 晴天
    cloudy_mask = daily_ghi_mean <= 400  # 阴天

    weather_results = {}
    for model_name, pred in predictions.items():
        wr = {"sunny": {}, "cloudy": {}}
        if sunny_mask.sum() > 0:
            wr["sunny"] = compute_metrics(y_test[sunny_mask].flatten(), pred[sunny_mask].flatten())
        else:
            wr["sunny"] = {"nRMSE": 0, "R2": 0, "RMSE_MW": 0, "MAE_MW": 0, "MAPE_pct": 0}
        if cloudy_mask.sum() > 0:
            wr["cloudy"] = compute_metrics(y_test[cloudy_mask].flatten(), pred[cloudy_mask].flatten())
        else:
            wr["cloudy"] = {"nRMSE": 0, "R2": 0, "RMSE_MW": 0, "MAE_MW": 0, "MAPE_pct": 0}
        weather_results[model_name] = wr

    print(f"\n   测试集: 晴天={sunny_mask.sum()}天, 阴天={cloudy_mask.sum()}天")
    print(f"\n   {'模型':<16} {'晴天nRMSE':>10} {'晴天R2':>10} {'阴天nRMSE':>10} {'阴天R2':>10}")
    print("   " + "-" * 56)
    for model_name in ["Persistence", "GRU", "XGBoost", "GRU-XGBoost"]:
        if model_name in weather_results:
            s = weather_results[model_name]["sunny"]
            c = weather_results[model_name]["cloudy"]
            print(f"   {model_name:<16} {s['nRMSE']:>10.4f} {s['R2']:>10.4f} {c['nRMSE']:>10.4f} {c['R2']:>10.4f}")

    return weather_results


# ══════════════════════════════════════════
# 7. 可视化
# ══════════════════════════════════════════
def plot_pred_vs_actual(y_test, predictions, save_dir, n_days=7):
    """测试集首周预测vs实际对比"""
    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(n_days * 15)

    ax.plot(t, y_test[:n_days].flatten(), 'ko-', linewidth=2, markersize=3, label='实际PV', zorder=3)
    colors_pred = {"Persistence": "#7F7F7F", "GRU": "#FF7F0E", "GRU-XGBoost": "#1F77B4"}
    for model_name in ["Persistence", "GRU", "GRU-XGBoost"]:
        if model_name in predictions:
            ax.plot(t, predictions[model_name][:n_days].flatten(), '--', color=colors_pred.get(model_name),
                    linewidth=1.8, label=model_name, alpha=0.8)

    # 画日分隔线
    for d in range(1, n_days):
        ax.axvline(x=d * 15, color='gray', linestyle=':', alpha=0.4)

    set_chinese_label(ax, xlabel="时间 (15h/天 × 7天)", ylabel="光伏出力 (MW)",
                      title=f"日前PV预测 — 测试集前{n_days}天")
    ax.legend(prop={"family": "SimHei", "size": 10}, frameon=False, loc="upper right")
    clean_axes(ax)

    path = os.path.join(save_dir, "pred_vs_actual_week.png")
    save_and_show(fig, path)


def plot_scatter_comparison(y_test, pred_gru, pred_hybrid, save_dir):
    """散点图对比GRU vs GRU-XGBoost（对标论文Fig.8/9）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    y_flat = y_test.flatten()
    for ax, pred, title in [
        (ax1, pred_gru.flatten(), "纯GRU"),
        (ax2, pred_hybrid.flatten(), "GRU-XGBoost混合模型"),
    ]:
        ax.scatter(y_flat, pred, c='#1F77B4', alpha=0.4, s=12, edgecolors='none')
        mm = max(y_flat.max(), pred.max())
        ax.plot([0, mm], [0, mm], 'k--', linewidth=1, alpha=0.6)
        ax.set_xlim([0, mm * 1.05])
        ax.set_ylim([0, mm * 1.05])
        set_chinese_label(ax, xlabel="实际PV出力 (MW)", ylabel="预测PV出力 (MW)", title=title)
        clean_axes(ax)
        # 标注R^2
        m = compute_metrics(y_flat, pred)
        ax.text(0.05, 0.92, f"nRMSE={m['nRMSE']:.4f}\nR^2={m['R2']:.4f}",
                transform=ax.transAxes, fontsize=11, family="SimHei",
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "scatter_gru_vs_hybrid.png"))


def plot_model_comparison(results, save_dir):
    """多模型nRMSE/R^2对比柱状图（对标论文Fig.12）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ["Persistence", "XGBoost", "GRU", "GRU-XGBoost"]
    colors_bar = ["#7F7F7F", "#FF7F0E", "#2CA02C", "#1F77B4"]

    nrmse_vals = [results[m]["nRMSE"] * 100 for m in models]  # 转百分比
    r2_vals = [results[m]["R2"] for m in models]

    for ax, vals, ylabel, title in [
        (ax1, nrmse_vals, "nRMSE (%)", "nRMSE 对比（越低越好）"),
        (ax2, r2_vals, "R^2", "R^2 对比（越高越好）"),
    ]:
        bars = ax.bar(models, vals, color=colors_bar, edgecolor='white', width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(vals) * 0.02),
                    f"{val:.2f}", ha='center', fontsize=10, family="SimHei")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, family="SimHei", fontsize=10, rotation=15)
        set_chinese_label(ax, ylabel=ylabel, title=title)
        clean_axes(ax)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "model_comparison_bar.png"))


def plot_training_history(history, pso_history_gru, pso_history_xgb, save_dir):
    """训练曲线 + PSO收敛曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # GRU训练曲线
    ax = axes[0]
    ax.plot(history['train'], label='训练Loss', color='#1F77B4', linewidth=1.5)
    ax.plot(history['val'], label='验证Loss', color='#D62728', linewidth=1.5)
    set_chinese_label(ax, xlabel="Epoch", ylabel="MSE Loss", title="GRU训练曲线")
    ax.legend(prop={"family": "SimHei", "size": 9}, frameon=False)
    clean_axes(ax)

    # PSO收敛(GPU)
    if pso_history_gru:
        ax = axes[1]
        ax.plot(pso_history_gru, 'o-', color='#2CA02C', linewidth=1.5, markersize=4)
        set_chinese_label(ax, xlabel="迭代", ylabel="验证RMSE", title="PSO收敛 — GRU超参")
        clean_axes(ax)

    # PSO收敛(XGBoost)
    if pso_history_xgb:
        ax = axes[2]
        ax.plot(pso_history_xgb, 's-', color='#FF7F0E', linewidth=1.5, markersize=4)
        set_chinese_label(ax, xlabel="迭代", ylabel="验证RMSE", title="PSO收敛 — XGBoost超参")
        clean_axes(ax)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "training_history.png"))


def plot_forecast_single_day(y_test, predictions, daily_masks, norm_params, X_gru, save_dir,
                              nasa_start_date="2025-01-01"):
    """
    单日预测展示：按 GRU-XGBoost 的每日 nRMSE 排序，
    选最佳预测 / 典型预测 / 最差预测 各一天，展示实际 vs 预测 PV 曲线。
    """
    test_idx = daily_masks["test"]
    # 样本索引 → 日历日期（样本i预测的是 Day i+1 的PV，对应 Jan 2 + i）
    base_date = pd.Timestamp("2025-01-02")
    test_dates = [base_date + pd.Timedelta(days=int(i)) for i in test_idx]

    # 计算每天GRU-XGBoost的nRMSE
    daily_nrmse = []
    for i, idx in enumerate(range(len(test_idx))):
        actual = y_test[idx]
        pred = predictions["GRU-XGBoost"][idx]
        m = compute_metrics(actual, pred)
        daily_nrmse.append(m["nRMSE"])

    order = np.argsort(daily_nrmse)  # 升序：小误差在前
    best_idx = order[0]              # 最佳
    median_idx = order[len(order)//2]  # 典型
    worst_idx = order[-1]             # 最差

    labels = [
        (best_idx, "最佳预测", "#2CA02C"),
        (median_idx, "典型预测", "#1F77B4"),
        (worst_idx, "最差预测", "#D62728"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t_hours = np.arange(6, 21)

    for ax, (idx, tag, color) in zip(axes, labels):
        actual = y_test[idx]
        hybrid_pred = predictions["GRU-XGBoost"][idx]
        persist_pred = predictions["Persistence"][idx]
        date_str = test_dates[idx].strftime("%Y-%m-%d")

        ax.plot(t_hours, actual, 'ko-', linewidth=2.5, markersize=6, label='实际PV')
        ax.plot(t_hours, hybrid_pred, 's-', color=color, linewidth=2, markersize=5,
                label='GRU-XGBoost预测')
        ax.plot(t_hours, persist_pred, '--', color='#7F7F7F', linewidth=1.5, label='Persistence')
        ax.fill_between(t_hours, hybrid_pred, actual, alpha=0.15, color=color)

        m = compute_metrics(actual, hybrid_pred)
        set_chinese_label(ax, xlabel="时间 (h)", ylabel="光伏出力 (MW)",
                          title=f"{tag} ({date_str})")
        set_time_xticks(ax, hours=t_hours)
        ax.legend(prop={"family": "SimHei", "size": 9}, frameon=False, loc="upper left")
        ax.text(0.97, 0.15,
                f"nRMSE={m['nRMSE']:.3f}\nR2={m['R2']:.3f}\nRMSE={m['RMSE_MW']:.1f} MW",
                transform=ax.transAxes, fontsize=9, family="SimHei",
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        clean_axes(ax)

    plt.tight_layout()
    path = os.path.join(save_dir, "forecast_single_day.png")
    save_and_show(fig, path)


# ══════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="GRU-XGBoost-PSO 日前PV预测")
    parser.add_argument("--pso", action="store_true", help="运行完整PSO超参优化（默认使用论文最优值）")
    parser.add_argument("--balanced", action="store_true",
                        help="测试集跨四季均衡采样（默认时序尾部分割）")
    parser.add_argument("--no-gpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--export-date", type=str, default="best",
                        help="导出哪个测试日的24h PV: best/typical/worst 或具体日期如2025-07-15")
    args = parser.parse_args()

    if not TORCH_AVAILABLE or not XGB_AVAILABLE:
        print("[ERR] 缺少依赖。请执行: pip install torch xgboost")
        sys.exit(1)

    device = 'cuda' if (torch.cuda.is_available() and not args.no_gpu) else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    params = load_params()
    os.makedirs("outputs/pv_forecast", exist_ok=True)

    # ── 数据准备 ──
    X_gru, y, X_xgb_extra, norm_params, daily_masks = prepare_forecast_data(
        params, balanced_test=args.balanced)
    train_idx = daily_masks["train"]
    val_idx = daily_masks["val"]

    # ── 阶段1: GRU超参 + 训练 ──
    print("\n" + "=" * 60)
    print("【阶段1：GRU 训练】")
    print("=" * 60)

    if args.pso:
        print("运行PSO优化GRU超参...")
        gru_bounds = [
            (32, 256),      # units
            (0.1, 0.5),     # dropout
            (0.001, 0.01),  # lr
            (16, 128),      # batch_size
        ]
        pso_gru = PSO(gru_bounds, n_particles=8, n_iterations=10,
                      discrete_indices=[0, 3], random_state=42)
        best_gru, best_score_gru = pso_gru.optimize(
            lambda v: gru_fitness_fn(v, X_gru[train_idx], y[train_idx],
                                      X_gru[val_idx], y[val_idx], device),
            verbose=True,
        )
        gru_params = {"units": int(best_gru[0]), "dropout": float(best_gru[1]),
                      "lr": float(best_gru[2]), "batch_size": int(best_gru[3])}
        pso_history_gru = pso_gru.best_score_history
    else:
        # 使用论文最优值
        gru_params = {"units": 128, "dropout": 0.2, "lr": 0.01, "batch_size": 32}
        pso_history_gru = []
        print(f"使用论文最优GRU超参: {gru_params}")

    print(f"   GRU超参: units={gru_params['units']}, dropout={gru_params['dropout']}, "
          f"lr={gru_params['lr']}, batch_size={gru_params['batch_size']}")

    # ── 阶段2: XGBoost超参 ──
    print("\n" + "=" * 60)
    print("【阶段2：XGBoost 超参】")
    print("=" * 60)

    # 先用论文最优值训练GRU，然后为XGBoost构造输入
    input_dim = X_gru.shape[2]
    model_gru_temp = GRUForecaster(input_dim=input_dim, hidden_dim=gru_params['units'],
                                    dropout=gru_params['dropout'])
    train_loader = to_tensor_loader(X_gru[train_idx], y[train_idx],
                                     batch_size=gru_params['batch_size'], shuffle=True)
    val_loader = to_tensor_loader(X_gru[val_idx], y[val_idx],
                                   batch_size=gru_params['batch_size'], shuffle=False)
    model_gru_temp, _, _ = train_gru_model(
        model_gru_temp, train_loader, val_loader,
        epochs=300, lr=gru_params['lr'], patience=50, device=device, verbose=True,
    )

    # GRU对训练/验证集预测
    model_gru_temp.eval()
    with torch.no_grad():
        gru_pred_train = model_gru_temp(
            torch.tensor(X_gru[train_idx], dtype=torch.float32).to(device)).cpu().detach().numpy()
        gru_pred_val = model_gru_temp(
            torch.tensor(X_gru[val_idx], dtype=torch.float32).to(device)).cpu().detach().numpy()

    # 计算GRU逐时残差作为XGBoost的校准特征
    gru_bias_temp = (gru_pred_val - y[val_idx]).mean(axis=0)  # (15,)

    # 展平：XGBoost输入 GRU预测(15) + Day_n原始(90) + Day_n+1天气(45) + GRU偏差(15) = 165维
    X_xgb_train = np.concatenate([
        gru_pred_train,
        X_gru[train_idx].reshape(len(train_idx), -1),
        X_xgb_extra[train_idx].reshape(len(train_idx), -1),
        np.tile(gru_bias_temp, (len(train_idx), 1)),
    ], axis=1)
    X_xgb_val = np.concatenate([
        gru_pred_val,
        X_gru[val_idx].reshape(len(val_idx), -1),
        X_xgb_extra[val_idx].reshape(len(val_idx), -1),
        np.tile(gru_bias_temp, (len(val_idx), 1)),
    ], axis=1)
    y_train_flat = y[train_idx].reshape(len(train_idx), -1)
    y_val_flat = y[val_idx].reshape(len(val_idx), -1)

    if args.pso:
        print("运行PSO优化XGBoost超参...")
        xgb_bounds = [
            (0.01, 0.3),   # learning_rate
            (3, 10),       # max_depth
            (0.5, 1.0),    # subsample
            (0.5, 1.0),    # colsample_bytree
        ]
        pso_xgb = PSO(xgb_bounds, n_particles=10, n_iterations=15,
                      discrete_indices=[1], random_state=42)
        best_xgb, best_score_xgb = pso_xgb.optimize(
            lambda v: xgb_fitness_fn(v, X_xgb_train, y_train_flat,
                                      X_xgb_val, y_val_flat),
            verbose=True,
        )
        xgb_params = {"learning_rate": float(best_xgb[0]), "max_depth": int(best_xgb[1]),
                      "subsample": float(best_xgb[2]), "colsample_bytree": float(best_xgb[3])}
        pso_history_xgb = pso_xgb.best_score_history
    else:
        xgb_params = {"learning_rate": 0.26, "max_depth": 4, "subsample": 0.91, "colsample_bytree": 0.97}
        pso_history_xgb = []
        print(f"使用论文最优XGBoost超参: {xgb_params}")

    # ── 最终评估 ──
    print("\n" + "=" * 60)
    print("【最终评估 — 测试集】")
    print("=" * 60)

    results, y_test, predictions, model_gru, history = evaluate_models(
        X_gru, y, X_xgb_extra, daily_masks, norm_params, gru_params, xgb_params, device,
    )

    # 分天气评估
    weather_results = evaluate_by_weather(X_gru, y, y_test, predictions, daily_masks, norm_params)

    # ── 打印最终结果表 ──
    print("\n" + "=" * 60)
    print("【最终结果汇总】")
    print("=" * 60)
    print(f"\n   {'模型':<16} {'nRMSE':>10} {'R2':>10} {'RMSE(MW)':>10} {'MAE(MW)':>10} {'MAPE(%)':>10}")
    print("   " + "-" * 66)
    for model_name in ["Persistence", "XGBoost", "GRU", "GRU-XGBoost"]:
        if model_name in results:
            r = results[model_name]
            print(f"   {model_name:<16} {r['nRMSE']:>10.4f} {r['R2']:>10.4f} "
                  f"{r['RMSE_MW']:>10.2f} {r['MAE_MW']:>10.2f} {r['MAPE_pct']:>10.2f}")

    # ── 绘图 ──
    print("\n--- 生成图表 ---")
    save_dir = "outputs/pv_forecast"
    plot_pred_vs_actual(y_test, predictions, save_dir, n_days=7)
    plot_scatter_comparison(y_test, predictions["GRU"], predictions["GRU-XGBoost"], save_dir)
    plot_model_comparison(results, save_dir)
    plot_training_history(history, pso_history_gru, pso_history_xgb, save_dir)
    plot_forecast_single_day(y_test, predictions, daily_masks, norm_params, X_gru, save_dir)

    # ── 保存指标 ──
    metrics_out = {
        "model": "GRU-XGBoost-PSO",
        "reference": "Khayat et al. (2025) Scientific African 29, e02884",
        "device": device,
        "data": {
            "n_samples": len(y),
            "n_train": len(daily_masks["train"]),
            "n_val": len(daily_masks["val"]),
            "n_test": len(daily_masks["test"]),
            "sunlight_range": f"{params['forecast']['sunlight_start_hour']}:00-{params['forecast']['sunlight_end_hour']}:00",
            "test_mode": "balanced" if args.balanced else "temporal_tail",
        },
        "hyperparameters": {
            "gru": gru_params,
            "xgboost": xgb_params,
        },
        "test_results": results,
        "weather_results": weather_results,
        "pso_used": args.pso,
    }

    metrics_path = os.path.join(save_dir, "forecast_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[OK] 全部成果已保存至: {save_dir}/")
    print(f"  [OK] forecast_metrics.json")
    print(f"  [OK] pred_vs_actual_week.png")
    print(f"  [OK] scatter_gru_vs_hybrid.png")
    print(f"  [OK] model_comparison_bar.png")
    print(f"  [OK] training_history.png")

    # ── 导出代表性预测日 24h PV 曲线，供模块二使用 ──
    test_idx = daily_masks["test"]
    hybrid_pred = predictions["GRU-XGBoost"]

    # 计算每天 nRMSE 和日均 PV
    daily_nrmse = []
    daily_pv_mean = []
    for i in range(len(test_idx)):
        m = compute_metrics(y_test[i], hybrid_pred[i])
        daily_nrmse.append(m["nRMSE"])
        daily_pv_mean.append(y_test[i].mean())

    base_date = pd.Timestamp("2025-01-02")
    test_dates = [base_date + pd.Timedelta(days=int(ti)) for ti in test_idx]
    date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(test_dates)}

    # 按 --export-date 选择日期
    nrmse_order = np.argsort(daily_nrmse)
    tag_map = {
        "best":    int(nrmse_order[0]),
        "typical": int(nrmse_order[len(nrmse_order) // 2]),
        "worst":   int(nrmse_order[-1]),
    }

    export_date = args.export_date
    if export_date in tag_map:
        picks = {export_date: tag_map[export_date]}
    elif export_date in date_to_idx:
        picks = {export_date: date_to_idx[export_date]}
    else:
        print(f"\n[WARN] 未知日期: {export_date}")
        print(f"  可用: best/typical/worst, 或测试集中的具体日期")
        print(f"  测试集日期范围: {test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  共 {len(test_dates)} 天")
        picks = {"best": tag_map["best"]}  # fallback

    s_start = params["forecast"]["sunlight_start_hour"]
    s_end = params["forecast"]["sunlight_end_hour"] + 1

    for tag, idx in picks.items():
        date = test_dates[idx]
        pv_24h = np.zeros(24)
        pv_24h[s_start:s_end] = hybrid_pred[idx]
        pv_24h = np.maximum(pv_24h, 0)

        csv_path = os.path.join(save_dir, f"demo_day_24h_{tag.replace('-', '_')}.csv")
        pd.DataFrame({"Hour": np.arange(24), "PV_MW": pv_24h.round(4)}).to_csv(
            csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 导出预测日 {date.strftime('%Y-%m-%d')}  "
              f"nRMSE={daily_nrmse[idx]:.4f}  日均PV={daily_pv_mean[idx]:.1f}MW  →  {csv_path}")

    # 最终对比
    hybrid = results.get("GRU-XGBoost", {})
    persistence = results.get("Persistence", {})
    if hybrid and persistence:
        nrmse_improve = (persistence['nRMSE'] - hybrid['nRMSE']) / (persistence['nRMSE'] + 1e-10) * 100
        print(f"\n{'=' * 60}")
        print(f"GRU-XGBoost nRMSE={hybrid['nRMSE']:.4f} (vs Persistence 提升 {nrmse_improve:.1f}%)")
        print(f"对标论文: nRMSE=0.0417, R2=0.9880")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
