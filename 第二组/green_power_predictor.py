"""
green_power_predictor.py — 模块一：分布式绿电出力预测技术

============================================================
对应技术：技术一 — 分布式绿电出力预测技术
============================================================

功能：
  1. 从NASA POWER气象数据计算光伏出力（物理模型+温度修正）
  2. K-Means聚类提取典型日曲线 -> 层一（通用县域模型）
  3. 波动性量化分析（标准差、爬坡率、置信区间）
  4. 选取闽清县具体真实日期 -> 层二（闽清验证）
  5. 构造闽清小水电典型日出力曲线 -> 层二

输出：
  · data/green_power_curves.csv      层一：4条K-Means典型日曲线
  · data/minqing_raw_days.csv        层二：2-3个真实日期的逐时GHI+光伏出力
  · data/minqing_hydro_curve.csv     层二：闽清小水电典型日曲线
  · outputs/通用模型/green_power_curves.png
  · outputs/通用模型/fluctuation_analysis.png
  · outputs/闽清验证/minqing_pv_hydro_curves.png
  · outputs/闽清验证/fluctuation_report.json

使用方法：
  python green_power_predictor.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_params, load_nasa_power
from utils.clustering import kmeans_manual
from utils.plot_utils import (
    plot_green_power_curves,
    plot_fluctuation_analysis,
    save_and_show,
    set_chinese_label,
    clean_axes,
    set_time_xticks,
)
import matplotlib.pyplot as plt




# ══════════════════════════════════════════
# 1. 光伏出力物理模型
# ══════════════════════════════════════════
def compute_pv_power(ghi, t2m, params):
    """
    标准光伏温度补偿物理模型

    公式:
        T_cell = T2M + (NOCT - 20) / 800 * GHI
        P_pv   = P_rated * (GHI / G_STC) * [1 + k_temp * (T_cell - T_STC)]  (电池片直流侧)
        P_pv   = max(P_pv, 0)
        P_pv_ac = P_pv * η_system  (并网交流侧，含逆变器/线损/衰减等)

    参数
    ----------
    ghi : np.ndarray
        逐时全球水平辐照度 (W/m²)
    t2m : np.ndarray
        逐时2米环境温度 (°C)
    params : dict
        pv_system参数字典

    返回
    ----------
    P_pv : np.ndarray
        逐时光伏出力 (MW)，与输入同长度
    """
    P_rated = params["pv_system"]["P_rated_MW"]
    G_STC = params["pv_system"]["G_STC"]
    T_STC = params["pv_system"]["T_STC"]
    k_temp = params["pv_system"]["k_temp"]
    NOCT = params["pv_system"]["NOCT"]
    eta_sys = params["pv_system"].get("system_efficiency", 1.0)  # 94.52%=逆变器+线损+衰减

    # 电池片温度
    T_cell = t2m + (NOCT - 20) / 800.0 * ghi

    # 光伏出力（直流侧 → 交流侧并网，乘以系统效率）
    P_pv = P_rated * (ghi / G_STC) * (1 + k_temp * (T_cell - T_STC))
    P_pv = np.maximum(P_pv, 0)
    P_pv = P_pv * eta_sys  # DC → AC 并网

    return P_pv


# ══════════════════════════════════════════
# 2. K-Means 典型日聚类 -> 层一
# ══════════════════════════════════════════
def cluster_typical_days(P_pv, n_clusters=4, days_per_year=365):
    """
    对全年365天的光伏出力进行K-Means聚类

    参数
    ----------
    P_pv : np.ndarray shape (8760,)
        全年逐时光伏出力
    n_clusters : int
        聚类数量（默认4）
    days_per_year : int
        全年天数

    返回
    ----------
    typical_days : np.ndarray shape (n_clusters, 24)
        聚类中心（4条典型日曲线）
    labels : np.ndarray shape (days_per_year,)
        每天所属的聚类标签
    """
    hours = 24
    total_days = min(len(P_pv) // hours, days_per_year)

    # 重塑为 (days × 24) 矩阵
    P_matrix = P_pv[:total_days * hours].reshape((total_days, hours))

    centroids, labels = kmeans_manual(P_matrix, n_clusters=n_clusters, random_state=42)
    typical_days = centroids

    print(f"\n[OK] K-Means聚类完成：{n_clusters}个典型场景")
    for i in range(n_clusters):
        count = np.sum(labels == i)
        mean_p = typical_days[i].mean()
        max_p = typical_days[i].max()
        print(f"   场景{i+1}: {count}天, 日均出力={mean_p:.2f}MW, 峰值={max_p:.2f}MW")

    return typical_days, labels, P_matrix


# ══════════════════════════════════════════
# 3. 波动性量化分析
# ══════════════════════════════════════════
def compute_fluctuation_stats(typical_days, params, P_matrix=None, labels=None):
    """
    对每条典型日曲线计算波动性指标

    返回
    ----------
    stats : dict
        {
            "mean":        {场景名: array(24,)},
            "std":         {场景名: float},
            "max_ramp":    {场景名: float},
            "effective_h": {场景名: float},
            "pv_ratio":    {场景名: float},
            "ci_upper":    {场景名: array(24,)},
            "ci_lower":    {场景名: array(24,)},
        }
    """
    P_rated = params["pv_system"]["P_rated_MW"]
    n_clusters = typical_days.shape[0]
    scene_names = [f"场景{i+1}" for i in range(n_clusters)]

    stats = {
        "curve": {}, "mean": {}, "std": {}, "max_ramp": {},
        "effective_h": {}, "pv_ratio": {},
        "ci_upper": {}, "ci_lower": {},
    }

    for i, name in enumerate(scene_names):
        curve = typical_days[i]

        stats["mean"][name] = curve.mean()
        stats["std"][name] = curve.std()
        stats["max_ramp"][name] = np.max(np.abs(np.diff(curve)))
        stats["effective_h"][name] = np.sum(curve > 0.1 * P_rated)
        stats["pv_ratio"][name] = (curve.max() - curve.min()) / P_rated

        # 95%置信区间（逐时聚类内标准差，非全局标量）
        stats["curve"][name] = curve
        hourly_std = np.zeros(24)
        for h in range(24):
            hourly_std[h] = P_matrix[labels == i, h].std()
        stats["ci_upper"][name] = curve + 1.96 * hourly_std
        stats["ci_lower"][name] = np.maximum(curve - 1.96 * hourly_std, 0)

    print("\n[OK] 波动性分析完成：")
    for name in scene_names:
        print(f"   {name}: 均值={stats['mean'][name]:.2f}MW, "
              f"标准差={stats['std'][name]:.2f}MW, "
              f"最大爬坡={stats['max_ramp'][name]:.2f}MW/h, "
              f"有效小时={stats['effective_h'][name]}h")

    return stats


# ══════════════════════════════════════════
# 4. 保存典型日曲线CSV -> 层一
# ══════════════════════════════════════════
def save_curves_csv(typical_days, filepath):
    """保存典型日曲线为CSV"""
    n_clusters = typical_days.shape[0]
    df = pd.DataFrame()
    df["Hour"] = np.arange(24)

    # 按峰值从高到低排序
    peak_order = np.argsort(typical_days.max(axis=1))[::-1]
    for rank, idx in enumerate(peak_order):
        df[f"场景{rank+1}"] = typical_days[idx]

    df.to_csv(filepath, index=False)
    print(f"\n[OK] 典型日曲线已保存: {filepath}")


# ══════════════════════════════════════════
# 5. 选取闽清验证日期 -> 层二
# ══════════════════════════════════════════
def select_validation_dates(df_nasa, P_pv, params):
    """
    从8760h中自动筛选2-3个代表性日期用于闽清验证

    筛选逻辑：
        · 晴天：GHI日均值最高、波动最小
        · 阴天：GHI日均值最低
        · 波动日：GHI变异系数最大

    返回
    ----------
    selected : dict
        { "晴天": {"date": "2025-07-15", "ghi": np.array, "pv": np.array}, ... }
    """
    daily_data = []
    dates_unique = df_nasa["datetime"].dt.strftime("%Y-%m-%d").unique()

    for date_str in dates_unique[:365]:  # 只取365天
        mask = df_nasa["datetime"].dt.strftime("%Y-%m-%d") == date_str
        day_ghi = df_nasa.loc[mask, "GHI"].values
        day_pv = P_pv[mask.values]
        if len(day_ghi) == 24:
            daily_data.append({
                "date": date_str,
                "ghi": day_ghi,
                "pv": day_pv,
                "ghi_mean": day_ghi.mean(),
                "ghi_std": day_ghi.std(),
                "ghi_cv": day_ghi.std() / (day_ghi.mean() + 1e-6),  # 变异系数
                "sunlight_hours": np.sum(day_ghi > 50),  # GHI > 50 W/m²的小时数
            })

    df_daily = pd.DataFrame(daily_data)

    # 晴天：GHI均值最高 + 日照时长最长
    sunny_candidates = df_daily.nlargest(5, "sunlight_hours")
    sunny = sunny_candidates.iloc[0]

    # 阴天：GHI均值最低（但至少有一些日照，代表冬季典型）
    cloudy_candidates = df_daily[df_daily["sunlight_hours"] >= 3].nsmallest(5, "sunlight_hours")
    cloudy = cloudy_candidates.iloc[0] if len(cloudy_candidates) > 0 else df_daily.nsmallest(1, "sunlight_hours").iloc[0]

    # 波动日：变异系数最大
    volatile = df_daily.nlargest(1, "ghi_cv").iloc[0]

    selected = {
        "晴天": {"date": sunny["date"], "ghi": sunny["ghi"], "pv": sunny["pv"]},
        "阴天": {"date": cloudy["date"], "ghi": cloudy["ghi"], "pv": cloudy["pv"]},
        "波动日": {"date": volatile["date"], "ghi": volatile["ghi"], "pv": volatile["pv"]},
    }

    print("\n[OK] 闽清验证日期筛选完成：")
    for label, data in selected.items():
        print(f"   {label}: {data['date']}, GHI均值={data['ghi'].mean():.1f} W/m2, "
              f"光伏峰值={data['pv'].max():.2f} MW, 日照={np.sum(data['ghi']>50)}h")

    return selected


def save_minqing_raw_days(selected, filepath):
    """保存闽清验证原始日期数据为CSV"""
    df = pd.DataFrame()
    df["Hour"] = np.arange(24)

    for label, data in selected.items():
        col_ghi = f"{label}_{data['date']}_GHI"
        col_pv = f"{label}_{data['date']}_PV"
        df[col_ghi] = data["ghi"]
        df[col_pv] = data["pv"]

    df.to_csv(filepath, index=False)
    print(f"[OK] 闽清原始日期数据已保存: {filepath}")


# ══════════════════════════════════════════
# 6. 构造闽清小水电曲线 -> 层二
# ══════════════════════════════════════════
def build_hydro_curve(params, season="summer"):
    """
    构造闽清径流式小水电典型日出力曲线

    **简化模型**：径流式小水电实际出力受降雨量、上游来水量、水库调度等多因素影响，
    日内曲线并非简单的正弦波。本模型基于典型径流式小水电的出力特征（丰水期基荷较高、
    日内随调度微调）进行简化构造，用于第一性原理层面的可行性论证。如需更高精度，
    应替换为福建省小水电实测日出力数据。

    参数
    ----------
    params : dict
    season : str
        "summer" (丰水期) or "winter" (枯水期)

    返回
    ----------
    P_hydro : np.ndarray shape (24,)
        小水电逐时出力 (MW)
    """
    P_rated = params["minqing_hydro"]["P_hydro_rated_MW"]
    t = np.arange(24)

    if season == "summer":
        # 丰水期：基荷70%，日间随调度微调
        base = params["minqing_hydro"]["base_load_ratio"]
        boost = params["minqing_hydro"]["peak_boost_ratio"]
        # 日间9-17点小幅增加（上游来水+调度）
        P_hydro = P_rated * (base + boost * np.sin(np.pi * (t - 6) / 12))
        P_hydro = np.clip(P_hydro, 0, P_rated)
    else:
        # 枯水期：基荷30%
        base = 0.30
        P_hydro = P_rated * (base + 0.05 * np.sin(np.pi * (t - 6) / 12))
        P_hydro = np.clip(P_hydro, 0, P_rated)

    print(f"\n[OK] 闽清小水电({season})曲线构造完成: "
          f"均值={P_hydro.mean():.2f}MW, 峰值={P_hydro.max():.2f}MW")

    return P_hydro


def save_hydro_curve(P_hydro_summer, P_hydro_winter, filepath):
    """保存小水电曲线为CSV"""
    df = pd.DataFrame()
    df["Hour"] = np.arange(24)
    df["丰水期"] = P_hydro_summer
    df["枯水期"] = P_hydro_winter
    df.to_csv(filepath, index=False)
    print(f"[OK] 小水电曲线已保存: {filepath}")


def plot_pv_hydro_curves(selected, P_hydro_summer, P_hydro_winter, save_path):
    """绘制闽清光伏+小水电叠加曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scene_labels = list(selected.keys())
    colors_pv = ["#FDB813", "#7F7F7F", "#FF7F0E"]
    colors_hydro = ["#1F77B4", "#2CA02C"]

    for idx, (label, data) in enumerate(selected.items()):
        ax = axes[idx // 2][idx % 2]
        t = np.arange(24)

        ax.fill_between(t, 0, data["pv"], color=colors_pv[idx], alpha=0.5, label=f"光伏 ({data['date']})")
        if idx < 2:  # 前2个subplot画丰水期小水电
            ax.plot(t, data["pv"] + P_hydro_summer, color="#D62728", linewidth=2, label="光伏+小水电(丰水)")
        ax.plot(t, data["pv"], color=colors_pv[idx], linewidth=2.5)

        set_chinese_label(ax, xlabel="时间 (h)", ylabel="功率 (MW)", title=f"闽清{label} ({data['date']})")
        set_time_xticks(ax)
        ax.legend(prop={"family": "SimHei", "size": 9}, frameon=False)
        clean_axes(ax)

    # 右下：小水电单独曲线
    ax = axes[1][1]
    ax.plot(t, P_hydro_summer, color=colors_hydro[0], linewidth=2.5, label="丰水期小水电")
    ax.plot(t, P_hydro_winter, color=colors_hydro[1], linewidth=2.5, label="枯水期小水电")
    set_chinese_label(ax, xlabel="时间 (h)", ylabel="功率 (MW)", title="闽清径流式小水电典型出力")
    set_time_xticks(ax)
    ax.legend(prop={"family": "SimHei", "size": 10}, frameon=False)
    clean_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"   [CHART] 已保存: {save_path}")


# ══════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════
def main():
    print("=" * 60)
    print("模块一：分布式绿电出力预测技术")
    print("=" * 60)

    # --- 加载 ---
    params = load_params()
    df_nasa = load_nasa_power(params)

    ghi = df_nasa["GHI"].values
    t2m = df_nasa["T2M"].values

    # --- 计算光伏出力 ---
    P_pv = compute_pv_power(ghi, t2m, params)
    print(f"\n[CHART] 光伏出力计算完成: 总能量={P_pv.sum()/1000:.1f} MWh, "
          f"均值={P_pv.mean():.2f}MW, 峰值={P_pv.max():.2f}MW")

    # ==========================================
    # 层一：通用县域模型
    # ==========================================
    print("\n" + "─" * 40)
    print("【层一：通用县域模型 — K-Means典型日聚类】")
    print("─" * 40)

    typical_days, labels, P_matrix = cluster_typical_days(P_pv, n_clusters=params["simulation"]["kmeans_clusters"])

    # 保存CSV
    save_curves_csv(typical_days, params["data_paths"]["green_power_curves"])

    # 波动性分析
    stats = compute_fluctuation_stats(typical_days, params, P_matrix=P_matrix, labels=labels)

    # 保存波动性报告JSON（排除曲线数组，只保存标量指标）
    stats_json = {}
    for k in ["mean", "std", "max_ramp", "effective_h", "pv_ratio"]:
        stats_json[k] = {}
        for sk, sv in stats[k].items():
            stats_json[k][sk] = float(sv)
    report_path = "outputs/通用模型/fluctuation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"[OK] 波动性报告已保存: {report_path}")

    # 绘图
    curves_df = pd.read_csv(params["data_paths"]["green_power_curves"])
    curves_df.set_index("Hour", inplace=True)
    plot_green_power_curves(curves_df, "outputs/通用模型/green_power_curves.png")
    plot_fluctuation_analysis(stats, "outputs/通用模型/fluctuation_analysis.png")

    # ==========================================
    # 层二：闽清县验证
    # ==========================================
    print("\n" + "─" * 40)
    print("【层二：闽清县验证 — 真实日期+小水电】")
    print("─" * 40)

    # 选取验证日期
    selected = select_validation_dates(df_nasa, P_pv, params)

    # 保存原始日期CSV
    save_minqing_raw_days(selected, params["data_paths"]["minqing_raw_days"])

    # 构造小水电
    P_hydro_summer = build_hydro_curve(params, season="summer")
    P_hydro_winter = build_hydro_curve(params, season="winter")
    save_hydro_curve(P_hydro_summer, P_hydro_winter, params["data_paths"]["minqing_hydro_curve"])

    # 绘图
    plot_pv_hydro_curves(selected, P_hydro_summer, P_hydro_winter, "outputs/闽清验证/minqing_pv_hydro_curves.png")

    # ==========================================
    # 汇总
    # ==========================================
    print("\n" + "=" * 60)
    print("模块一完成！输出文件清单：")
    print("=" * 60)
    print("层一（通用模型）：")
    print(f"  [OK] {params['data_paths']['green_power_curves']}")
    print("  [OK] outputs/通用模型/green_power_curves.png")
    print("  [OK] outputs/通用模型/fluctuation_analysis.png")
    print("  [OK] outputs/通用模型/fluctuation_report.json")
    print("层二（闽清验证）：")
    print(f"  [OK] {params['data_paths']['minqing_raw_days']}")
    print(f"  [OK] {params['data_paths']['minqing_hydro_curve']}")
    print("  [OK] outputs/闽清验证/minqing_pv_hydro_curves.png")


if __name__ == "__main__":
    main()
