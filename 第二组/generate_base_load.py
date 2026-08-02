"""
generate_base_load.py — 生成基础负荷曲线CSV

复用数据组的构造方法（行业标准化日负荷基线），输出为CSV供模块二使用。
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.clustering import kmeans_manual

# 参数
import json
with open('params.json', 'r', encoding='utf-8') as _f:
    _params = json.load(_f)
P_peak = _params.get('base_load', {}).get('P_peak_MW', 50.0)  # 从params.json读取
days = 365
hours = 24
np.random.seed(42)

# 构造标准化日负荷基线
t = np.arange(24)
res_base = 0.3 + 0.2 * np.exp(-0.5 * ((t - 8) / 1.5) ** 2) + 0.5 * np.exp(-0.5 * ((t - 20) / 2.5) ** 2)
ind_base = 0.2 + 0.7 * (1 / (1 + np.exp(-2 * (t - 8)))) * (1 / (1 + np.exp(2 * (t - 18))))
agr_base = 0.5 + 0.1 * np.sin(np.pi * t / 12)

# 8760小时全年负荷合成
annual_load = np.zeros(days * hours)
for day in range(days):
    season_multiplier = 0.85 + 0.15 * np.cos(4 * np.pi * (day - 200) / 365)
    weekend_multiplier = 0.85 if day % 7 >= 5 else 1.0
    # 从 params.json 读取负荷结构比例（默认值对应通用县域模型）
    load_config = _params.get('base_load', {})
    res_r = load_config.get('residential_ratio', 0.40)
    ind_r = load_config.get('industrial_ratio', 0.45)
    agr_r = load_config.get('agriculture_ratio', 0.15)
    assert abs(res_r + ind_r + agr_r - 1.0) < 0.01, f"负荷比例之和应等于1.0，当前: {res_r}+{ind_r}+{agr_r}={res_r+ind_r+agr_r}"
    daily_profile = res_r * res_base + ind_r * ind_base + agr_r * agr_base
    noise = np.random.normal(0, 0.03, hours)
    actual_daily = P_peak * daily_profile * season_multiplier * weekend_multiplier * (1 + noise)
    actual_daily = np.maximum(actual_daily, P_peak * 0.15)
    annual_load[day * hours: (day + 1) * hours] = actual_daily

# K-means 手动聚类
L_matrix = annual_load.reshape((days, hours))

centroids, _ = kmeans_manual(L_matrix, n_clusters=4)
order = np.argsort(centroids.sum(axis=1))[::-1]
typical_loads = centroids[order]

# 场景命名
scene_names = ["夏/冬极端高峰日", "工作日重载日", "春秋平稳日", "周末/节假日轻载日"]

# 保存CSV
df = pd.DataFrame()
df["Hour"] = np.arange(24)
for i, name in enumerate(scene_names):
    df[name] = typical_loads[i]

df.to_csv("data/base_load_curves.csv", index=False)
print("[OK] base_load_curves.csv 已保存")
print(f"   4条典型日负荷曲线: {scene_names}")
for i, name in enumerate(scene_names):
    print(f"   {name}: 均值={typical_loads[i].mean():.1f}MW, 峰值={typical_loads[i].max():.1f}MW, 谷值={typical_loads[i].min():.1f}MW")
