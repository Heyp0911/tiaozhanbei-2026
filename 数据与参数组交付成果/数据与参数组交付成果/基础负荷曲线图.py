import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# 1. 顶刊级绘图全局参数设置 (全局新罗马，局部黑体)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ==========================================
# 2. 核心参数假设 (请填入你的【交付物1: 参数表】)
# ==========================================
P_peak = 50.0  # 假设闽清县该节点年度最高峰值负荷为 50 MW
days = 365
hours = 24

# ==========================================
# 3. 构造行业标准化日负荷基线 (Normalized Base Curves)
# ==========================================
t = np.arange(24)
# 居民负荷 (早晚双峰)
res_base = 0.3 + 0.2 * np.exp(-0.5 * ((t - 8) / 1.5) ** 2) + 0.5 * np.exp(-0.5 * ((t - 20) / 2.5) ** 2)
# 工商业负荷 (白昼平顶)
ind_base = 0.2 + 0.7 * (1 / (1 + np.exp(-2 * (t - 8)))) * (1 / (1 + np.exp(2 * (t - 18))))
# 农业及市政 (平缓+夜间微峰)
agr_base = 0.5 + 0.1 * np.sin(np.pi * t / 12)

# ==========================================
# 4. 8760小时全年负荷自下而上合成
# ==========================================
annual_load = np.zeros(days * hours)
for day in range(days):
    # a. 季节波动乘子 (夏季和冬季出现双高峰)
    # 用余弦波模拟，1月和7-8月处于峰值，春秋处于谷底
    season_multiplier = 0.85 + 0.15 * np.cos(4 * np.pi * (day - 200) / 365)

    # b. 周末/节假日折减 (假设每周六日负荷下降 15%)
    weekend_multiplier = 0.85 if day % 7 >= 5 else 1.0

    # c. 每日基线合成 (权重: 居民 40%, 工商 45%, 农业市政 15%)
    daily_profile = 0.4 * res_base + 0.45 * ind_base + 0.15 * agr_base

    # d. 注入高斯白噪声模拟随机性 (均值0，标准差3%)
    noise = np.random.normal(0, 0.03, hours)

    # e. 计算当日 24 小时绝对负荷 (MW)
    actual_daily = P_peak * daily_profile * season_multiplier * weekend_multiplier * (1 + noise)

    # 防止负荷跌破绝对底线
    actual_daily = np.maximum(actual_daily, P_peak * 0.15)
    annual_load[day * hours: (day + 1) * hours] = actual_daily

# ==========================================
# 5. K-means 典型日场景聚类
# ==========================================
L_matrix = annual_load.reshape((days, hours))

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(L_matrix)
typical_loads = kmeans.cluster_centers_

# 将聚类结果按全天总用电量进行排序 (方便定义场景)
order = np.argsort(typical_loads.sum(axis=1))[::-1]
typical_loads = typical_loads[order]

# ==========================================
# 6. 论文级高质量可视化输出
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

# 使用与绿电图不同的色系区分，这里采用学术紫/青配色
colors = ['#4B0082', '#008080', '#DAA520', '#B22222']
labels = ['夏/冬极端高峰日', '工作日重载日', '春秋平稳日', '周末/节假日轻载日']
line_styles = ['-', '--', '-.', ':']

for i in range(n_clusters):
    ax.plot(t, typical_loads[i],
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=2.5,
            label=labels[i])

# 局部显式指定中文黑体
ax.set_xlabel('时间 (h)', family='SimHei', fontsize=12, fontweight='bold')
ax.set_ylabel('基础电网负荷 (MW)', family='SimHei', fontsize=12, fontweight='bold')

ax.set_xticks(np.arange(0, 25, 2))
ax.set_xlim([0, 23])
ax.set_ylim([0, P_peak * 1.1])  # 留出顶部空间

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

# 图例局部显式指定中文黑体
ax.legend(prop={'family': 'SimHei', 'size': 11}, frameon=False, loc='lower right')

plt.tight_layout()
save_path = r"D:\杂七杂八材料\典型日基础负荷曲线_KMeans.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"\n✅ 负荷聚类完成！图表已保存至: {save_path}")