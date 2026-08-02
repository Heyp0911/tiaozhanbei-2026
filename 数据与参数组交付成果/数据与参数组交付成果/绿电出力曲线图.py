import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# 1. 顶刊级绘图全局参数设置 (全局数字/英文锁定新罗马)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'  # 核心修改：全局锁定新罗马，确保坐标轴数字完美渲染
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'  # 刻度线向内(学术规范)
plt.rcParams['ytick.direction'] = 'in'

# ==========================================
# 2. 智能读取 Excel 数据 (自动跳过元数据并修复 float 报错)
# ==========================================
file_path = r"D:\杂七杂八材料\NASA POWER 原始气象数据说明文档（闽清县-2025全整年）.xlsx"

try:
    print("⚙️ 正在启动智能表头扫描...")
    # 先不设表头，将整个表作为纯数据读入
    df_raw = pd.read_excel(file_path, header=None)

    # 动态寻找真正的表头所在行 (扫描前30行)
    header_row_index = -1
    for i in range(min(30, len(df_raw))):
        row_values = df_raw.iloc[i].values
        if any('ALLSKY_SFC_SW_DWN' in str(val) for val in row_values):
            header_row_index = i
            break

    if header_row_index == -1:
        raise ValueError("❌ 扫描了前30行，未能找到包含 'ALLSKY_SFC_SW_DWN' 的表头，请确认文件内容！")

    print(f"🎯 成功锁定表头！位于 Excel 的第 {header_row_index + 1} 行。正在提取数据...")

    # 重新读取，指定真正的表头行
    df = pd.read_excel(file_path, header=header_row_index)
    df.columns = df.columns.str.strip()

    ghi = pd.to_numeric(df['ALLSKY_SFC_SW_DWN'], errors='coerce').fillna(0).values
    temp = pd.to_numeric(df['T2M'], errors='coerce').fillna(25).values

    # ==========================================
    # 3. 带有环境温度修正的光伏物理出力模型
    # ==========================================
    P_rated = 10.0
    G_STC = 1000.0
    T_STC = 25.0
    k = -0.0043

    P_pv = P_rated * (ghi / G_STC) * (1 + k * (temp - T_STC))
    P_pv = np.maximum(P_pv, 0)

    # ==========================================
    # 4. K-means 典型日场景聚类
    # ==========================================
    total_hours = len(P_pv)
    days = total_hours // 24
    if days > 365:
        days = 365

    hours = 24
    P_matrix = P_pv[:days * hours].reshape((days, hours))

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(P_matrix)

    typical_days = kmeans.cluster_centers_

    # ==========================================
    # 5. 论文级高质量可视化输出 (无标题)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#D62728', '#1F77B4', '#FF7F0E', '#2CA02C']
    labels = ['典型场景一', '典型场景二', '典型场景三', '典型场景四']
    line_styles = ['-', '--', '-.', ':']

    time_axis = np.arange(0, 24)
    for i in range(n_clusters):
        ax.plot(time_axis, typical_days[i],
                color=colors[i],
                linestyle=line_styles[i],
                linewidth=2.5,
                label=labels[i])

    # 💡 核心修复点：通过 family='SimHei' 局部强制指定中文为黑体，避开新罗马的吞字 Bug
    ax.set_xlabel('时间 (h)', family='SimHei', fontsize=12, fontweight='bold')
    ax.set_ylabel('光伏出力功率 (MW)', family='SimHei', fontsize=12, fontweight='bold')

    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim([0, 23])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 💡 核心修复点：图例部分同样强制赋予黑体属性
    ax.legend(prop={'family': 'SimHei', 'size': 12}, frameon=False, loc='upper left')

    plt.tight_layout()

    save_path = r"D:\杂七杂八材料\典型日绿电出力曲线_KMeans.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 完美收工！中英双规字体图表已保存至: {save_path}")

except Exception as e:
    print(f"❌ 运行中出现错误: {e}")