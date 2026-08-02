"""
plot_utils.py — 统一绘图风格

论文级 matplotlib 图表输出，全局英文/数字锁定 Times New Roman，
中文通过 family='SimHei' 局部指定。
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np

# ══════════════════════════════════════════
# 全局参数设置（顶刊级）
# ══════════════════════════════════════════
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2

# 中文字体回退检测
_available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'SimHei' in _available_fonts:
    CHINESE_FONT = 'SimHei'
elif 'Microsoft YaHei' in _available_fonts:
    CHINESE_FONT = 'Microsoft YaHei'
elif 'WenQuanYi Micro Hei' in _available_fonts:
    CHINESE_FONT = 'WenQuanYi Micro Hei'
else:
    CHINESE_FONT = 'sans-serif'
    print("[WARN] 未找到中文字体(SimHei/Microsoft YaHei/WenQuanYi)，图表中文标签可能显示为方框。"
          "请安装中文字体: pip install matplotlib --upgrade 或手动安装 SimHei.ttf")

# 学术配色方案（色盲友好）
COLORS = {
    "pv": "#FDB813",         # 光伏-金黄色
    "grid": "#D62728",       # 电网-红
    "battery_dis": "#1F77B4", # 储能放电-蓝
    "battery_ch": "#2CA02C",  # 储能充电-绿
    "load_base": "#7F7F7F",   # 基础负荷-灰
    "rigid_task": "#8B0000",  # 刚性任务-暗红
    "elastic_task": "#FF7F0E",# 弹性任务-橙
    "cold_task": "#9467BD",   # 温冷任务-紫
    "idle_power": "#BCBD22",  # 空闲功耗-黄绿
    "scenario_1": "#7F7F7F",  # 场景一-灰
    "scenario_2": "#D62728",  # 场景二-红
    "scenario_3": "#1F77B4",  # 场景三-蓝（算随电走）
}

COLORS_SCENARIO = ["#7F7F7F", "#D62728", "#1F77B4"]
SCENARIO_LABELS = [
    "场景一：无算力",
    "场景二：固定算力（电随算走）",
    "场景三：弹性算力（算随电走）",
]


def set_chinese_label(ax, xlabel=None, ylabel=None, title=None):
    """局部设置中文标签，避免与全局 Times New Roman 冲突"""
    if xlabel:
        ax.set_xlabel(xlabel, family=CHINESE_FONT, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, family=CHINESE_FONT, fontsize=12, fontweight="bold")
    if title:
        ax.set_title(title, family=CHINESE_FONT, fontsize=14, fontweight="bold")


def clean_axes(ax):
    """去除顶部和右侧边框，添加虚线网格"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)


def set_time_xticks(ax, hours=None):
    """设置X轴为24小时格式"""
    if hours is None:
        hours = np.arange(0, 24)
    ax.set_xticks(hours[::2])
    ax.set_xlim([0, 23])


def save_and_show(fig, save_path, dpi=600):
    """统一保存和显示"""
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"   [CHART] 已保存: {save_path}")


def plot_power_balance_stack(
    pv_use,
    p_grid,
    p_dis,
    pv_curt,
    p_ch,
    p_load_base,
    p_rigid,
    p_elastic,
    p_cold,
    p_idle,
    scenario_label,
    save_path,
    pue=1.0,
):
    """
    功率平衡与调度结果图（论文标准画法）

    上方子图：源侧 — 24 小时堆叠柱状图，每根柱子 = 光伏(黄) + 电网购电(红) + 储能放电(蓝)
               叠加黑色折线 = 总负荷（电网侧实际功率，含PUE），直观验证供需平衡
    下方子图：荷侧 — 24 小时堆叠柱状图，每根柱子 = 基础负荷(灰) + 刚性任务(暗红)
               + 弹性任务(橙) + 温冷任务(紫) + 储能充电(绿) + 空闲(黄绿)
               （算力项已×PUE转为电网侧实际功率）
               叠加光伏可用出力折线(黄色虚线)，展示"算随电走"的调度逻辑

    PUE说明：算力负荷(p_rigid/p_elastic/p_cold/p_idle)为IT侧功率，
             需×PUE得到电网侧实际功率。MILP模型中已含PUE，此函数同步。
    """
    t = np.arange(24)

    # 算力IT侧 → 电网侧（×PUE）
    p_rigid_ac = p_rigid * pue
    p_elastic_ac = p_elastic * pue
    p_cold_ac = p_cold * pue
    p_idle_ac = p_idle * pue

    fig, (ax_src, ax_load) = plt.subplots(2, 1, figsize=(12, 8))

    width = 0.7

    # ====== 上方：源侧堆叠 ======
    # 堆叠顺序：光伏 → 电网购电 → 储能放电（自下而上）
    src_bottom = np.zeros(24)
    for comp, label, color in [
        (pv_use, "光伏利用", COLORS["pv"]),
        (p_grid, "电网购电", COLORS["grid"]),
        (p_dis, "储能放电", COLORS["battery_dis"]),
    ]:
        ax_src.bar(t, comp, width, bottom=src_bottom, color=color, alpha=0.85, label=label, edgecolor='white', linewidth=0.2)
        src_bottom = src_bottom + comp

    # 叠加总负荷线（电网侧实际功率，含PUE+储能充电 → 应与源侧堆叠高度相等）
    total_load = p_load_base + p_rigid_ac + p_elastic_ac + p_cold_ac + p_ch + p_idle_ac
    ax_src.plot(t, total_load, 'ko-', linewidth=2.0, markersize=4, label="总负荷(供需验证)", zorder=5)

    # 叠加算力专线（不含储能充电 → 直观展示"算随电走"vs"电随算走"）
    compute_only = p_load_base + p_rigid_ac + p_elastic_ac + p_cold_ac + p_idle_ac
    ax_src.plot(t, compute_only, 'D--', color='#2CA02C', linewidth=2.2, markersize=6,
                markerfacecolor='white', label="算力负荷(不含储能)", zorder=6)

    set_chinese_label(ax_src, ylabel="功率 (MW)", title=f"{scenario_label} — 源侧调度")
    set_time_xticks(ax_src)
    ax_src.legend(prop={"family": CHINESE_FONT, "size": 8}, frameon=False, loc="upper left", ncol=5)
    clean_axes(ax_src)

    # ====== 下方：荷侧堆叠（电网侧实际功率，含PUE）======
    load_bottom = np.zeros(24)
    for comp, label, color in [
        (p_load_base, "基础负荷", COLORS["load_base"]),
        (p_rigid_ac, "刚性算力任务", COLORS["rigid_task"]),
        (p_elastic_ac, "弹性算力任务", COLORS["elastic_task"]),
        (p_cold_ac, "温冷算力任务", COLORS["cold_task"]),
        (p_ch, "储能充电", COLORS["battery_ch"]),
        (p_idle_ac, "节点空闲功耗", COLORS["idle_power"]),
    ]:
        ax_load.bar(t, comp, width, bottom=load_bottom, color=color, alpha=0.85, label=label, edgecolor='white', linewidth=0.2)
        load_bottom = load_bottom + comp

    # 叠加光伏可用出力（虚线），展示"算随电走"调度依据
    pv_avail_est = pv_use + pv_curt
    ax_load.plot(t, pv_avail_est, '--', color=COLORS["pv"], linewidth=2.5, markersize=5, marker='s', label="光伏可用出力", zorder=5)

    set_chinese_label(ax_load, xlabel="时间 (h)", ylabel="功率 (MW)", title="荷侧构成（虚线=光伏可用出力，算力项已×PUE）")
    set_time_xticks(ax_load)
    ax_load.legend(prop={"family": CHINESE_FONT, "size": 8}, frameon=False, loc="upper left", ncol=3)
    clean_axes(ax_load)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"   [CHART] 已保存: {save_path}")


def plot_scenario_comparison_bar(metrics_list, metric_names, save_path):
    """
    三场景指标对比柱状图（双面板：百分比指标 + 绝对值指标）

    自动将指标分为两组：
      · 面板A：百分比/比率类指标（名称含 %、率）→ 共用 0-100% Y轴
      · 面板B：绝对量类指标 → 各自独立 Y轴

    参数
    ----------
    metrics_list : list of dict
        [场景一dict, 场景二dict, 场景三dict]
    metric_names : list of str
        要绘制的指标名称列表
    """
    # 分离指标：百分比类 vs 绝对量类
    pct_names = [n for n in metric_names if '%' in n or '率' in n or '比' in n]
    abs_names = [n for n in metric_names if n not in pct_names]

    # 如果一类为空，全放一起（回退到原逻辑）
    if not pct_names or not abs_names:
        pct_names = metric_names
        abs_names = []

    n_panels = 2 if abs_names else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5.5 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    width = 0.25

    # ── 面板A：百分比类指标 ──
    ax = axes[0]
    x_a = np.arange(len(pct_names))
    max_bar_height = 0
    bars_all = []
    for i, (metrics, label, color) in enumerate(
        zip(metrics_list, SCENARIO_LABELS, COLORS_SCENARIO)
    ):
        values = [metrics.get(name, 0) for name in pct_names]
        bars = ax.bar(x_a + i * width, values, width, label=label, color=color, alpha=0.85, edgecolor='white', linewidth=0.3)
        bars_all.append(bars)
        max_bar_height = max(max_bar_height, max(values))

    # 标注数值
    for bars in bars_all:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                        f"{h:.1f}", ha='center', va='bottom', fontsize=7.5,
                        family=CHINESE_FONT)

    set_chinese_label(ax, ylabel="百分比 / 比率")
    ax.set_xticks(x_a + width)
    ax.set_xticklabels(pct_names, family=CHINESE_FONT, fontsize=10, rotation=20, ha="right")
    ax.set_ylim(0, max(max_bar_height * 1.15, 105))
    ax.set_title("面板A：比率类指标", family=CHINESE_FONT, fontsize=13, fontweight="bold")
    ax.legend(prop={"family": CHINESE_FONT, "size": 10}, frameon=False, loc="upper right")
    clean_axes(ax)

    # ── 面板B：绝对量类指标（归一化到各自最大值）──
    if abs_names:
        ax = axes[1]
        x_b = np.arange(len(abs_names))

        # 为每个指标单独归一化：各自的最大值映射到1.0
        for i, (metrics, label, color) in enumerate(
            zip(metrics_list, SCENARIO_LABELS, COLORS_SCENARIO)
        ):
            raw_values = [metrics.get(name, 0) for name in abs_names]
            # 找出该指标在三场景中的最大值
            all_vals = []
            for j, name in enumerate(abs_names):
                all_vals.append(max(m.get(name, 1e-6) for m in metrics_list))
            max_vals = np.array(all_vals)
            norm_values = np.array(raw_values) / (max_vals + 1e-10)
            bars = ax.bar(x_b + i * width, norm_values, width, label=label,
                          color=color, alpha=0.85, edgecolor='white', linewidth=0.3)

            # 在每个柱子上标注实际数值
            for bar, raw in zip(bars, raw_values):
                if raw > 0:
                    # 格式化：大于100用整数，小于1用两位小数，其余用一位小数
                    if raw >= 100:
                        label_text = f"{raw:.0f}"
                    elif raw >= 1:
                        label_text = f"{raw:.1f}"
                    else:
                        label_text = f"{raw:.3f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            label_text,
                            ha='center', va='bottom', fontsize=7,
                            family=CHINESE_FONT, rotation=90)

        set_chinese_label(ax, ylabel="归一化比值（各指标 / 自身最大值）")
        ax.set_xticks(x_b + width)
        ax.set_xticklabels(abs_names, family=CHINESE_FONT, fontsize=10, rotation=20, ha="right")
        ax.set_title("面板B：绝对量类指标（归一化，柱上标注为实际值）", family=CHINESE_FONT, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.25)
        ax.legend(prop={"family": CHINESE_FONT, "size": 10}, frameon=False, loc="upper right")
        clean_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"   [CHART] 已保存: {save_path} (面板A: {len(pct_names)}个指标, 面板B: {len(abs_names)}个指标)")


def plot_soc_curves(soc_curves_dict, save_path):
    """
    多组储能容量的SOC曲线对比

    参数
    ----------
    soc_curves_dict : dict
        { "0 MWh": array(24,), "3 MWh": array(24,), ... }
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(24)

    linestyles = ["--", "-.", ":", "-"]
    colors_soc = ["#7F7F7F", "#FF7F0E", "#2CA02C", "#1F77B4"]

    for (label, soc_curve), ls, color in zip(
        soc_curves_dict.items(), linestyles[:len(soc_curves_dict)], colors_soc[:len(soc_curves_dict)]
    ):
        # SOC转换为百分比
        ax.plot(t, soc_curve, linestyle=ls, color=color, linewidth=2.0, label=label, marker="o", markersize=3)

    # 标注SOC上下限
    ax.axhline(y=1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5)

    set_chinese_label(ax, xlabel="时间 (h)", ylabel="储能SOC (比例)")
    set_time_xticks(ax)
    ax.set_ylim([0, 1.05])
    ax.legend(prop={"family": CHINESE_FONT, "size": 10}, frameon=False)
    clean_axes(ax)

    save_and_show(fig, save_path)


def plot_sensitivity_storage(capacities, absorption_rates, total_costs, save_path):
    """
    储能灵敏度分析 — 双Y轴图

    参数
    ----------
    capacities : list of float
        储能容量 [0, 3, 5, 10]
    absorption_rates : list of float
        对应的绿电消纳率 (%)
    total_costs : list of float
        对应的系统总成本 (元)
    """
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # 左Y轴：绿电消纳率
    color1 = "#1F77B4"
    ax1.plot(capacities, absorption_rates, "o-", color=color1, linewidth=2.0, markersize=8, label="绿电消纳率")
    set_chinese_label(ax1, xlabel="储能容量 (MWh)", ylabel="绿电消纳率 (%)")
    ax1.tick_params(axis="y", labelcolor=color1)

    # 右Y轴：系统成本
    ax2 = ax1.twinx()
    color2 = "#D62728"
    ax2.plot(capacities, total_costs, "s--", color=color2, linewidth=2.0, markersize=8, label="系统总成本")
    ax2.set_ylabel("系统总成本 (元)", family=CHINESE_FONT, fontsize=12, fontweight="bold", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, prop={"family": CHINESE_FONT, "size": 10}, frameon=False, loc="best")

    set_time_xticks(ax1, hours=capacities)
    ax1.set_xlim([-0.5, max(capacities) + 2])
    clean_axes(ax1)

    save_and_show(fig, save_path)


def plot_green_power_curves(curves_df, save_path, title="典型日绿电出力曲线"):
    """四条典型日光伏出力曲线叠加图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(24)

    colors = ["#D62728", "#1F77B4", "#FF7F0E", "#2CA02C"]
    labels = ["典型场景一", "典型场景二", "典型场景三", "典型场景四"]
    linestyles = ["-", "--", "-.", ":"]

    for i, col in enumerate(curves_df.columns):
        ax.plot(t, curves_df[col].values, color=colors[i], linestyle=linestyles[i], linewidth=2.5, label=labels[i])

    set_chinese_label(ax, xlabel="时间 (h)", ylabel="光伏出力功率 (MW)", title=title)
    set_time_xticks(ax)
    ax.legend(prop={"family": CHINESE_FONT, "size": 11}, frameon=False, loc="upper left")
    clean_axes(ax)

    save_and_show(fig, save_path)


def plot_fluctuation_analysis(stats_dict, save_path):
    """波动性分析 — 箱线图 + 置信带"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：4场景的波动性箱线数据
    # 这里用简化方式：柱状图展示标准差和爬坡率
    if "std" in stats_dict:
        scenes = list(stats_dict["std"].keys())
        x = np.arange(len(scenes))
        width = 0.35

        std_vals = [stats_dict["std"][s] for s in scenes]
        ramp_vals = [stats_dict["max_ramp"][s] for s in scenes]

        ax1.bar(x - width/2, std_vals, width, label="标准差 (MW)", color="#1F77B4", alpha=0.8)
        ax1.bar(x + width/2, ramp_vals, width, label="最大爬坡率 (MW/h)", color="#D62728", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"场景{i+1}" for i in range(len(scenes))], family=CHINESE_FONT)
        ax1.legend(prop={"family": CHINESE_FONT, "size": 9}, frameon=False)
        clean_axes(ax1)

    # 右图：置信带（以场景一为例）
    if "ci_upper" in stats_dict and "ci_lower" in stats_dict:
        t = np.arange(24)
        first_scene = list(stats_dict["ci_upper"].keys())[0]
        ax2.fill_between(t, stats_dict["ci_lower"][first_scene], stats_dict["ci_upper"][first_scene],
                         alpha=0.3, color="#1F77B4", label="95%置信区间")
        curve_vals = stats_dict.get("curve", {}).get(first_scene, stats_dict["ci_upper"][first_scene] * 0.5)
        ax2.plot(t, curve_vals, color="#1F77B4", linewidth=2, label="均值")
        set_chinese_label(ax2, xlabel="时间 (h)", ylabel="光伏出力 (MW)", title=f"场景{first_scene[-1]} 置信带")
        set_time_xticks(ax2)
        ax2.legend(prop={"family": CHINESE_FONT, "size": 9}, frameon=False)
        clean_axes(ax2)

    set_chinese_label(ax1, xlabel="", ylabel="波动性指标", title="波动性分析")

    save_and_show(fig, save_path)


def plot_topsis_ranking(tech_names, c_values, save_path):
    """TOPSIS排序水平柱状图"""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # 按C_i降序排列
    sorted_idx = np.argsort(c_values)[::-1]
    sorted_names = [tech_names[i] for i in sorted_idx]
    sorted_values = [c_values[i] for i in sorted_idx]

    colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_names)))

    bars = ax.barh(sorted_names, sorted_values, color=colors_bar[::-1], edgecolor="white", height=0.6)

    # 数值标注
    for bar, val in zip(bars, sorted_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=11)

    set_chinese_label(ax, xlabel="相对贴近度 C_i", ylabel="", title="关键技术TOPSIS综合评价")
    ax.set_xlim([0, max(sorted_values) * 1.15])
    ax.tick_params(axis="y", labelsize=10)
    for label in ax.get_yticklabels():
        label.set_family(CHINESE_FONT)
    clean_axes(ax)

    save_and_show(fig, save_path)


if __name__ == "__main__":
    print("plot_utils.py 加载完成，可用函数：")
    print("  plot_power_balance_stack()")
    print("  plot_scenario_comparison_bar()")
    print("  plot_soc_curves()")
    print("  plot_sensitivity_storage()")
    print("  plot_green_power_curves()")
    print("  plot_fluctuation_analysis()")
    print("  plot_topsis_ranking()")
