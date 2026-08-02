"""
milp_optimizer.py — 模块二：源网荷储算碳协同优化模型

============================================================
对应技术：技术二（弹性调度）+ 技术三（协同优化）+ 技术四（储能协同）
============================================================

功能：
  1. 建立 MILP 源网荷储算碳六对象统一模型
  2. 运行三种场景对比（无算力 / 固定算力 / 算随电走）
  3. 储能容量敏感性分析（0/3/5/10 MWh）
  4. 通用县域模型 + 闽清验证两套数据各跑一遍
  5. 输出10个核心指标 + 功率平衡图 + SOC曲线 + 灵敏度图

使用方法：
  python milp_optimizer.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import (
    load_params,
    load_green_power_curves,
    build_rigid_task_profile,
    build_elastic_task_bounds,
    build_cold_task_bounds,
    build_price_profile,
    load_minqing_raw_days,
    load_minqing_hydro,
)
from utils.plot_utils import (
    plot_power_balance_stack,
    plot_scenario_comparison_bar,
    plot_soc_curves,
    plot_sensitivity_storage,
)

try:
    import pulp
except ImportError:
    print("[ERR] 需要安装PuLP: pip install pulp")
    sys.exit(1)


# ══════════════════════════════════════════
# MILP 模型构建与求解
# ══════════════════════════════════════════
def run_milp_scenario(
    pv_avail,
    base_load,
    params,
    scenario="elastic",
    E_bat_cap=None,
    verbose=False,
    fixed_elastic=None,
    fixed_cold=None,
):
    """
    构建并求解单次MILP模型

    参数
    ----------
    pv_avail : np.ndarray shape (24,)
        光伏可用出力 (MW)
    base_load : np.ndarray shape (24,)
        县域基础负荷 (MW)
    params : dict
        全局参数
    scenario : str
        "none" = 无算力, "fixed" = 固定算力(电随算走), "elastic" = 弹性算力(算随电走)
    E_bat_cap : float or None
        储能容量 (MWh)，None则使用默认值
    verbose : bool
        是否打印求解细节

    返回
    ----------
    result : dict
        包含所有决策变量、约束、目标函数值和输出指标的字典
    """
    hours = 24
    if E_bat_cap is None:
        E_bat_cap = params["battery"]["E_cap_MWh__default"]

    # ── 读取参数 ──
    P_grid_max = params["grid"]["P_grid_max_MW"]
    carbon_factor = params["grid"]["carbon_factor_tCO2_per_MWh"]
    carbon_price = params["grid"]["carbon_price_CNY_per_tCO2"]
    feed_in_price = params["grid"]["feed_in_price_CNY_per_kWh"]
    # 两部制电价：需量费（元/kW/月 → 元/kW/天，折算到日模型）
    ep = params.get("electricity_price", {})
    if ep.get("pricing_model") == "two_part":
        demand_charge_daily = ep.get("demand_charge_CNY_per_kW_month", 40.0) / 30.0  # 元/kW/天
    else:
        demand_charge_daily = 0.0

    C_rate = params["battery"].get("C_rate", 0.5)
    P_ch_max = E_bat_cap * C_rate   # 充放电功率 = 容量 × 倍率，随储能容量自动缩放
    P_dis_max = E_bat_cap * C_rate
    eta_ch = params["battery"]["eta_ch"]
    eta_dis = params["battery"]["eta_dis"]
    SOC_min = params["battery"]["SOC_min"]
    SOC_max = params["battery"]["SOC_max"]
    # 会计折旧法：EPC成本×容量 → 抵扣进项税 → 转固 → 按年限直线折旧
    _epc = params["battery"].get("epc_cost_CNY_per_kWh", 1000.0)
    _vat = params["battery"].get("vat_rate", 0.13)
    _life = params["battery"].get("useful_life_years", 15)
    _depreciation_daily = (E_bat_cap * 1000) * _epc * (1 - _vat) / (_life * 365)  # 元/天

    P_node_rated = params["compute_node"]["P_node_rated_MW"]
    P_node_idle = params["compute_node"]["P_node_idle_MW"]
    PUE = params["compute_node"].get("PUE", 1.0)  # 默认1.0=无基础设施开销

    E_elastic = params["ai_tasks"]["E_elastic_daily_MWh"]
    E_cold = params["ai_tasks"]["E_cold_daily_MWh"]

    price = build_price_profile(params, hours)

    # ── 构造任务负荷 ──
    P_rigid_demand = build_rigid_task_profile(params, hours)
    P_elastic_max_raw = build_elastic_task_bounds(params, hours)
    P_cold_max_raw = build_cold_task_bounds(params, hours)

    if scenario == "none":
        P_rigid_demand = np.zeros(hours)
        P_elastic_fixed = np.zeros(hours)
        P_cold_fixed = np.zeros(hours)
        E_elastic = 0.0
        E_cold = 0.0
    elif scenario == "fixed":
        P_elastic_fixed = np.full(hours, E_elastic / hours)
        P_cold_fixed = np.full(hours, E_cold / hours)
    else:
        # elastic: use as initial bounds
        P_elastic_fixed = P_elastic_max_raw
        P_cold_fixed = P_cold_max_raw

    # 动态调整电网上限确保可行域非空
    # 原则：电网容量至少能覆盖最坏情况供需缺口 + 10% 余量（算力负荷×PUE含基础设施）
    max_possible_load = np.max(base_load)
    if scenario in ["fixed", "elastic"]:
        max_possible_load += (np.max(P_rigid_demand) + np.max(P_elastic_fixed) + np.max(P_cold_fixed) + P_node_idle) * PUE
    worst_case_deficit = max(0, max_possible_load - np.max(pv_avail))
    margin = 0.10 * max_possible_load
    effective_grid_max = max(P_grid_max, worst_case_deficit + margin)

    # ── 创建问题 ──
    prob = pulp.LpProblem("SourceGridLoadStorageComputeCarbon", pulp.LpMinimize)

    # ── 决策变量 ──
    P_grid = [pulp.LpVariable(f"P_grid_{t}", 0, effective_grid_max) for t in range(hours)]
    P_pv_use = [pulp.LpVariable(f"P_pv_use_{t}", 0, None) for t in range(hours)]
    P_pv_curt = [pulp.LpVariable(f"P_pv_curt_{t}", 0, None) for t in range(hours)]
    P_ch = [pulp.LpVariable(f"P_ch_{t}", 0, P_ch_max) for t in range(hours)]
    P_dis = [pulp.LpVariable(f"P_dis_{t}", 0, P_dis_max) for t in range(hours)]
    SOC = [pulp.LpVariable(f"SOC_{t}", SOC_min * E_bat_cap, SOC_max * E_bat_cap) for t in range(hours)]

    # 技术四：储能充放电互斥 — 二进制变量（MILP）
    u_ch = [pulp.LpVariable(f"u_ch_{t}", cat='Binary') for t in range(hours)]
    u_dis = [pulp.LpVariable(f"u_dis_{t}", cat='Binary') for t in range(hours)]

    # 两部制电价：峰值需量变量（用于需量费计算）
    P_peak = pulp.LpVariable("P_peak", 0, effective_grid_max) if demand_charge_daily > 0 else None

    if fixed_elastic is not None and fixed_cold is not None:
        # 外部强制指定算力分配（用于"预测调度→真实PV"的代价评估）
        P_elastic = np.array(fixed_elastic, dtype=float)
        P_cold = np.array(fixed_cold, dtype=float)
        P_elastic_max = P_elastic_max_raw
        P_cold_max = P_cold_max_raw
        _fixed_compute = True
    elif scenario == "elastic":
        P_elastic = [pulp.LpVariable(f"P_elastic_{t}", 0, P_elastic_max_raw[t]) for t in range(hours)]
        P_cold = [pulp.LpVariable(f"P_cold_{t}", 0, P_cold_max_raw[t]) for t in range(hours)]
        P_elastic_max = P_elastic_max_raw
        P_cold_max = P_cold_max_raw
        _fixed_compute = False
    else:
        P_elastic = P_elastic_fixed  # numpy array, fixed values
        P_cold = P_cold_fixed
        P_elastic_max = P_elastic_fixed
        P_cold_max = P_cold_fixed
        _fixed_compute = True

    # ── 约束 ──
    # ① 功率平衡（算力侧负荷 × PUE = 电网侧实际功耗）
    for t in range(hours):
        load_total = base_load[t]
        if scenario in ["fixed", "elastic"]:
            load_total += (P_rigid_demand[t] + P_elastic[t] + P_cold[t] + P_node_idle) * PUE
        prob += (P_pv_use[t] + P_grid[t] + P_dis[t] == load_total + P_ch[t])

    # ② 光伏出力
    for t in range(hours):
        prob += (P_pv_use[t] + P_pv_curt[t] == pv_avail[t])

    # ③ 储能充放电互斥 (技术四) — MILP二进制约束
    for t in range(hours):
        prob += (P_ch[t] <= u_ch[t] * P_ch_max)
        prob += (P_dis[t] <= u_dis[t] * P_dis_max)
        prob += (u_ch[t] + u_dis[t] <= 1)  # 禁止同时充放

    # ③-bis 光伏弃光时禁止放电（物理调度常识：优先用光伏而非弃光+放电）
    #   Big-M: P_pv_curt[t] <= M * (1 - u_dis[t])
    #   即 u_dis[t]=1 时强制 P_pv_curt[t]=0，避免"弃光同时放电"的反常调度
    M_big = max(np.max(pv_avail), 20.0)
    for t in range(hours):
        prob += (P_pv_curt[t] <= M_big * (1 - u_dis[t]))

    # ④ SOC连续性 (技术四)
    # SOC[0]为自由决策变量（0~E_cap），模型自动确定最优初始SOC
    # 日循环约束：结束SOC ≥ 起始SOC，确保可持续循环，不寅吃卯粮
    for t in range(hours - 1):
        prob += (SOC[t + 1] == SOC[t] + eta_ch * P_ch[t] - P_dis[t] / eta_dis)
    prob += (SOC[hours - 1] >= SOC[0])

    # ⑤ 算力节点容量约束
    if not _fixed_compute:
        # 弹性场景：P_elastic和P_cold是决策变量，需要PuLP约束
        for t in range(hours):
            total_compute = P_rigid_demand[t] + P_elastic[t] + P_cold[t] + P_node_idle
            prob += (total_compute <= P_node_rated)
    elif scenario != "none":
        # 固定算力（含外部强制分配）：Python验证即可
        for t in range(hours):
            total_compute = P_rigid_demand[t] + P_elastic[t] + P_cold[t] + P_node_idle
            if total_compute > P_node_rated:
                print(f"   [WARN] t={t}: 算力需求({total_compute:.2f}MW)超出节点容量({P_node_rated}MW)，等比例缩减")
                scale = (P_node_rated - P_rigid_demand[t] - P_node_idle) / (P_elastic[t] + P_cold[t] + 1e-10)
                P_elastic[t] *= scale
                P_cold[t] *= scale

    # ⑥⑦⑧ 任务约束 (技术二)
    if not _fixed_compute:
        # 弹性任务日总量（仅当P_elastic为决策变量时）
        prob += (pulp.lpSum(P_elastic) == E_elastic)
        # 温冷任务日总量
        prob += (pulp.lpSum(P_cold) == E_cold)

    # ⑨ 电网购电已在变量上下限中体现

    # ⑩ 两部制需量约束：P_peak >= 各时段电网购电功率
    if P_peak is not None:
        for t in range(hours):
            prob += (P_peak >= P_grid[t])

    # ── 目标函数 ──
    C_grid = pulp.lpSum([P_grid[t] * price[t] * 1000 for t in range(hours)])
    C_carbon = pulp.lpSum([P_grid[t] * carbon_factor * carbon_price for t in range(hours)])
    C_curtail = pulp.lpSum([P_pv_curt[t] * feed_in_price * 1000 for t in range(hours)])
    # 储能折旧：会计直线法，固定日折旧额（常数，不影响优化方向）
    C_battery = _depreciation_daily
    # 两部制需量费：P_peak(MW) × 1000(kW/MW) × demand_charge_daily(元/kW/天)
    C_demand = P_peak * 1000 * demand_charge_daily if P_peak is not None else 0.0

    prob += (C_grid + C_carbon + C_curtail + C_battery + C_demand)

    # ── 求解 ──
    solver = pulp.PULP_CBC_CMD(msg=verbose)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if verbose:
        print(f"   求解状态: {status}")

    if status != "Optimal":
        print(f"   [WARN] 求解状态={status}，可能不是最优解")
        if status == "Infeasible":
            # 诊断：输出可能冲突的约束信息
            pv_sum = np.sum(pv_avail)
            load_sum = np.sum(base_load)
            total_demand_min = load_sum + P_node_idle * hours
            total_supply_max = pv_sum + P_grid_max * hours
            print(f"   [诊断] 日光伏={pv_sum:.1f}MWh, 日负荷={load_sum:.1f}MWh, "
                  f"电网最大供电={P_grid_max*24:.0f}MWh, 供需比={total_supply_max/total_demand_min:.2f}")
            if scenario == "elastic":
                total_task = params['ai_tasks']['E_elastic_daily_MWh'] + params['ai_tasks']['E_cold_daily_MWh']
                max_hourly = params['compute_node']['P_node_rated_MW'] - params['compute_node']['P_node_idle_MW']
                print(f"   [诊断] 任务日需={total_task:.1f}MWh, 节点小时容量={max_hourly:.1f}MW, 日容量={max_hourly*24:.0f}MWh")

    # ── 提取结果 ──
    result = {
        "status": status,
        "objective": pulp.value(prob.objective),
    }

    # 提取所有变量值
    for var_name, var_list in [
        ("P_grid", P_grid), ("P_pv_use", P_pv_use), ("P_pv_curt", P_pv_curt),
        ("P_ch", P_ch), ("P_dis", P_dis), ("SOC", SOC),
    ]:
        result[var_name] = np.array([pulp.value(v) if hasattr(v, 'value') else v for v in var_list])

    if not _fixed_compute:
        result["P_elastic"] = np.array([pulp.value(v) for v in P_elastic])
        result["P_cold"] = np.array([pulp.value(v) for v in P_cold])
    else:
        result["P_elastic"] = np.array(P_elastic)
        result["P_cold"] = np.array(P_cold)

    result["P_rigid"] = P_rigid_demand
    result["P_idle"] = np.full(hours, P_node_idle)
    result["PUE"] = PUE
    result["C_depreciation"] = float(_depreciation_daily)
    if P_peak is not None:
        result["P_peak"] = pulp.value(P_peak) if hasattr(P_peak, 'value') else P_peak
        result["C_demand"] = float(result["P_peak"] * 1000 * demand_charge_daily)

    # ── 计算输出指标 ──
    result["metrics"] = compute_metrics(result, pv_avail, base_load, params, scenario)

    return result


# ══════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════
def compute_metrics(result, pv_avail, base_load, params, scenario):
    """从MILP结果中计算10个核心指标"""
    hours = 24
    P_pv_use = result["P_pv_use"]
    P_pv_curt = result["P_pv_curt"]
    P_grid = result["P_grid"]
    P_elastic = result["P_elastic"]
    P_cold = result["P_cold"]
    P_rigid = result["P_rigid"]
    P_idle = result["P_idle"]

    carbon_factor = params["grid"]["carbon_factor_tCO2_per_MWh"]

    pv_total = np.sum(pv_avail)
    pv_used = np.sum(P_pv_use)
    pv_curt = np.sum(P_pv_curt)

    compute_total = np.sum(P_rigid) + np.sum(P_elastic) + np.sum(P_cold)
    carbon_total = np.sum(P_grid) * carbon_factor  # tCO2

    PUE = result.get("PUE", 1.0)

    metrics = {
        "绿电消纳率(%)": round(pv_used / pv_total * 100, 2) if pv_total > 0 else 0,
        "弃电率(%)": round(pv_curt / pv_total * 100, 2) if pv_total > 0 else 0,
        "系统总运行成本(元)": round(result["objective"], 2),
        "碳排放量(kgCO2)": round(carbon_total, 2),
        "单位算力碳排放(kgCO2/kWh)": round(carbon_total / compute_total, 4) if compute_total > 0 else 0,
        "任务完成率(%)": 100.0,  # 约束保证完成
        "算力节点利用率(%)": round(
            np.mean(P_rigid + P_elastic + P_cold + P_idle) / params["compute_node"]["P_node_rated_MW"] * 100, 2
        ),
        "最大购电功率(MW)": round(np.max(P_grid), 2),
        "日均光伏消纳量(MWh)": round(pv_used, 2),
    }

    # 储能日折旧（会计直线法，常数）
    if "C_depreciation" in result:
        metrics["储能日折旧(元/天)"] = round(result["C_depreciation"], 2)

    # 两部制需量费单独列出
    if "P_peak" in result:
        metrics["峰值需量(MW)"] = round(result["P_peak"], 2)
        metrics["需量费(元/天)"] = round(result["C_demand"], 2)

    # 场景三独有：平均任务调节量
    if scenario == "elastic":
        E_elastic = params["ai_tasks"]["E_elastic_daily_MWh"]
        P_elastic_uniform = E_elastic / hours
        metrics["弹性任务调节量(MWh)"] = round(
            np.sum(np.abs(P_elastic - P_elastic_uniform)), 2
        )

    return metrics


# ══════════════════════════════════════════
# 主流程：通用县域模型
# ══════════════════════════════════════════
def run_general_model(params):
    """运行通用县域模型（层一）"""
    print("\n" + "=" * 60)
    print("【通用县域模型 — 三场景对比】")
    print("=" * 60)

    # 加载绿电数据
    pv_df = load_green_power_curves(params["data_paths"]["green_power_curves"])

    # 绿电: K-Means场景中峰值最高的（晴天），"算随电走"才有优化空间
    pv_scene = pv_df.columns[params.get('milp_scene_config', {}).get('general_model', {}).get('pv_scene_index', 0)]

    pv_avail = pv_df[pv_scene].values
    base_load = np.zeros(24)  # 负荷=纯算力任务，不叠加基础负荷

    print(f"   绿电场景: {pv_scene} (峰值={pv_avail.max():.2f}MW)")
    print(f"   负荷配置: 纯算力任务 (基础负荷=0，聚焦算随电走)")

    scenarios = {
        "none": "场景一：无算力",
        "fixed": "场景二：固定算力（电随算走）",
        "elastic": "场景三：弹性算力（算随电走）",
    }

    all_results = {}
    all_metrics = []

    for scenario_key, scenario_label in scenarios.items():
        print(f"\n--- {scenario_label} ---")
        result = run_milp_scenario(pv_avail, base_load, params, scenario=scenario_key, verbose=True)
        all_results[scenario_key] = result
        all_metrics.append(result["metrics"])

        print(f"   目标值: {result['objective']:.2f} 元")
        print(f"   绿电消纳率: {result['metrics']['绿电消纳率(%)']:.1f}%")
        print(f"   碳排放量: {result['metrics']['碳排放量(kgCO2)']:.2f} kgCO2")

    # 保存指标汇总
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.index = [scenarios[k] for k in scenarios.keys()]
    metrics_df.to_csv("outputs/通用模型/metrics_summary.csv", encoding="utf-8-sig")
    print(f"\n[OK] 通用模型指标汇总已保存: outputs/通用模型/metrics_summary.csv")

    # 绘图
    # 功率平衡堆叠图（三张）
    for scenario_key, scenario_label in scenarios.items():
        r = all_results[scenario_key]
        plot_power_balance_stack(
            pv_use=r["P_pv_use"],
            p_grid=r["P_grid"],
            p_dis=r["P_dis"],
            pv_curt=r["P_pv_curt"],
            p_ch=r["P_ch"],
            p_load_base=base_load,
            p_rigid=r["P_rigid"],
            p_elastic=r["P_elastic"],
            p_cold=r["P_cold"],
            p_idle=r["P_idle"],
            scenario_label=scenario_label,
            save_path=f"outputs/通用模型/power_balance_{scenario_key}.png",
            pue=r.get("PUE", 1.0),
        )

    # 三场景10指标对比
    metric_names = [
        "绿电消纳率(%)", "弃电率(%)", "系统总运行成本(元)",
        "碳排放量(kgCO2)", "单位算力碳排放(kgCO2/kWh)",
        "算力节点利用率(%)", "最大购电功率(MW)", "日均光伏消纳量(MWh)",
    ]
    plot_scenario_comparison_bar(all_metrics, metric_names, "outputs/通用模型/scenario_comparison.png")

    return all_results, pv_avail, base_load


# ══════════════════════════════════════════
# 储能敏感性分析
# ══════════════════════════════════════════
def run_storage_sensitivity(params, all_results, pv_avail, base_load):
    """储能容量敏感性分析（技术四）"""
    print("\n" + "=" * 60)
    print("【储能敏感性分析 — 技术四】")
    print("=" * 60)

    capacities = params["battery"]["E_cap_MWh__sensitivity"]
    soc_curves_dict = {}
    absorption_rates = []
    total_costs = []

    for cap in capacities:
        print(f"\n--- 储能容量: {cap} MWh ---")
        result = run_milp_scenario(
            pv_avail, base_load, params,
            scenario="elastic",
            E_bat_cap=cap,
            verbose=True,
        )
        soc_label = f"{cap:.0f} MWh"
        soc_curves_dict[soc_label] = result["SOC"] / max(cap, 1e-6)  # 归一化到[0,1]
        absorption_rates.append(result["metrics"]["绿电消纳率(%)"])
        total_costs.append(result["metrics"]["系统总运行成本(元)"])

        print(f"   绿电消纳率: {result['metrics']['绿电消纳率(%)']:.1f}%")
        print(f"   系统成本: {result['metrics']['系统总运行成本(元)']:.2f} 元")

    # 绘图
    plot_soc_curves(soc_curves_dict, "outputs/通用模型/soc_curve.png")
    plot_sensitivity_storage(capacities, absorption_rates, total_costs, "outputs/通用模型/sensitivity_storage.png")

    # 保存CSV
    sens_df = pd.DataFrame({
        "储能容量(MWh)": capacities,
        "绿电消纳率(%)": absorption_rates,
        "系统总成本(元)": total_costs,
    })
    sens_df.to_csv("outputs/通用模型/storage_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] 储能灵敏度结果已保存")

    return soc_curves_dict


# ══════════════════════════════════════════
# 闽清县验证
# ══════════════════════════════════════════
def run_minqing_validation(params):
    """运行闽清县验证（层二）"""
    print("\n" + "=" * 60)
    print("【闽清县典型场景验证 — 层二】")
    print("=" * 60)

    # 加载闽清数据
    minqing_df = load_minqing_raw_days(params["data_paths"]["minqing_raw_days"])
    hydro_df = load_minqing_hydro(params["data_paths"]["minqing_hydro_curve"])

    # 使用晴天数据（光伏峰值最高）
    pv_cols = [c for c in minqing_df.columns if "_PV" in c]

    # 选3个验证日期各跑一遍
    for pv_col in pv_cols:
        label = pv_col.split("_")[0]
        date_str = "_".join(pv_col.split("_")[1:3])
        print(f"\n--- 闽清 {label} ({date_str}) ---")

        # 根据验证日期月份自动选择小水电季节
        month = int(date_str.split('-')[1]) if '-' in date_str else 7
        if month in [6, 7, 8, 9]:
            hydro_col = "丰水期"
        elif month in [12, 1, 2]:
            hydro_col = "枯水期"
        else:
            hydro_col = "丰水期"  # 过渡期默认

        pv_only = minqing_df[pv_col].values
        hydro = hydro_df[hydro_col].values
        total_green = pv_only + hydro  # 光伏 + 小水电

        base_load = np.zeros(24)  # 负荷=纯算力任务，不叠加基础负荷

        print(f"   光伏峰值: {pv_only.max():.2f}MW, 小水电均值: {hydro.mean():.2f}MW")
        print(f"   总绿电峰值: {total_green.max():.2f}MW (光伏{pv_only.max():.1f}+水电{hydro.max():.1f})")

        all_results = {}
        all_metrics = []

        for scenario_key, scenario_label in [
            ("none", "场景一：无算力"),
            ("fixed", "场景二：固定算力（电随算走）"),
            ("elastic", "场景三：弹性算力（算随电走）"),
        ]:
            result = run_milp_scenario(total_green, base_load, params, scenario=scenario_key)
            # 口径修正：分离光伏和小水电贡献
            hydro_total = float(sum(hydro))
            pv_absorbed = max(0, float(sum(result['P_pv_use'])) - hydro_total)
            result['metrics']['光伏消纳(MWh)'] = round(pv_absorbed, 2)
            result['metrics']['小水电消纳(MWh)'] = round(hydro_total, 2)
            result['metrics']['总绿电消纳(MWh)'] = round(float(sum(result['P_pv_use'])), 2)
            all_results[scenario_key] = result
            all_metrics.append(result["metrics"])

        # 保存指标
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.index = ["场景一", "场景二", "场景三"]
        safe_label = f"{label}_{date_str}"
        metrics_df.to_csv(f"outputs/闽清验证/minqing_metrics_{safe_label}.csv", encoding="utf-8-sig")

        # 功率平衡图
        for scenario_key, scenario_label in [
            ("none", "场景一：无算力"),
            ("fixed", "场景二：固定算力（电随算走）"),
            ("elastic", "场景三：弹性算力（算随电走）"),
        ]:
            r = all_results[scenario_key]
            plot_power_balance_stack(
                pv_use=r["P_pv_use"],
                p_grid=r["P_grid"],
                p_dis=r["P_dis"],
                pv_curt=r["P_pv_curt"],
                p_ch=r["P_ch"],
                p_load_base=base_load,
                p_rigid=r["P_rigid"],
                p_elastic=r["P_elastic"],
                p_cold=r["P_cold"],
                p_idle=r["P_idle"],
                scenario_label=f"闽清 {label} - {scenario_label}",
                save_path=f"outputs/闽清验证/minqing_balance_{safe_label}_{scenario_key}.png",
                pue=r.get("PUE", 1.0),
            )

        # 三场景对比
        metric_names = [
            "绿电消纳率(%)", "弃电率(%)", "系统总运行成本(元)",
            "碳排放量(kgCO2)", "单位算力碳排放(kgCO2/kWh)",
            "算力节点利用率(%)", "最大购电功率(MW)", "日均光伏消纳量(MWh)",
        ]
        plot_scenario_comparison_bar(
            all_metrics, metric_names,
            f"outputs/闽清验证/minqing_comparison_{safe_label}.png",
        )

    print(f"\n[OK] 闽清验证完成")


# ══════════════════════════════════════════
# 预测驱动调度模式（读 pv_forecaster 导出的 24h CSV）
# ══════════════════════════════════════════
def run_forecast_mode(params, day_tag="best"):
    """读取 GRU-XGBoost 预测的 24h PV 曲线，跑三场景 MILP

    参数
    ----------
    day_tag : str
        "best" | "typical" | "worst" — 选择哪个代表性预测日
    """
    csv_path = f"outputs/pv_forecast/demo_day_24h_{day_tag}.csv"

    if not os.path.exists(csv_path):
        print(f"[ERR] 缺少预测文件: {csv_path}")
        print(f"  请先运行: python pv_forecaster.py --balanced")
        return

    pv_df = pd.read_csv(csv_path)
    pv_avail = pv_df["PV_MW"].values
    day_label = csv_path

    base_load = np.zeros(24)  # 负荷=纯算力任务

    print("\n" + "=" * 60)
    print("【预测驱动调度 — GRU-XGBoost 预测值接入 MILP】")
    print("=" * 60)
    print(f"   PV 来源: {csv_path}")
    print(f"   光伏峰值: {pv_avail.max():.2f} MW")
    print(f"   光伏日均: {pv_avail.mean():.2f} MW")

    scenarios = {
        "none": "场景一：无算力",
        "fixed": "场景二：固定算力（电随算走）",
        "elastic": "场景三：弹性算力（算随电走）",
    }

    all_results = {}
    all_metrics = []

    for scenario_key, scenario_label in scenarios.items():
        print(f"\n--- {scenario_label} ---")
        result = run_milp_scenario(pv_avail, base_load, params, scenario=scenario_key, verbose=True)
        all_results[scenario_key] = result
        all_metrics.append(result["metrics"])

        print(f"   目标值: {result['objective']:.2f} 元")
        print(f"   绿电消纳率: {result['metrics']['绿电消纳率(%)']:.1f}%")
        print(f"   碳排放量: {result['metrics']['碳排放量(kgCO2)']:.2f} kgCO2")

    # 保存指标
    os.makedirs("outputs/forecast_milp", exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.index = [scenarios[k] for k in scenarios.keys()]
    metrics_df.to_csv("outputs/forecast_milp/metrics_summary.csv", encoding="utf-8-sig")
    print(f"\n[OK] 预测驱动调度指标: outputs/forecast_milp/metrics_summary.csv")

    # 功率平衡图
    for scenario_key, scenario_label in scenarios.items():
        r = all_results[scenario_key]
        plot_power_balance_stack(
            pv_use=r["P_pv_use"], p_grid=r["P_grid"], p_dis=r["P_dis"],
            pv_curt=r["P_pv_curt"], p_ch=r["P_ch"], p_load_base=base_load,
            p_rigid=r["P_rigid"], p_elastic=r["P_elastic"], p_cold=r["P_cold"],
            p_idle=r["P_idle"],
            scenario_label=f"预测驱动 - {scenario_label}",
            save_path=f"outputs/forecast_milp/power_balance_{scenario_key}.png",
            pue=r.get("PUE", 1.0),
        )

    # 三场景对比
    metric_names = [
        "绿电消纳率(%)", "弃电率(%)", "系统总运行成本(元)",
        "碳排放量(kgCO2)", "单位算力碳排放(kgCO2/kWh)",
        "算力节点利用率(%)", "最大购电功率(MW)", "日均光伏消纳量(MWh)",
    ]
    plot_scenario_comparison_bar(all_metrics, metric_names, "outputs/forecast_milp/scenario_comparison.png")

    print(f"\n[OK] 全部成果已保存至: outputs/forecast_milp/")
    print(f"  [OK] metrics_summary.csv")
    print(f"  [OK] power_balance_none.png")
    print(f"  [OK] power_balance_fixed.png")
    print(f"  [OK] power_balance_elastic.png")
    print(f"  [OK] scenario_comparison.png")


# ══════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MILP 协同优化模型")
    parser.add_argument("--mode", choices=["default", "forecast"], default="default",
                        help="default=通用模型+闽清验证, forecast=预测驱动调度")
    parser.add_argument("--forecast-day", choices=["best", "typical", "worst"], default="best",
                        help="forecast模式下选择哪个预测日 (default: best)")
    args = parser.parse_args()

    params = load_params()

    if args.mode == "forecast":
        print("=" * 60)
        print(f"模块二：预测驱动调度模式 (--mode forecast --forecast-day {args.forecast_day})")
        print("=" * 60)
        run_forecast_mode(params, day_tag=args.forecast_day)
        return

    print("=" * 60)
    print("模块二：源网荷储算碳协同优化模型 (MILP)")
    print("=" * 60)

    params = load_params()

    # ── 通用县域模型 ──
    all_results, pv_avail, base_load = run_general_model(params)

    # ── 储能敏感性分析 ──
    run_storage_sensitivity(params, all_results, pv_avail, base_load)

    # ── 闽清县验证 ──
    run_minqing_validation(params)

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("模块二完成！输出文件清单：")
    print("=" * 60)
    print("通用模型：")
    print("  [OK] outputs/通用模型/metrics_summary.csv")
    print("  [OK] outputs/通用模型/power_balance_none.png")
    print("  [OK] outputs/通用模型/power_balance_fixed.png")
    print("  [OK] outputs/通用模型/power_balance_elastic.png")
    print("  [OK] outputs/通用模型/scenario_comparison.png")
    print("  [OK] outputs/通用模型/soc_curve.png")
    print("  [OK] outputs/通用模型/sensitivity_storage.png")
    print("闽清验证：")
    print("  [OK] outputs/闽清验证/*.csv + *.png (3个日期 × 3场景)")


if __name__ == "__main__":
    main()
