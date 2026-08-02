"""
irr_calc.py — IRR对比计算（三个场景：无算力/电随算走/算随电走）

============================================================
基于MILP仿真结果 → 年度化 → 20年现金流 → IRR比较
============================================================

使用方法：
  python irr_calc.py
"""

import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_params, load_nasa_power
from utils.clustering import kmeans_manual
from green_power_predictor import compute_pv_power
from milp_optimizer import run_milp_scenario


# ══════════════════════════════════════════
# 1. 年度化：K-Means聚类 → 各典型日加权求和
# ══════════════════════════════════════════
def annualize_milp_results(params, n_pv_clusters=3):
    """
    用K-Means(k=3)对全年365天光伏聚类 → 对每个聚类中心跑MILP → 按天数加权年度化

    返回
    ----------
    annual : dict
        {scenario: {"grid_cost": 元, "curtail_mwh": MWh, "carbon_kg": kg, ...}}
    """
    print("=" * 60)
    print("[1/4] 年度化：K-Means聚类 + MILP批量仿真")
    print("=" * 60)

    # 加载气象 + 计算光伏
    df = load_nasa_power(params)
    P_pv = compute_pv_power(df["GHI"].values, df["T2M"].values, params)

    days = 365
    hours = 24
    P_matrix = P_pv[:days * hours].reshape(days, hours)

    # K-Means
    centroids, labels = kmeans_manual(P_matrix, n_clusters=n_pv_clusters, random_state=42)

    # 按峰值排序
    order = np.argsort(centroids.max(axis=1))
    cluster_info = []
    for rank, idx in enumerate(order):
        count = np.sum(labels == idx)
        curve = centroids[idx]
        cluster_info.append({
            "name": f"PV_{['low','mid','high'][rank]}",
            "count": int(count),
            "curve": curve,
            "peak": float(curve.max()),
            "daily_total": float(curve.sum()),
        })
        print(f"   {cluster_info[-1]['name']}: {count}天, 峰值={curve.max():.1f}MW, 日总={curve.sum():.1f}MWh")

    # 对每个聚类中心跑三场景MILP
    base_load = np.zeros(24)
    annual = {"none": {}, "fixed": {}, "elastic": {}}
    scenario_keys = {"none": "S1_无算力", "fixed": "S2_电随算走", "elastic": "S3_算随电走"}

    for ci in cluster_info:
        pv_avail = ci["curve"]
        count = ci["count"]
        print(f"\n--- {ci['name']} ({count}天) ---")

        for sk, sname in scenario_keys.items():
            r = run_milp_scenario(pv_avail, base_load, params, scenario=sk)
            daily_cost = r["objective"]  # 元/天
            daily_pv_use = float(np.sum(r["P_pv_use"]))  # MWh
            daily_pv_curt = float(np.sum(r["P_pv_curt"]))
            daily_grid = float(np.sum(r["P_grid"]))  # MWh
            daily_carbon = r["metrics"]["碳排放量(kgCO2)"]

            # 累加年度值
            for key in ["grid_cost", "pv_use_mwh", "pv_curt_mwh", "grid_mwh", "carbon_kg"]:
                if key not in annual[sk]:
                    annual[sk][key] = 0.0
            annual[sk]["grid_cost"] += daily_cost * count
            annual[sk]["pv_use_mwh"] += daily_pv_use * count
            annual[sk]["pv_curt_mwh"] += daily_pv_curt * count
            annual[sk]["grid_mwh"] += daily_grid * count
            annual[sk]["carbon_kg"] += daily_carbon * count

            if sk == "elastic":
                if "elastic_adj_mwh" not in annual[sk]:
                    annual[sk]["elastic_adj_mwh"] = 0.0
                annual[sk]["elastic_adj_mwh"] += r["metrics"].get("弹性任务调节量(MWh)", 0) * count

    # 打印年度汇总
    print(f"\n{'='*60}")
    print(f"年度汇总 (365天)")
    print(f"{'='*60}")
    for sk in ["none", "fixed", "elastic"]:
        a = annual[sk]
        print(f"  {scenario_keys[sk]}: 电网成本={a['grid_cost']:.0f}元, "
              f"PV消纳={a['pv_use_mwh']:.0f}MWh, 弃光={a['pv_curt_mwh']:.0f}MWh, "
              f"碳排={a['carbon_kg']:.0f}kg")

    return annual


# ══════════════════════════════════════════
# 2. 投资计算
# ══════════════════════════════════════════
def calc_investment(params):
    """
    计算三个场景的初始投资

    返回
    ----------
    inv : dict
        {scenario: {"total": 元, "pv": 元, "battery": 元, "compute": 元}}
    """
    print("\n" + "=" * 60)
    print("[2/4] 投资估算")
    print("=" * 60)

    irr_cfg = params["irr"]
    pv = params["pv_system"]
    bat = params["battery"]
    comp = params["compute_node"]

    # 光伏投资（三场景相同）
    pv_kw = pv["P_rated_MW"] * 1000
    pv_cost = pv_kw * irr_cfg["pv_system"]["unit_cost_CNY_per_kW"]

    # 储能投资（三场景相同）
    bat_kwh = bat["E_cap_MWh__default"] * 1000
    bat_cost = bat_kwh * bat["epc_cost_CNY_per_kWh"]

    # 算力节点投资（仅S2/S3）
    comp_kw = comp["P_node_rated_MW"] * 1000
    comp_cost = comp_kw * irr_cfg["compute_node"]["unit_cost_CNY_per_kW_IT"]

    inv = {
        "none": {
            "pv": pv_cost, "battery": bat_cost, "compute": 0,
            "total": pv_cost + bat_cost,
        },
        "fixed": {
            "pv": pv_cost, "battery": bat_cost, "compute": comp_cost,
            "total": pv_cost + bat_cost + comp_cost,
        },
        "elastic": {
            "pv": pv_cost, "battery": bat_cost, "compute": comp_cost,
            "total": pv_cost + bat_cost + comp_cost,
        },
    }

    for sk, sname in [("none", "S1_无算力"), ("fixed", "S2_电随算走"), ("elastic", "S3_算随电走")]:
        v = inv[sk]
        print(f"  {sname}: PV={v['pv']/1e4:.0f}万 储能={v['battery']/1e4:.0f}万 "
              f"算力={v['compute']/1e4:.0f}万 总计={v['total']/1e4:.0f}万元")

    return inv


# ══════════════════════════════════════════
# 3. 现金流模型 + IRR计算（手写Newton法，无需numpy_financial）
# ══════════════════════════════════════════
def npv(rate, cashflows):
    """计算净现值"""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def compute_irr(cashflows, tol=1e-6, max_iter=200):
    """Newton法计算IRR，多初始猜测遍历"""
    # 尝试多个初始猜测
    for guess in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.01, 0.001]:
        rate = guess
        for _ in range(max_iter):
            f = npv(rate, cashflows)
            if abs(f) < tol:
                if 0.001 <= rate <= 0.99:
                    return rate
                break  # 超出合理范围，试下一个
            df = (npv(rate + 1e-4, cashflows) - f) / 1e-4
            if abs(df) < 1e-10:
                break
            rate_new = rate - f / df
            if abs(rate_new - rate) < tol:
                if 0.001 <= rate_new <= 0.99:
                    return rate_new
                break
            rate = rate_new
            if rate <= -0.99 or rate >= 2.0:
                break
    return 0.0  # 无法收敛返回0


def build_cashflows(params, annual, inv):
    """
    构建20年现金流

    返回
    ----------
    cf : dict {scenario: [year0, year1, ..., year20]}
    """
    print("\n" + "=" * 60)
    print("[3/4] 现金流模型")
    print("=" * 60)

    irr_cfg = params["irr"]
    life = irr_cfg["project_life_years"]
    tax = irr_cfg["income_tax_rate"]
    ai_price = irr_cfg["revenue"]["ai_service_price_CNY_per_kWh"]
    ai_hours = irr_cfg["revenue"]["annual_ai_service_hours"]

    # 省电基准：S1（无算力）的年电网成本
    baseline_grid_cost = annual["none"]["grid_cost"]

    cashflows = {}
    for sk, sname in [("none", "S1_无算力"), ("fixed", "S2_电随算走"), ("elastic", "S3_算随电走")]:
        cf = [0.0] * (life + 1)
        total_inv = inv[sk]["total"]
        # AI服务节点能耗全部计入算力负荷，年收益=节点IT年耗电×AI单价
        if sk == "none":
            ai_revenue = 0.0
        else:
            node_it_mw = params["compute_node"]["P_node_rated_MW"]  # 5 MW
            ai_revenue = node_it_mw * 1000 * ai_hours * ai_price  # 元/年

        # Year 0: 初始投资
        cf[0] = -total_inv

        for yr in range(1, life + 1):
            # 省电收益 = 相比S1省下的电网电费
            grid_saving = baseline_grid_cost - annual[sk]["grid_cost"]

            # 年运维成本
            om = (inv[sk]["pv"] * irr_cfg["pv_system"]["om_rate"]
                  + inv[sk]["battery"] * irr_cfg["battery"]["om_rate"]
                  + inv[sk]["compute"] * irr_cfg["compute_node"]["om_rate"])

            revenue = grid_saving + ai_revenue
            ebit = revenue - om

            # 折旧
            dep = (inv[sk]["pv"] / irr_cfg["pv_system"]["useful_life_years"]
                   + inv[sk]["battery"] / irr_cfg["battery"]["useful_life_years"]
                   + inv[sk]["compute"] / max(irr_cfg["compute_node"]["useful_life_years"], 1))

            taxable = max(0, ebit - dep)
            net_income = ebit - taxable * tax
            cf[yr] = net_income

            # 储能更换
            bat_life = irr_cfg["battery"]["useful_life_years"]
            for repl_yr in range(bat_life, life + 1, bat_life):
                if yr == repl_yr:
                    cf[yr] -= inv[sk]["battery"]

            # 算力节点更新（基础设施15年，GPU服务器8年刷新50%）
            if sk != "none":
                refresh_interval = irr_cfg["compute_node"]["refresh_interval_years"]
                refresh_rate = irr_cfg["compute_node"]["refresh_rate"]
                comp_life = irr_cfg["compute_node"]["useful_life_years"]
                # GPU服务器定期刷新
                for repl_yr in range(refresh_interval, life + 1, refresh_interval):
                    if yr == repl_yr and repl_yr % comp_life != 0:
                        cf[yr] -= inv[sk]["compute"] * refresh_rate
                # 基础设施完全更新
                for repl_yr in range(comp_life, life + 1, comp_life):
                    if yr == repl_yr:
                        cf[yr] -= inv[sk]["compute"]

        # 期末残值
        terminal = 0.0
        terminal += inv[sk]["pv"] * (1 - life / irr_cfg["pv_system"]["useful_life_years"]) * irr_cfg["pv_system"]["salvage_rate"]
        bat_remaining = (irr_cfg["battery"]["useful_life_years"] - life % irr_cfg["battery"]["useful_life_years"]) / irr_cfg["battery"]["useful_life_years"]
        terminal += inv[sk]["battery"] * max(0, bat_remaining) * irr_cfg["battery"]["salvage_rate"]
        if sk != "none":
            comp_age = life % irr_cfg["compute_node"]["useful_life_years"]
            comp_remaining = (irr_cfg["compute_node"]["useful_life_years"] - comp_age) / irr_cfg["compute_node"]["useful_life_years"]
            terminal += inv[sk]["compute"] * max(0, comp_remaining) * irr_cfg["compute_node"]["salvage_rate"]
        cf[life] += terminal

        cashflows[sk] = cf

        irr_val = compute_irr(cf)
        total_inflow = sum(cf[1:])
        payback = None
        cum = 0.0
        for yr in range(life + 1):
            cum += cf[yr]
            if cum >= 0 and payback is None:
                payback = yr + (0 if cum == cf[yr] else -cf[yr-1]/cf[yr] if yr>0 else 0) - 1
                payback = round(payback, 1)
        if payback is None:
            payback = ">20年"

        print(f"  {sname}: IRR={irr_val*100:.2f}%  "
              f"年净利润均={total_inflow/life/1e4:.1f}万元  "
              f"回收期={payback}年")

    return cashflows


# ══════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════
def main():
    print("=" * 60)
    print("IRR对比计算 — 三个场景投资回报分析")
    print("=" * 60)

    params = load_params()

    # Step 1: 年度化
    annual = annualize_milp_results(params)

    # Step 2: 投资估算
    inv = calc_investment(params)

    # Step 3: 现金流+IRR
    cashflows = build_cashflows(params, annual, inv)

    # Step 4: 输出汇总表
    print("\n" + "=" * 60)
    print("[4/4] 汇总对比")
    print("=" * 60)

    scenario_labels = {"none": "S1_无算力", "fixed": "S2_电随算走", "elastic": "S3_算随电走"}
    results = []
    for sk in ["none", "fixed", "elastic"]:
        cf = cashflows[sk]
        irr_val = compute_irr(cf)
        avg_profit = sum(cf[1:]) / params["irr"]["project_life_years"]
        total_in = -cf[0]
        cum = 0.0
        payback = ">20"
        for yr in range(len(cf)):
            cum += cf[yr]
            if cum >= 0:
                # 线性插值
                if yr == 0 or cf[yr] == 0:
                    payback = str(yr)
                else:
                    frac = -cum_before / cf[yr] if yr > 0 else 0
                    payback = "%.1f" % (yr - 1 + frac)
                break
            cum_before = cum

        a = annual[sk]
        results.append({
            "场景": scenario_labels[sk],
            "总投资(万元)": round(total_in / 1e4, 1),
            "年电网成本(万元)": round(a["grid_cost"] / 1e4, 1),
            "年PV消纳(MWh)": round(a["pv_use_mwh"], 1),
            "年碳排(kg)": round(a["carbon_kg"], 1),
            "年净利润(万元)": round(avg_profit / 1e4, 1),
            "IRR(%)": round(irr_val * 100, 2) if irr_val else "N/A",
            "回收期(年)": payback,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # 保存
    df.to_csv("outputs/irr_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n[OK] outputs/irr_comparison.csv 已保存")

    # 关键结论
    irr_s2 = compute_irr(cashflows["fixed"])
    irr_s3 = compute_irr(cashflows["elastic"])
    if irr_s2 and irr_s3:
        delta_irr = (irr_s3 - irr_s2) * 100
        print(f"\n关键结论: S3(算随电走) IRR={irr_s3*100:.2f}% 比 S2(电随算走) IRR={irr_s2*100:.2f}% 高 {delta_irr:.2f}个百分点")
        print(f"相同硬件投入下，仅通过弹性调度即提升了 {delta_irr:.2f}pp 的投资回报率")


if __name__ == "__main__":
    main()
