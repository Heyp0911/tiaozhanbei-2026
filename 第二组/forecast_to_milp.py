"""
forecast_to_milp.py — 桥接脚本： 将 GRU-XGBoost 预测结果接入 MILP 协同优化

================================================================
闭合"预测 → 调度"回路：
  1. 运行 pv_forecaster 的 GRU-XGBoost 管线，获取测试集预测
  2. 将 15h 日照预测扩展为 24h 光伏出力曲线（夜间补零）
  3. 选取最佳/典型/最差预测日，分别跑 MILP 三场景
  4. 对比：用"完美预测(实际值)"调度 vs 用"GRU-XGBoost预测"调度
  5. 量化预测误差对调度质量的影响

输出：
  · outputs/forecast_milp_bridge/metrics_comparison.csv
  · outputs/forecast_milp_bridge/forecast_error_cost.png
  · outputs/forecast_milp_bridge/power_balance_best.png
  · outputs/forecast_milp_bridge/power_balance_worst.png

使用方法：
  python forecast_to_milp.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 复用已有模块 ──
from utils.data_loader import (
    load_params,
    build_rigid_task_profile, build_elastic_task_bounds,
    build_cold_task_bounds, build_price_profile,
)
from green_power_predictor import compute_pv_power
from pv_forecaster import (
    prepare_forecast_data, GRUForecaster, train_gru_model,
    to_tensor_loader, compute_metrics,
)
from milp_optimizer import run_milp_scenario
from utils.plot_utils import (
    COLORS, SCENARIO_LABELS, COLORS_SCENARIO,
    set_chinese_label, clean_axes, set_time_xticks,
    save_and_show, CHINESE_FONT,
)

import matplotlib.pyplot as plt
import torch

try:
    import xgboost as xgb
except ImportError:
    print("[ERR] 需要 xgboost: pip install xgboost")
    sys.exit(1)


# ══════════════════════════════════════════
# 1. 将 15h 预测转为 24h 光伏曲线
# ══════════════════════════════════════════
def pred15h_to_24h(pred_15h, start_hour=6, end_hour=20):
    """
    将 pv_forecaster 的 15h 日照时段预测扩展为 24h 逐时曲线

    参数
    ----------
    pred_15h : np.ndarray shape (15,)
        GRU-XGBoost 预测的 06:00-20:00 PV 出力 (MW)
    start_hour : int
        日照起始小时 (默认 6)
    end_hour : int
        日照结束小时 (默认 20，对应索引 20)

    返回
    ----------
    pv_24h : np.ndarray shape (24,)
        24h 光伏出力，夜间为 0
    """
    pv_24h = np.zeros(24)
    sun_len = end_hour - start_hour + 1  # = 15
    pv_24h[start_hour:end_hour + 1] = pred_15h[:sun_len]
    return np.maximum(pv_24h, 0)


# ══════════════════════════════════════════
# 2. 获取测试集的预测/实际 PV 曲线
# ══════════════════════════════════════════
def get_test_predictions(device='cpu'):
    """
    运行完整的 GRU-XGBoost 管线，返回测试集上每条样本的：
      - 完美 PV（实际值）
      - 预测 PV（GRU-XGBoost）
      - 测试日期标签
    """
    print("=" * 60)
    print("【阶段0：运行 GRU-XGBoost 预测管线】")
    print("=" * 60)

    params = load_params()
    X_gru, y, X_xgb_extra, norm_params, daily_masks = prepare_forecast_data(
        params, balanced_test=True)

    train_idx = daily_masks["train"]
    val_idx = daily_masks["val"]
    test_idx = daily_masks["test"]

    y_min = norm_params["PV_target"]["min"]
    y_max = norm_params["PV_target"]["max"]

    # ── 训练 GRU ──
    gru_params = {"units": 128, "dropout": 0.2, "lr": 0.01, "batch_size": 32}
    input_dim = X_gru.shape[2]

    model_gru = GRUForecaster(input_dim=input_dim, hidden_dim=gru_params['units'],
                               dropout=gru_params['dropout'], num_layers=2)

    train_loader = to_tensor_loader(X_gru[train_idx], y[train_idx],
                                     batch_size=gru_params['batch_size'], shuffle=True)
    val_loader = to_tensor_loader(X_gru[val_idx], y[val_idx],
                                   batch_size=gru_params['batch_size'], shuffle=False)

    print("训练 GRU...")
    model_gru, _, _ = train_gru_model(
        model_gru, train_loader, val_loader,
        epochs=300, lr=gru_params['lr'], patience=50, device=device, verbose=True,
    )

    # ── GRU 预测 ──
    model_gru.eval()
    with torch.no_grad():
        gru_pred_test = model_gru(
            torch.tensor(X_gru[test_idx], dtype=torch.float32).to(device)
        ).cpu().detach().numpy()
        gru_pred_train = model_gru(
            torch.tensor(X_gru[train_idx], dtype=torch.float32).to(device)
        ).cpu().detach().numpy()
        gru_pred_val = model_gru(
            torch.tensor(X_gru[val_idx], dtype=torch.float32).to(device)
        ).cpu().detach().numpy()

    # ── GRU 逐时偏差 ──
    gru_bias_temp = (gru_pred_val - y[val_idx]).mean(axis=0)

    # ── 训练 XGBoost 精炼 ──
    xgb_params = {"learning_rate": 0.26, "max_depth": 4,
                  "subsample": 0.91, "colsample_bytree": 0.97}

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
    X_xgb_test = np.concatenate([
        gru_pred_test,
        X_gru[test_idx].reshape(len(test_idx), -1),
        X_xgb_extra[test_idx].reshape(len(test_idx), -1),
        np.tile(gru_bias_temp, (len(test_idx), 1)),
    ], axis=1)

    print("训练 XGBoost 精炼...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=xgb_params['learning_rate'],
        max_depth=xgb_params['max_depth'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        objective='reg:squarederror', verbosity=0, n_jobs=-1,
    )
    model_xgb.fit(X_xgb_train, y[train_idx].reshape(len(train_idx), -1),
                  eval_set=[(X_xgb_val, y[val_idx].reshape(len(val_idx), -1))],
                  verbose=False)

    # ── 混合预测 ──
    y_hybrid_norm = model_xgb.predict(X_xgb_test)
    y_hybrid = y_hybrid_norm.reshape(len(test_idx), -1) * (y_max - y_min) + y_min

    # ── 实际值 (denormalize) ──
    y_actual = y[test_idx] * (y_max - y_min) + y_min

    # ── 日期标签 ──
    base_date = pd.Timestamp("2025-01-02")
    test_dates = [base_date + pd.Timedelta(days=int(i)) for i in test_idx]

    # ── 计算每天 nRMSE ──
    daily_errors = []
    for i in range(len(test_idx)):
        m = compute_metrics(y_actual[i], y_hybrid[i])
        daily_errors.append(m["nRMSE"])

    print(f"\n[OK] 预测管线完成：{len(test_idx)} 个测试日")
    print(f"   GRU-XGBoost 整体 nRMSE = {compute_metrics(y_actual.flatten(), y_hybrid.flatten())['nRMSE']:.4f}")

    return y_actual, y_hybrid, test_dates, daily_errors, params


# ══════════════════════════════════════════
# 3. 选取代表性预测日
# ══════════════════════════════════════════
def pick_demo_days(y_actual, y_hybrid, test_dates, daily_errors):
    """
    从测试集中选取 3 个代表性预测日

    返回
    ----------
    demo_days : dict
        { "最佳预测": {...}, "典型预测": {...}, "最差预测": {...} }
    """
    order = np.argsort(daily_errors)  # 升序
    best_idx = order[0]
    median_idx = order[len(order) // 2]
    worst_idx = order[-1]

    # 额外的代表性日期
    # 找 GHI 日均值最高的一天（高绿电日，调度压力最大）
    ghi_daily = y_actual.mean(axis=1)
    high_pv_idx = np.argmax(ghi_daily)
    # 找 GHI 波动最大的一天
    ghi_std = y_actual.std(axis=1)
    volatile_idx = np.argmax(ghi_std)

    demo_days = {
        "最佳预测": {
            "idx": best_idx,
            "date": test_dates[best_idx],
            "nRMSE": daily_errors[best_idx],
            "actual": y_actual[best_idx],
            "predicted": y_hybrid[best_idx],
            "ghi_daily": y_actual[best_idx].mean(),
            "desc": f"nRMSE最低 — 预测最准的一天"
        },
        "典型预测": {
            "idx": median_idx,
            "date": test_dates[median_idx],
            "nRMSE": daily_errors[median_idx],
            "actual": y_actual[median_idx],
            "predicted": y_hybrid[median_idx],
            "ghi_daily": y_actual[median_idx].mean(),
            "desc": f"nRMSE中位数 — 典型预测质量"
        },
        "最差预测": {
            "idx": worst_idx,
            "date": test_dates[worst_idx],
            "nRMSE": daily_errors[worst_idx],
            "actual": y_actual[worst_idx],
            "predicted": y_hybrid[worst_idx],
            "ghi_daily": y_actual[worst_idx].mean(),
            "desc": f"nRMSE最高 — 预测最差的一天，压力测试"
        },
    }

    print("\n【选取代表性预测日】")
    for label, d in demo_days.items():
        print(f"   {label}: {d['date'].strftime('%Y-%m-%d')}, "
              f"nRMSE={d['nRMSE']:.4f}, 日均PV={d['ghi_daily']:.1f}MW, {d['desc']}")

    return demo_days


# ══════════════════════════════════════════
# 4. 运行 MILP 调度对比
# ══════════════════════════════════════════
def run_scheduling_comparison(demo_days, params):
    """
    对每个代表性预测日，分别用实际 PV 和预测 PV 运行 MILP，
    对比调度质量差异。
    """
    print("\n" + "=" * 60)
    print("【阶段1：MILP 调度 — 预测 vs 完美】")
    print("=" * 60)

    # 负荷=纯算力任务，不叠加基础负荷
    base_load = np.zeros(24)

    all_comparisons = {}

    for day_label, day_data in demo_days.items():
        print(f"\n{'─' * 40}")
        print(f"【{day_label}】{day_data['date'].strftime('%Y-%m-%d')}  日均PV={day_data['ghi_daily']:.1f}MW,  nRMSE={day_data['nRMSE']:.4f}")
        print(f"{'─' * 40}")

        # 构造 24h PV 曲线
        pv_perfect_24h = pred15h_to_24h(day_data["actual"])
        pv_predicted_24h = pred15h_to_24h(day_data["predicted"])

        # MILP: 场景三 "算随电走"（弹性调度最能体现预测不准的代价）
        print(f"\n  [完美预测] 实际PV值驱动MILP — 调度质量上界")
        r_perfect = run_milp_scenario(
            pv_perfect_24h, base_load, params,
            scenario="elastic", verbose=False)
        print(f"    目标值: {r_perfect['objective']:.2f} 元"
              f"  |  消纳率: {r_perfect['metrics']['绿电消纳率(%)']:.1f}%"
              f"  |  碳排: {r_perfect['metrics']['碳排放量(kgCO2)']:.2f} kg")

        print(f"\n  [GRU-XGBoost预测] 预测PV值驱动MILP — 实际操作效果")
        r_pred = run_milp_scenario(
            pv_predicted_24h, base_load, params,
            scenario="elastic", verbose=False)
        print(f"    目标值: {r_pred['objective']:.2f} 元"
              f"  |  消纳率: {r_pred['metrics']['绿电消纳率(%)']:.1f}%"
              f"  |  碳排: {r_pred['metrics']['碳排放量(kgCO2)']:.2f} kg")

        # 误差代价（MILP自己认为的）
        cost_gap_planned = r_pred['objective'] - r_perfect['objective']

        # ── 实际执行成本：用预测出的算力分配在真实PV下执行 ──
        print(f"\n  [实际执行] 预测调度方案 + 真实PV — 实际会发生的成本")
        r_real = run_milp_scenario(
            pv_perfect_24h, base_load, params,
            scenario="elastic", verbose=False,
            fixed_elastic=r_pred["P_elastic"],
            fixed_cold=r_pred["P_cold"])
        cost_gap_real = r_real['objective'] - r_perfect['objective']
        print(f"    目标值: {r_real['objective']:.2f} 元"
              f"  |  消纳率: {r_real['metrics']['绿电消纳率(%)']:.1f}%"
              f"  |  碳排: {r_real['metrics']['碳排放量(kgCO2)']:.2f} kg")

        print(f"\n  成本对比: 完美={r_perfect['objective']:.0f}元  "
              f"预测计划={r_pred['objective']:.0f}元  "
              f"实际执行={r_real['objective']:.0f}元  "
              f"(预测误差真实代价 +{cost_gap_real:.0f}元)")

        all_comparisons[day_label] = {
            "date": day_data["date"],
            "nRMSE": day_data["nRMSE"],
            "desc": day_data["desc"],
            "pv_perfect_24h": pv_perfect_24h,
            "pv_predicted_24h": pv_predicted_24h,
            "result_perfect": r_perfect,
            "result_predicted": r_pred,
            "result_real": r_real,
            "cost_gap": cost_gap_real,
            "absorption_gap": r_real['metrics']['绿电消纳率(%)'] - r_perfect['metrics']['绿电消纳率(%)'],
        }

    return all_comparisons, base_load


# ══════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════
def plot_forecast_vs_perfect(all_comparisons, base_load, save_dir):
    """绘制 PV 预测 vs 实际 + MILP 调度结果对比"""

    # ── 图1: 预测 vs 实际 PV 曲线 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    day_labels = ["最佳预测", "典型预测", "最差预测"]

    for ax, label in zip(axes, day_labels):
        d = all_comparisons[label]
        t = np.arange(24)

        ax.plot(t, d["pv_perfect_24h"], 'ko-', linewidth=2, markersize=4, label='实际PV')
        ax.plot(t, d["pv_predicted_24h"], 's--', color='#D62728', linewidth=2,
                markersize=4, label='GRU-XGBoost预测')
        ax.fill_between(t, d["pv_perfect_24h"], d["pv_predicted_24h"],
                         alpha=0.2, color='#D62728')

        ax.fill_between(t, 0, d["pv_perfect_24h"], alpha=0.1, color='#FDB813')

        date_str = d["date"].strftime("%Y-%m-%d")
        set_chinese_label(ax, xlabel="时间 (h)", ylabel="光伏出力 (MW)",
                          title=f"{label} ({date_str})\nnRMSE={d['nRMSE']:.4f}")
        set_time_xticks(ax)
        ax.legend(prop={"family": "SimHei", "size": 9}, frameon=False, loc="upper left")
        clean_axes(ax)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "forecast_vs_actual_3days.png"))

    # ── 图2: 预测误差对系统成本的影响（三柱对比）──
    fig, ax = plt.subplots(figsize=(12, 5.5))

    costs_perfect = [all_comparisons[l]["result_perfect"]["objective"] for l in day_labels]
    costs_predicted = [all_comparisons[l]["result_predicted"]["objective"] for l in day_labels]
    costs_real = [all_comparisons[l]["result_real"]["objective"] for l in day_labels]
    gaps_real = [all_comparisons[l]["cost_gap"] for l in day_labels]
    nrmses = [all_comparisons[l]["nRMSE"] for l in day_labels]

    x = np.arange(len(day_labels))
    width = 0.25

    bars1 = ax.bar(x - width, costs_perfect, width, label='完美信息调度 (理论上界)', color='#2CA02C', alpha=0.85)
    bars2 = ax.bar(x, costs_predicted, width, label='预测驱动调度 (MILP计划成本)', color='#FF7F0E', alpha=0.85)
    bars3 = ax.bar(x + width, costs_real, width, label='实际执行成本 (预测方案+真实PV)', color='#D62728', alpha=0.85)

    # 标注真实代价（完美→实际执行）
    for i, (cp, cr, gap, nrm) in enumerate(zip(costs_perfect, costs_real, gaps_real, nrmses)):
        if gap > 0:
            ax.annotate(f'预测误差\n真实代价 +{gap:.0f}元',
                        xy=(i + width, cr), xytext=(i + width + 0.15, cr + gap * 0.4),
                        fontsize=8, family='SimHei', ha='center', color='#D62728',
                        arrowprops=dict(arrowstyle='->', color='#D62728', alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{all_comparisons[l]['date'].strftime('%m-%d')}\nnRMSE={all_comparisons[l]['nRMSE']:.3f}" for l in day_labels],
                       family="SimHei", fontsize=9)
    set_chinese_label(ax, ylabel="系统总运行成本 (元/日)",
                      title="预测误差对调度经济性的真实影响")
    ax.legend(prop={"family": "SimHei", "size": 9}, frameon=False, loc="upper left")
    clean_axes(ax)

    plt.tight_layout()
    save_and_show(fig, os.path.join(save_dir, "forecast_error_cost.png"))

    # ── 图3: 最佳和最差预测日的功率平衡堆叠 ──
    from utils.plot_utils import plot_power_balance_stack

    for label in ["最佳预测", "最差预测"]:
        d = all_comparisons[label]
        r = d["result_predicted"]
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
            scenario_label=f"{label} (GRU-XGBoost预测驱动MILP)\n{d['date'].strftime('%Y-%m-%d')} nRMSE={d['nRMSE']:.4f}",
            save_path=os.path.join(save_dir, f"power_balance_{label}.png"),
        )


# ══════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("outputs/forecast_milp_bridge", exist_ok=True)
    save_dir = "outputs/forecast_milp_bridge"

    # ── 阶段0: 获取预测 ──
    y_actual, y_hybrid, test_dates, daily_errors, params = get_test_predictions(device)

    # ── 阶段1: 选取代表性日期 ──
    demo_days = pick_demo_days(y_actual, y_hybrid, test_dates, daily_errors)

    # ── 阶段2: MILP 对比 ──
    all_comparisons, base_load = run_scheduling_comparison(demo_days, params)

    # ── 阶段3: 可视化 ──
    print("\n" + "=" * 60)
    print("【阶段2：生成对比图表】")
    print("=" * 60)
    plot_forecast_vs_perfect(all_comparisons, base_load, save_dir)

    # ── 阶段4: 保存指标 ──
    rows = []
    for label, d in all_comparisons.items():
        rp = d["result_perfect"]["metrics"]
        rpr = d["result_predicted"]["metrics"]
        rr = d["result_real"]["metrics"]
        rows.append({
            "日期": d["date"].strftime("%Y-%m-%d"),
            "预测nRMSE": d["nRMSE"],
            "预测类型": label,
            "完美_成本(元)": round(d["result_perfect"]["objective"], 2),
            "MILP计划成本(元)": round(d["result_predicted"]["objective"], 2),
            "实际执行成本(元)": round(d["result_real"]["objective"], 2),
            "预测误差真实代价(元)": round(d["result_real"]["objective"] - d["result_perfect"]["objective"], 2),
            "MILP低估金额(元)": round(d["result_real"]["objective"] - d["result_predicted"]["objective"], 2),
            "完美_消纳率(%)": rp["绿电消纳率(%)"],
            "实际_消纳率(%)": rr["绿电消纳率(%)"],
            "完美_碳排(kg)": rp["碳排放量(kgCO2)"],
            "实际_碳排(kg)": rr["碳排放量(kgCO2)"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "metrics_comparison.csv"), index=False, encoding="utf-8-sig")

    # ── 总结 ──
    print(f"\n{'=' * 60}")
    print(f"预测→调度 闭环验证完成")
    print(f"{'=' * 60}")
    print(f"\n{'日期':<12} {'nRMSE':>8} {'完美成本':>10} {'MILP计划':>10} {'实际执行':>10} {'真实代价':>10}")
    print("-" * 60)
    for label in ["最佳预测", "典型预测", "最差预测"]:
        d = all_comparisons[label]
        real_gap = d['result_real']['objective'] - d['result_perfect']['objective']
        print(f"{d['date'].strftime('%Y-%m-%d'):<12} {d['nRMSE']:>8.4f} "
              f"{d['result_perfect']['objective']:>10.2f} "
              f"{d['result_predicted']['objective']:>10.2f} "
              f"{d['result_real']['objective']:>10.2f} "
              f"{real_gap:>+10.2f}")

    avg_cost_gap = np.mean([all_comparisons[l]["cost_gap"] for l in ["最佳预测", "典型预测", "最差预测"]])
    print(f"\n平均预测误差代价: +{avg_cost_gap:.2f} 元/日")

    print(f"\n[OK] 全部成果已保存至: {save_dir}/")
    print(f"  [OK] forecast_vs_actual_3days.png")
    print(f"  [OK] forecast_error_cost.png")
    print(f"  [OK] power_balance_最佳预测.png")
    print(f"  [OK] power_balance_最差预测.png")
    print(f"  [OK] metrics_comparison.csv")


if __name__ == "__main__":
    main()
