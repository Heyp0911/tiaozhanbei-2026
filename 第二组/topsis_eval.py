"""
topsis_eval.py — 模块三：TOPSIS关键技术综合评价

功能：
  1. 从params.json读取5×8评价矩阵
  2. 执行TOPSIS算法（正向化→标准化→加权→理想解→贴近度）
  3. 输出排序柱状图

使用方法：
  python topsis_eval.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_params
from utils.plot_utils import plot_topsis_ranking


def topsis(matrix, weights, directions):
    """
    TOPSIS 综合评价算法

    参数
    ----------
    matrix : np.ndarray shape (m, n)
        原始评价矩阵，m个方案，n个指标
    weights : list of float, length n
        各指标权重
    directions : list of int, length n
        1 = 正向指标（越大越好），-1 = 负向指标（越小越好）

    返回
    ----------
    C : np.ndarray shape (m,)
        相对贴近度，越接近1越好
    rank : np.ndarray shape (m,)
        排序（1为最优）
    """
    m, n = matrix.shape

    # 步骤1：正向化 — 负向指标用差值法（保持线性尺度）
    X = matrix.copy().astype(np.float64)
    for j in range(n):
        if directions[j] == -1:
            # 标准做法: max(x) - x，保持线性关系
            X[:, j] = np.max(X[:, j]) - X[:, j]

    # 步骤2：向量归一化
    norm = np.sqrt(np.sum(X ** 2, axis=0))
    R = X / (norm + 1e-10)

    # 步骤3：加权
    W = np.array(weights)
    V = R * W

    # 步骤4：正负理想解
    A_plus = np.max(V, axis=0)
    A_minus = np.min(V, axis=0)

    # 步骤5：距离
    D_plus = np.sqrt(np.sum((V - A_plus) ** 2, axis=1))
    D_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=1))

    # 步骤6：贴近度
    C = D_minus / (D_plus + D_minus + 1e-10)

    # 步骤7：排序
    rank = np.argsort(-C) + 1  # 降序，1为最优

    return C, rank


def main():
    print("=" * 60)
    print("模块三：TOPSIS 关键技术综合评价")
    print("=" * 60)

    params = load_params()

    t = params["topsis"]
    tech_names = list(t["matrix"].keys())
    matrix_raw = np.array(list(t["matrix"].values()))
    weights = t["weights"]
    directions = t["indicator_directions"]
    indicators = t["indicators"]

    print(f"\n评价对象: {len(tech_names)}项关键技术")
    print(f"评价指标: {len(indicators)}个")
    print(f"权重: 等权 (各{weights[0]:.3f})")

    # 执行TOPSIS
    C, rank = topsis(matrix_raw, weights, directions)

    # 排序输出
    sort_idx = np.argsort(-C)
    print("\n===== TOPSIS 综合评价结果 =====")
    print(f"{'排名':<5} {'技术名称':<22} {'贴近度C_i':<10} {'结论'}")
    print("-" * 55)

    for r, idx in enumerate(sort_idx):
        rank_num = r + 1
        name = tech_names[idx]
        ci = C[idx]
        if rank_num == 1:
            conclusion = "核心支撑技术（最高优先级）"
        elif rank_num <= 3:
            conclusion = "重要支撑技术"
        else:
            conclusion = "辅助支撑技术"
        print(f"{rank_num:<5} {name:<22} {ci:.4f}      {conclusion}")

    # 保存CSV
    result_df = pd.DataFrame({
        "技术名称": tech_names,
        "贴近度C_i": C.round(4),
        "排名": rank,
    })
    result_df = result_df.sort_values("排名")
    result_df.to_csv("outputs/topsis_result.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] TOPSIS结果已保存: outputs/topsis_result.csv")

    # 绘图
    plot_topsis_ranking(tech_names, C, "outputs/topsis_ranking.png")

    print("\n模块三完成！")
    print("  [OK] outputs/topsis_result.csv")
    print("  [OK] outputs/topsis_ranking.png")


if __name__ == "__main__":
    main()
