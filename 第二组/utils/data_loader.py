"""
data_loader.py — 统一数据读取接口

负责读取项目中所有输入数据文件，返回标准化的DataFrame或ndarray。
所有路径从params.json读取，代码中零硬编码。
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path


def ensure_file_exists(filepath, hint=""):
    """验证输入文件存在，缺失时给出友好错误提示"""
    if not os.path.exists(filepath):
        hint_msg = f"\n提示: {hint}" if hint else ""
        raise FileNotFoundError(
            f"缺少输入文件: {filepath}\n"
            f"请确保已运行前置模块生成该文件，或将数据文件放置到正确路径。{hint_msg}"
        )


def load_params(params_path="params.json"):
    """加载全局参数文件"""
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nasa_power(params):
    """
    读取NASA POWER原始气象数据Excel

    参数
    ----------
    params : dict
        全局参数字典

    返回
    ----------
    df : pd.DataFrame
        列: YEAR, MO, DY, HR, GHI, DNI, DHI, T2M
        共8784行逐时数据
    """
    filepath = params["data_paths"]["nasa_power_excel"]

    # 先扫描找到表头行
    df_raw = pd.read_excel(filepath, header=None)

    header_row = -1
    for i in range(min(30, len(df_raw))):
        row_vals = df_raw.iloc[i].values
        if any("ALLSKY_SFC_SW_DWN" in str(v) for v in row_vals):
            header_row = i
            break

    if header_row == -1:
        raise ValueError("无法找到包含 'ALLSKY_SFC_SW_DWN' 的表头行，请检查Excel文件格式。")

    # 重新读取，指定表头行
    df = pd.read_excel(filepath, header=header_row)
    df.columns = df.columns.str.strip()

    # 统一列名
    col_map = {
        "ALLSKY_SFC_SW_DWN": "GHI",
        "ALLSKY_SFC_SW_DNI": "DNI",
        "ALLSKY_SFC_SW_DIFF": "DHI",
        "T2M": "T2M",
    }
    df = df.rename(columns=col_map)

    # 选择需要的列
    df_out = df[["YEAR", "MO", "DY", "HR", "GHI", "DNI", "DHI", "T2M"]].copy()

    # 转为数值
    for col in ["GHI", "DNI", "DHI", "T2M"]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0)

    # 构建datetime索引
    df_out["datetime"] = pd.to_datetime(
        df_out["YEAR"].astype(int).astype(str) + "-"
        + df_out["MO"].astype(int).astype(str).str.zfill(2) + "-"
        + df_out["DY"].astype(int).astype(str).str.zfill(2) + " "
        + df_out["HR"].astype(int).astype(str).str.zfill(2) + ":00:00"
    )

    print(f"[OK] NASA POWER数据加载完成：{len(df_out)}小时，时间范围 {df_out['datetime'].min()} ~ {df_out['datetime'].max()}")
    print(f"   GHI: {df_out['GHI'].min():.1f} ~ {df_out['GHI'].max():.1f} W/m2, mean={df_out['GHI'].mean():.1f}")
    print(f"   T2M: {df_out['T2M'].min():.1f} ~ {df_out['T2M'].max():.1f} degC, mean={df_out['T2M'].mean():.1f}")

    return df_out


def load_green_power_curves(filepath="data/green_power_curves.csv"):
    """加载模块一输出的典型日绿电曲线"""
    ensure_file_exists(filepath, hint="请先运行 python green_power_predictor.py")
    df = pd.read_csv(filepath, index_col=0)
    print(f"[OK] 绿电曲线加载：{df.shape[1]}条曲线，{df.shape[0]}小时")
    return df


def load_base_load_curves(filepath="data/base_load_curves.csv"):
    """加载基础负荷曲线"""
    ensure_file_exists(filepath, hint="请先运行 python generate_base_load.py")
    df = pd.read_csv(filepath, index_col=0)
    print(f"[OK] 基础负荷加载：{df.shape[1]}条曲线，{df.shape[0]}小时")
    return df


def build_rigid_task_profile(params, hours=24):
    """
    构造刚性任务负荷曲线：24小时均匀分布

    返回
    ----------
    P_rigid : np.ndarray shape (24,)
        每小时刚性任务功率 (MW)
    """
    E_rigid = params["ai_tasks"]["E_rigid_daily_MWh"]
    P_rigid_per_hour = E_rigid / hours  # 均匀分配
    return np.full(hours, P_rigid_per_hour)


def build_elastic_task_bounds(params, hours=24):
    """
    构造弹性任务每小时上限

    返回
    ----------
    P_elastic_max : np.ndarray shape (24,)
        弹性任务每小时功率上限 (MW)
    """
    E_elastic = params["ai_tasks"]["E_elastic_daily_MWh"]
    max_ratio = params["ai_tasks"]["hourly_max_ratio__elastic"]
    return np.full(hours, E_elastic * max_ratio)


def build_cold_task_bounds(params, hours=24):
    """
    构造温冷任务每小时上限

    返回
    ----------
    P_cold_max : np.ndarray shape (24,)
        温冷任务每小时功率上限 (MW)
    """
    E_cold = params["ai_tasks"]["E_cold_daily_MWh"]
    max_ratio = params["ai_tasks"]["hourly_max_ratio__cold"]
    return np.full(hours, E_cold * max_ratio)


def build_price_profile(params, hours=24):
    """
    根据分时电价参数构造24小时电价数组

    支持四级电价：尖峰 > 峰 > 平 > 谷
    尖峰时段在7-9月覆盖峰时段中的特定小时。

    返回
    ----------
    price : np.ndarray shape (24,)
        每小时电价 (元/kWh)
    """
    ep = params["electricity_price"]
    price = np.zeros(hours)

    # 默认全部按平段
    for h in range(hours):
        price[h] = ep["flat_price_CNY_per_kWh"]

    # 峰段覆盖
    for h in ep["peak_hours"]:
        price[h] = ep["peak_price_CNY_per_kWh"]

    # 谷段覆盖
    for h in ep["valley_hours"]:
        price[h] = ep["valley_price_CNY_per_kWh"]

    # 尖峰覆盖（仅7-9月，覆盖峰段中的特定小时）
    if "spike_hours" in ep and "spike_price_CNY_per_kWh" in ep:
        for h in ep["spike_hours"]:
            price[h] = ep["spike_price_CNY_per_kWh"]

    return price


def load_minqing_raw_days(filepath="data/minqing_raw_days.csv"):
    """加载闽清验证用的真实日期绿电数据"""
    df = pd.read_csv(filepath, index_col=0)
    print(f"[OK] 闽清原始绿电数据加载：{df.shape[1]}个日期，{df.shape[0]}小时")
    return df


def load_minqing_hydro(filepath="data/minqing_hydro_curve.csv"):
    """加载闽清小水电典型日曲线"""
    df = pd.read_csv(filepath, index_col=0)
    print(f"[OK] 闽清小水电曲线加载：{df.shape[1]}列")
    return df


def load_nasa_power_full(params):
    """
    读取NASA POWER CSV格式数据（含RH2M湿度）

    相比 load_nasa_power() 增加了湿度列，适配光伏预测模型需求。
    CSV格式无需扫描表头，直接解析。

    参数
    ----------
    params : dict
        全局参数字典

    返回
    ----------
    df : pd.DataFrame
        列: YEAR, MO, DY, HR, GHI, DNI, DHI, T2M, RH2M, datetime
    """
    filepath = params["data_paths"].get("nasa_power_csv", "data/nasa_power.csv")

    # 跳过NASA POWER CSV的元数据头部（以 -BEGIN HEADER- 到 -END HEADER-）
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if '-END HEADER-' in line:
            data_start = i + 1
            break

    if data_start == 0:
        raise ValueError("无法找到CSV的数据起始行（-END HEADER- 标记），请检查文件格式。")

    df = pd.read_csv(filepath, skiprows=data_start)

    # 统一列名
    col_map = {
        "ALLSKY_SFC_SW_DWN": "GHI",
        "ALLSKY_SFC_SW_DNI": "DNI",
        "ALLSKY_SFC_SW_DIFF": "DHI",
        "T2M": "T2M",
        "RH2M": "RH2M",
    }
    df = df.rename(columns=col_map)

    # 转为数值
    for col in ["GHI", "DNI", "DHI", "T2M", "RH2M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 构建datetime索引
    df["datetime"] = pd.to_datetime(
        df["YEAR"].astype(int).astype(str) + "-"
        + df["MO"].astype(int).astype(str).str.zfill(2) + "-"
        + df["DY"].astype(int).astype(str).str.zfill(2) + " "
        + df["HR"].astype(int).astype(str).str.zfill(2) + ":00:00"
    )

    # 截取365天（去掉多余的2026年1月1日数据）
    cutoff = df["datetime"] < pd.Timestamp("2026-01-01")
    df = df[cutoff].copy()

    print(f"[OK] NASA POWER全量CSV加载完成：{len(df)}小时，"
          f"{df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"   GHI: {df['GHI'].min():.1f} ~ {df['GHI'].max():.1f} W/m2, mean={df['GHI'].mean():.1f}")
    print(f"   T2M: {df['T2M'].min():.1f} ~ {df['T2M'].max():.1f} °C, mean={df['T2M'].mean():.1f}")
    print(f"   RH2M: {df['RH2M'].min():.1f} ~ {df['RH2M'].max():.1f} %, mean={df['RH2M'].mean():.1f}")

    return df


def get_available_dates(df_nasa):
    """
    返回NASA数据中所有可用的日期列表

    返回
    ----------
    dates : list of str
        ['2025-01-01', '2025-01-02', ...]
    """
    return sorted(df_nasa["datetime"].dt.strftime("%Y-%m-%d").unique())


if __name__ == "__main__":
    # 测试数据加载
    params = load_params()
    df = load_nasa_power(params)
    dates = get_available_dates(df)
    print(f"\n可用日期数: {len(dates)}")
    print(f"日期范围: {dates[0]} ~ {dates[-1]}")

    price = build_price_profile(params)
    print(f"\n分时电价: {price}")

    P_rigid = build_rigid_task_profile(params)
    print(f"刚性任务每小时: {P_rigid[0]:.2f} MW")
