import os
import sys
import math
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# =============================
# 配置中文字体显示（在Windows上尽量使用系统中文字体）
# =============================
def _setup_chinese_font():
    # 优先尝试常见中文字体
    preferred_fonts = ["Microsoft YaHei", "SimHei", "MS Gothic"]
    for f in preferred_fonts:
        try:
            plt.rcParams["font.sans-serif"] = [f]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    # 兜底使用 matplotlib 默认字体
    plt.rcParams["axes.unicode_minus"] = False


# =============================
# 日志记录函数
# =============================
def setup_logger(log_path: str) -> logging.Logger:
    """
    创建并返回日志记录器，输出到文件与控制台
    """
    logger = logging.getLogger("excel_heatmap")
    logger.setLevel(logging.INFO)
    # 避免重复添加Handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台日志
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# =============================
# 数据验证函数
# =============================
def validate_beam_id(beam: int) -> bool:
    """
    验证波束号是否在有效范围 0~63
    """
    return isinstance(beam, (int, np.integer)) and 0 <= int(beam) <= 63


def read_main_data(
    input_path: str,
    ue_col: str = "UE_Beam[5:0]十进制",
    bs_col: str = "BS_Beam[5:0]十进制",
    rss_col: str = "RSS十进制",
    sheet_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    读取主数据文件并做基本预处理（筛选所需列、去掉无效行）
    """
    if logger:
        logger.info(f"开始读取主数据文件: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"主数据文件不存在: {input_path}")

    try:
        df_obj = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"读取主数据Excel失败: {e}")

    # 兼容 sheet_name=None 返回 dict 的情况：选择包含所需列的工作表
    if isinstance(df_obj, dict):
        needed_cols = [ue_col, bs_col, rss_col]
        chosen_df = None
        chosen_sheet = None
        if sheet_name is not None and sheet_name in df_obj:
            chosen_sheet = sheet_name
            chosen_df = df_obj[sheet_name]
        else:
            for name, _df in df_obj.items():
                try:
                    if all(col in _df.columns for col in needed_cols):
                        chosen_sheet = name
                        chosen_df = _df
                        break
                except Exception:
                    continue
            if chosen_df is None:
                # 回退到第一个工作表（可能导致列校验失败，但给出更明确错误）
                chosen_sheet = next(iter(df_obj.keys()))
                chosen_df = df_obj[chosen_sheet]
        if logger:
            logger.info(f"使用主数据工作表: {chosen_sheet}")
        df = chosen_df
    else:
        df = df_obj
        if logger and sheet_name is not None:
            logger.info(f"使用主数据工作表: {sheet_name}")

    for col in [ue_col, bs_col, rss_col]:
        if col not in df.columns:
            raise KeyError(f"主数据缺少必要列: {col}")

    # 只保留相关列并重命名为简洁英文
    df = df[[ue_col, bs_col, rss_col]].rename(
        columns={ue_col: "ue_beam", bs_col: "bs_beam", rss_col: "rss"}
    )

    # 清洗数据类型
    def _to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return np.nan

    df["ue_beam"] = df["ue_beam"].apply(_to_int_safe)
    df["bs_beam"] = df["bs_beam"].apply(_to_int_safe)
    # RSSI 默认作为浮点数
    def _to_float_safe(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    df["rss"] = df["rss"].apply(_to_float_safe)

    # 去掉包含NaN的行
    before = len(df)
    df = df.dropna(subset=["ue_beam", "bs_beam", "rss"])
    after_dropna = len(df)

    # 验证波束ID范围
    mask_valid = df["ue_beam"].apply(validate_beam_id) & df["bs_beam"].apply(
        validate_beam_id
    )
    df = df[mask_valid]
    after_valid = len(df)

    if logger:
        logger.info(f"原始行数: {before}, 去NaN后: {after_dropna}, 波束有效后: {after_valid}")

    if df.empty:
        raise ValueError("清洗后主数据为空，请检查数据格式是否正确")

    return df


def read_mapping(
    map_path: str,
    id_col: str = "BeamID",
    angle_col: str = "Angle",
    sheet_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, float]:
    """
    读取映射文件，获取 BeamID(0~63) -> Angle(度) 的映射字典
    """
    if logger:
        logger.info(f"开始读取映射文件: {map_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"映射文件不存在: {map_path}")

    try:
        df_obj = pd.read_excel(map_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"读取映射Excel失败: {e}")

    # 兼容 sheet_name=None 返回 dict 的情况：选择包含映射列的工作表
    if isinstance(df_obj, dict):
        needed_cols = [id_col, angle_col]
        chosen_df = None
        chosen_sheet = None
        if sheet_name is not None and sheet_name in df_obj:
            chosen_sheet = sheet_name
            chosen_df = df_obj[sheet_name]
        else:
            for name, _df in df_obj.items():
                try:
                    if all(col in _df.columns for col in needed_cols):
                        chosen_sheet = name
                        chosen_df = _df
                        break
                except Exception:
                    continue
            if chosen_df is None:
                chosen_sheet = next(iter(df_obj.keys()))
                chosen_df = df_obj[chosen_sheet]
        if logger:
            logger.info(f"使用映射工作表: {chosen_sheet}")
        df_map = chosen_df
    else:
        df_map = df_obj
        if logger and sheet_name is not None:
            logger.info(f"使用映射工作表: {sheet_name}")

    for col in [id_col, angle_col]:
        if col not in df_map.columns:
            raise KeyError(f"映射文件缺少必要列: {col}")

    mapping: Dict[int, float] = {}
    for _, row in df_map.iterrows():
        try:
            bid = int(row[id_col])
            ang = float(row[angle_col])
        except Exception:
            continue
        if 0 <= bid <= 63:
            mapping[bid] = ang

    if logger:
        logger.info(f"映射条目数: {len(mapping)} (期望: 64)")
        missing = [i for i in range(64) if i not in mapping]
        if missing:
            logger.warning(f"映射缺失 BeamID: {missing}")

    if not mapping:
        raise ValueError("映射数据为空或格式错误")

    return mapping


# =============================
# 角度映射函数
# =============================
def map_angles(
    df: pd.DataFrame, mapping: Dict[int, float], logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    将 UE/BS 波束号映射为对应角度 AoA/AoD
    """
    df = df.copy()
    df["AoA"] = df["ue_beam"].map(mapping)  # UE侧角度（横轴）
    df["AoD"] = df["bs_beam"].map(mapping)  # BS侧角度（纵轴）

    before = len(df)
    df = df.dropna(subset=["AoA", "AoD"])
    if logger:
        logger.info(f"角度映射后有效行数: {len(df)} / {before}")
    if df.empty:
        raise ValueError("角度映射后数据为空，可能映射表与数据不一致")
    return df


# =============================
# RSSI计算函数
# =============================
def compute_rssi_matrix(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    计算每个 BS-UE 波束对（AoD, AoA）的平均RSSI，并返回矩阵与坐标刻度
    返回:
        matrix: 2D numpy数组，shape = [len(AoD_list), len(AoA_list)]
        AoD_list: 纵轴角度列表（升序）
        AoA_list: 横轴角度列表（升序）
    """
    # 使用pivot_table聚合平均值
    pivot = pd.pivot_table(
        df, index="AoD", columns="AoA", values="rss", aggfunc=np.mean
    )

    # 构建坐标列表（升序）
    aoa_list = sorted(set(df["AoA"].tolist()))
    aod_list = sorted(set(df["AoD"].tolist()))

    # 重新索引保证矩阵维度完整且有序
    pivot = pivot.reindex(index=aod_list, columns=aoa_list)
    matrix = pivot.to_numpy()

    if logger:
        logger.info(
            f"RSSI矩阵尺寸: {matrix.shape} (AoD: {len(aod_list)} x AoA: {len(aoa_list)})"
        )

    return matrix, aod_list, aoa_list


# =============================
# 可选高斯模糊（不依赖SciPy，适用于小矩阵）
# =============================
def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    构造二维高斯核，大小约为 6*sigma，且为奇数
    """
    if sigma <= 0:
        return np.array([[1.0]], dtype=np.float64)
    size = int(max(3, math.ceil(6 * sigma)))
    if size % 2 == 0:
        size += 1
    center = size // 2
    y, x = np.ogrid[-center : center + 1, -center : center + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


def gaussian_blur_nan_aware(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    对包含NaN的矩阵进行高斯模糊（权重归一化，忽略NaN）
    """
    if sigma <= 0:
        return data
    kernel = gaussian_kernel(sigma)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 掩码：非NaN为1，NaN为0
    mask = np.isfinite(data).astype(np.float64)
    data_filled = np.nan_to_num(data, nan=0.0)

    # 进行卷积（手工实现，适用于64x64等较小尺寸）
    H, W = data.shape
    padded_data = np.pad(data_filled, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    padded_mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    out = np.zeros_like(data_filled, dtype=np.float64)
    norm = np.zeros_like(data_filled, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            region = padded_data[i : i + kh, j : j + kw]
            region_m = padded_mask[i : i + kh, j : j + kw]
            w = kernel * region_m
            s = (region * w).sum()
            n = w.sum()
            out[i, j] = s / n if n > 1e-12 else np.nan
            norm[i, j] = n

    return out


# =============================
# 热力图生成函数
# =============================
def generate_heatmap(
    matrix: np.ndarray,
    aod_list: List[float],
    aoa_list: List[float],
    output_path: str,
    title: str,
    colormap: str = "viridis",
    use_log: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    blur_sigma: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    生成并保存热力图PNG
    - 横轴: AoA（UE侧）
    - 纵轴: AoD（BS侧）
    - 颜色: RSSI强度（支持对数颜色标度）
    """
    _setup_chinese_font()

    # 可选模糊
    mat = matrix.copy()
    if blur_sigma and blur_sigma > 0:
        if logger:
            logger.info(f"对矩阵进行高斯模糊，sigma={blur_sigma}")
        mat = gaussian_blur_nan_aware(mat, blur_sigma)

    # 处理NaN为透明遮罩
    finite_mask = np.isfinite(mat)
    finite_values = mat[finite_mask]
    if finite_values.size == 0:
        raise ValueError("矩阵数据全为NaN，无法生成热力图")

    # 对数颜色标度需要正数，RSSI通常为负值，做平移变换：value' = value - min + 1e-6
    if use_log:
        min_val = np.nanmin(finite_values)
        shifted = mat - min_val + 1e-6
        # vmin/vmax 若提供则按同样平移
        vmin_shift = None if vmin is None else (vmin - min_val + 1e-6)
        vmax_shift = None if vmax is None else (vmax - min_val + 1e-6)
        norm = LogNorm(
            vmin=vmin_shift if vmin_shift is not None else np.nanmin(shifted[finite_mask]),
            vmax=vmax_shift if vmax_shift is not None else np.nanmax(shifted[finite_mask]),
        )
        plot_data = shifted
    else:
        # 线性标度
        norm = None
        plot_data = mat

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # 构建坐标范围（边界基于角度列表的最小/最大）
    # 我们使用 pcolormesh 以获得更细的控制
    # 计算网格坐标（在列表中相邻角度之间使用中点扩展边界）
    def _edges(vals: List[float]) -> np.ndarray:
        vals = np.array(vals, dtype=np.float64)
        if len(vals) == 1:
            step = 1.0
            return np.array([vals[0] - step / 2, vals[0] + step / 2], dtype=np.float64)
        steps = np.diff(vals)
        edges = np.empty(len(vals) + 1, dtype=np.float64)
        edges[1:-1] = (vals[:-1] + vals[1:]) / 2.0
        edges[0] = vals[0] - steps[0] / 2.0
        edges[-1] = vals[-1] + steps[-1] / 2.0
        return edges

    x_edges = _edges(aoa_list)
    y_edges = _edges(aod_list)

    # 创建透明遮罩的颜色映射
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(color=(1, 1, 1, 0))  # NaN透明

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(plot_data),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSSI强度" + ("（对数刻度）" if use_log else "（线性刻度）"))

    ax.set_title(title)
    ax.set_xlabel("AoA（UE侧，度）")
    ax.set_ylabel("AoD（BS侧，度）")

    ax.grid(True, linestyle="--", alpha=0.2)

    # 保存图片
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    if logger:
        logger.info(f"热力图已保存: {output_path}")


# =============================
# 主处理函数
# =============================
def process_excel_to_heatmap(
    input_path: str,
    map_path: str,
    output_path: Optional[str] = None,
    sheet_name_main: Optional[str] = None,
    sheet_name_map: Optional[str] = None,
    colormap: str = "viridis",
    use_log: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    blur_sigma: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    整体流程：读取 -> 验证 -> 映射 -> 计算 -> 绘图 -> 保存
    返回生成的图片路径
    """
    # 日志准备
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_default = os.path.join(
        os.path.dirname(output_path) if output_path else os.path.dirname(input_path),
        f"excel_heatmap_{ts}.log",
    )
    lg = logger or setup_logger(log_default)

    try:
        lg.info("====== 开始处理Excel数据并生成热力图 ======")
        lg.info(f"主数据文件: {input_path}")
        lg.info(f"映射文件: {map_path}")
        # 读取与清洗主数据
        df_main = read_main_data(
            input_path=input_path, sheet_name=sheet_name_main, logger=lg
        )
        # 读取映射
        mapping = read_mapping(
            map_path=map_path, sheet_name=sheet_name_map, logger=lg
        )
        # 映射角度
        df_mapped = map_angles(df_main, mapping, logger=lg)
        # 计算矩阵
        matrix, aod_list, aoa_list = compute_rssi_matrix(df_mapped, logger=lg)

        # 输出路径处理
        if output_path is None or output_path.strip() == "":
            in_base = os.path.splitext(os.path.basename(input_path))[0]
            out_dir = os.path.join(os.path.dirname(input_path), "heatmap_outputs")
            output_path = os.path.join(out_dir, f"{in_base}_heatmap.png")

        # 图标题
        title = f"BS-UE 波束对平均RSSI热力图 ({os.path.basename(input_path)})"

        # 生成热力图
        generate_heatmap(
            matrix=matrix,
            aod_list=aod_list,
            aoa_list=aoa_list,
            output_path=output_path,
            title=title,
            colormap=colormap,
            use_log=use_log,
            vmin=vmin,
            vmax=vmax,
            blur_sigma=blur_sigma,
            logger=lg,
        )

        lg.info("====== 处理完成 ======")
        return output_path

    except Exception as e:
        lg.error(f"处理失败: {e}", exc_info=True)
        raise


# =============================
# 命令行入口
# =============================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取Excel数据并生成BS-UE平均RSSI热力图（支持中文显示、日志、可选模糊）"
    )
    # 默认路径取用户提供的需求
    parser.add_argument(
        "--input",
        type=str,
        default=r"D:\Code\SLAMPro\UE20\debugDoc\Serial Debug 2026-01-20 200425.xlsx",
        help="主数据Excel路径",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=r"D:\桌面\SLAMM\beam_angle.xlsx",
        help="BeamID->Angle映射Excel路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出PNG路径（默认与主数据同目录的heatmap_outputs下）",
    )
    parser.add_argument("--sheet-main", type=str, default=None, help="主数据工作表名")
    parser.add_argument("--sheet-map", type=str, default=None, help="映射工作表名")
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="热力图风格（如: viridis, plasma, inferno, magma, jet 等）",
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        default=True,
        help="使用对数颜色标度（默认关闭，启用需指定该参数）",
    )
    parser.add_argument(
        "--vmin", type=float, default=None, help="颜色刻度最小RSSI（可选，默认自动）"
    )
    parser.add_argument(
        "--vmax", type=float, default=None, help="颜色刻度最大RSSI（可选，默认自动）"
    )
    parser.add_argument(
        "--blur-sigma", type=float, default=1.0, help="高斯模糊强度（0为不模糊）"
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 构建日志文件名
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_for_log = (
        os.path.dirname(args.output)
        if args.output
        else os.path.dirname(args.input) if os.path.isfile(args.input) else os.getcwd()
    )
    log_path = os.path.join(out_dir_for_log, f"excel_heatmap_{ts}.log")
    logger = setup_logger(log_path)

    # 提示运行参数
    logger.info(
        f"参数: input='{args.input}', mapping='{args.mapping}', output='{args.output}', "
        f"sheet_main='{args.sheet_main}', sheet_map='{args.sheet_map}', "
        f"colormap='{args.colormap}', logscale={args.logscale}, "
        f"vmin={args.vmin}, vmax={args.vmax}, blur_sigma={args.blur_sigma}"
    )

    try:
        out_png = process_excel_to_heatmap(
            input_path=args.input,
            map_path=args.mapping,
            output_path=args.output if args.output else None,
            sheet_name_main=args.sheet_main,
            sheet_name_map=args.sheet_map,
            colormap=args.colormap,
            use_log=args.logscale,
            vmin=args.vmin,
            vmax=args.vmax,
            blur_sigma=args.blur_sigma,
            logger=logger,
        )
        logger.info(f"输出PNG: {out_png}")
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
