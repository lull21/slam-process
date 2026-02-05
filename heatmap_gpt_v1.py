import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter

import matplotlib.pyplot as plt
import numpy as np

def load_beam_mapping(file_path):
    """
    读取 beam_angle.xlsx，返回 BeamID 到 物理角度(度)的映射字典。
    """
    df = pd.read_excel(file_path)
    return {int(row.BeamID): row.Angle for row in df.itertuples()}

def load_measurements(file_path):
    """
    读取测量文件，保留 UE_Beam, BS_Beam, RSS 列，按 (UE,BS) 组合取平均RSS。
    返回包含列['UE_Beam','BS_Beam','RSS']的DataFrame。
    """
    df = pd.read_excel(file_path)
    df = df[['UE_Beam','BS_Beam','RSS']]
    df = df.groupby(['UE_Beam','BS_Beam'], as_index=False)['RSS'].mean()
    return df

def preprocess_data(meas_df, mapping):
    """
    将波束编号映射到角度，生成AoA、AoD和RSS数组。
    返回 AoA_list, AoD_list, RSS_list。
    """
    meas_df['AoA'] = meas_df['UE_Beam'].map(mapping)
    meas_df['AoD'] = meas_df['BS_Beam'].map(mapping)
    # 删除可能存在的缺失值
    meas_df = meas_df.dropna(subset=['AoA','AoD'])
    AoA = meas_df['AoA'].values
    AoD = meas_df['AoD'].values
    RSS = meas_df['RSS'].values
    return AoA, AoD, RSS

def build_heatmap(AoA, AoD, RSS, resolution=1.0, method='linear'):
    """
    基于给定的离散AoA/AoD/RSS数据，构建连续的热力图矩阵。
    - resolution: 网格分辨率(度), 可设为1.0度。
    返回: (AoD_grid, AoA_grid, RSS_grid) 三元组，其中 RSS_grid 是二维矩阵。
    """
    # 定义网格范围
    AoD_min, AoD_max = AoD.min(), AoD.max()
    AoA_min, AoA_max = AoA.min(), AoA.max()
    AoD_grid = np.arange(AoD_min, AoD_max+resolution, resolution)
    AoA_grid = np.arange(AoA_min, AoA_max+resolution, resolution)
    AoD_mesh, AoA_mesh = np.meshgrid(AoD_grid, AoA_grid)
    # 插值填充矩阵
    RSS_grid = griddata((AoD, AoA), RSS, (AoD_mesh, AoA_mesh), method=method)
    # 对于插值后的NaN，可用0填充或邻近值填补
    RSS_grid = np.nan_to_num(RSS_grid, nan=0.0)
    return AoD_grid, AoA_grid, RSS_grid

def detect_peaks(RSS_grid, AoD_grid, AoA_grid, threshold):
    """
    在RSS热力图中检测局部峰值。使用最大滤波寻找局部最大点，阈值过滤。
    返回峰值列表，每个元素是 (AoD_peak, AoA_peak, RSS_value)。
    """
    # 2D最大滤波找到邻域最大值
    neighborhood = np.ones((3,3))
    local_max = (RSS_grid == maximum_filter(RSS_grid, footprint=neighborhood))
    peaks = np.logical_and(local_max, RSS_grid > threshold)
    peak_indices = np.argwhere(peaks)
    peak_list = []
    for (i,j) in peak_indices:
        aoa = AoA_grid[i]
        aod = AoD_grid[j]
        power = RSS_grid[i, j]
        peak_list.append((aod, aoa, power))
    return peak_list

def plot_heatmap_with_paths(
    AoD_grid, AoA_grid, RSS_mat, peaks,
    max_nlos=3,      # 最多标注几个 NLoS
    power_gap=8.0    # NLoS 与 LoS 的最大能量差 (dB)
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 7))

    # --- 热力图 ---
    im = ax.imshow(
        RSS_mat,
        origin='lower',
        aspect='auto',
        extent=[
            AoD_grid.min(), AoD_grid.max(),
            AoA_grid.min(), AoA_grid.max()
        ]
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RSS (dB)")

    ax.set_xlabel("AoD (deg)")
    ax.set_ylabel("AoA (deg)")
    ax.set_title("AoA–AoD RSS Heatmap with Dominant Paths")

    # --- 按能量排序 ---
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)

    # --- LoS ---
    los_aod, los_aoa, los_rss = peaks[0]

    ax.scatter(
        los_aod, los_aoa,
        s=160, marker='*',
        edgecolors='k',
        label='LoS'
    )

    ax.annotate(
        f"LoS\n({los_aod:.1f}°, {los_aoa:.1f}°)",
        xy=(los_aod, los_aoa),
        xytext=(los_aod+4, los_aoa+4),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10
    )

    # --- NLoS：只选能量高的 ---
    nlos_count = 0
    for (aod, aoa, rss) in peaks[1:]:
        if rss < los_rss - power_gap:
            break  # 后面的更弱，直接退出

        if nlos_count >= max_nlos:
            break

        ax.scatter(
            aod, aoa,
            s=80, marker='o',
            edgecolors='k',
            label='NLoS' if nlos_count == 0 else None
        )

        ax.annotate(
            f"NLoS\n({aod:.1f}°, {aoa:.1f}°)",
            xy=(aod, aoa),
            xytext=(aod+3, aoa-5),
            arrowprops=dict(arrowstyle="->"),
            fontsize=9
        )

        nlos_count += 1

    # --- 对角线（辅助几何理解）---
    min_ang = max(AoD_grid.min(), AoA_grid.min())
    max_ang = min(AoD_grid.max(), AoA_grid.max())
    ax.plot(
        [min_ang, max_ang],
        [min_ang, max_ang],
        linestyle='--',
        linewidth=1
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


# 主流程示例
mapping = load_beam_mapping('D:\\桌面\\SLAMPro\\beam_angle.xlsx')
meas_df = load_measurements('D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-26 170305_filtered.xlsx')
AoA_vals, AoD_vals, RSS_vals = preprocess_data(meas_df, mapping)
AoD_grid, AoA_grid, RSS_mat = build_heatmap(AoA_vals, AoD_vals, RSS_vals, resolution=1.4, method='cubic')
peaks = detect_peaks(RSS_mat, AoD_grid, AoA_grid, threshold=np.percentile(RSS_mat, 90))
plot_heatmap_with_paths(
    AoD_grid=AoD_grid,
    AoA_grid=AoA_grid,
    RSS_mat=RSS_mat,
    peaks=peaks
)
print("检测到峰值路径 (AoD, AoA, RSS)：", peaks)
