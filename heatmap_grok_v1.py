import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit  # 用于CS

def load_data(beam_file, meas_file):
    """读取数据并映射角度"""
    beam_df = pd.read_excel(beam_file, sheet_name='Sheet1')
    pos_df = pd.read_excel(beam_file, sheet_name='Sheet2')
    meas_df = pd.read_excel(meas_file, sheet_name='Sheet1')
    
    beam_id_to_angle = dict(zip(beam_df['BeamID'], beam_df['Angle']))
    meas_df['AoA'] = meas_df['UE_Beam'].map(beam_id_to_angle)
    meas_df['AoD'] = meas_df['BS_Beam'].map(beam_id_to_angle)
    meas_df['RSS_dB'] = 10 * np.log10(meas_df['RSS'] + 1e-6)
    meas_df = meas_df.drop_duplicates(subset=['AoA', 'AoD'])
    
    # 计算LoS (假设UE4为示例)
    bs_pos = pos_df[pos_df['Node'] == 'BS'][['X', 'Y']].values[0]
    ue_pos = pos_df[pos_df['Node'] == 'UE4'][['X', 'Y']].values[0]  # 可替换其他UE
    los_aod = np.degrees(np.arctan2(ue_pos[1] - bs_pos[1], ue_pos[0] - bs_pos[0]))
    los_aoa = los_aod  # 简化假设
    return meas_df, beam_id_to_angle, los_aoa, los_aod

def generate_rss_matrix(meas_df, resolution=0.1):
    """插值生成连续RSS矩阵"""
    aoa_vals = meas_df['AoA'].unique()
    aod_vals = meas_df['AoD'].unique()
    aoa_grid = np.arange(min(aoa_vals) - 5, max(aoa_vals) + 5, resolution)
    aod_grid = np.arange(min(aod_vals) - 5, max(aod_vals) + 5, resolution)
    AOA, AOD = np.meshgrid(aoa_grid, aod_grid)
    rss_grid = griddata((meas_df['AoA'], meas_df['AoD']), meas_df['RSS_dB'], (AOA, AOD), method='cubic')
    rss_grid -= np.nanmax(rss_grid)  # 归一化
    return AOA, AOD, rss_grid

def cs_sparse_recovery(meas_df, beam_id_to_angle, K=5):
    """压缩感知恢复稀疏角度谱 (简化版，使用OMP)"""
    angles = np.array(list(beam_id_to_angle.values()))
    N = len(angles)
    A = np.exp(-1j * np.pi * np.outer(angles, np.arange(N))) / np.sqrt(N)  # 简化DFT矩阵作为码本
    y = meas_df['RSS_dB'].values.reshape(-1, 1)  # 假设扁平化
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K)
    omp.fit(A, y)
    x_hat = omp.coef_  # 恢复谱
    # 重构热力图 (此处简化，返回谱功率)
    spectrum = np.abs(x_hat.reshape(int(np.sqrt(len(x_hat))), -1))
    return spectrum

def identify_paths(rss_grid, AOA, AOD, los_aoa, los_aod, thresh=-20):
    """路径鉴别: 峰值搜索 + 几何匹配"""
    # 展平找峰
    peaks, _ = find_peaks(rss_grid.flatten(), height=thresh, distance=10)  # 距离避免密集峰
    peak_indices = np.unravel_index(peaks, rss_grid.shape)
    paths = []
    for i in range(len(peaks)):
        aoa = AOA[peak_indices[0][i], 0]
        aod = AOD[0, peak_indices[1][i]]
        power = rss_grid[peak_indices[0][i], peak_indices[1][i]]
        if abs(aoa - los_aoa) < 5 and abs(aod - los_aod) < 5 and power == np.max(rss_grid):  # 判据: 最强 + 接近LoS
            type_ = 'LoS'
        else:
            type_ = 'NLoS'  # 可扩展反射匹配
        paths.append({'AoA': aoa, 'AoD': aod, 'Power_dB': power, 'Type': type_})
    return pd.DataFrame(paths)

def plot_heatmap(AOA, AOD, rss_grid, paths):
    """生成热力图"""
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(AOA, AOD, rss_grid, shading='gouraud', cmap='hot')
    plt.colorbar(label='Normalized RSS (dB)')
    plt.xlabel('AoA (degrees)')
    plt.ylabel('AoD (degrees)')
    plt.title('AoA-AoD Heatmap')
    # 标注路径
    for _, path in paths.iterrows():
        plt.scatter(path['AoA'], path['AoD'], color='blue' if path['Type']=='LoS' else 'green')
        plt.text(path['AoA'], path['AoD'], f"{path['Type']} {path['Power_dB']:.1f}dB")
    plt.savefig('heatmap.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    beam_file = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    meas_file = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-26 164520_filtered.xlsx'
    meas_df, beam_map, los_aoa, los_aod = load_data(beam_file, meas_file)
    AOA, AOD, rss_grid = generate_rss_matrix(meas_df)
    # 可选CS增强: spectrum = cs_sparse_recovery(meas_df, beam_map)  # 替换rss_grid if needed
    paths = identify_paths(rss_grid, AOA, AOD, los_aoa, los_aod)
    plot_heatmap(AOA, AOD, rss_grid, paths)
    print(paths)