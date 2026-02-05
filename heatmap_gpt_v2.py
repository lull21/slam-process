# AoA-AoD 热力图生成与多径参数估计
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter, label, find_objects
from scipy.signal import savgol_filter
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from pathlib import Path

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# -------------------------------
# I/O & Preprocessing
# -------------------------------

def load_beam_map(path):
    """读取波束角度映射表,返回 dict beam_id->angle_deg"""
    df = pd.read_excel(path)
    # 启发式列选择
    beam_col = [c for c in df.columns if 'BEAM' in c.upper()][0]
    angle_col = [c for c in df.columns if 'ANGLE' in c.upper() or 'DEG' in c.upper()][0]
    df = df[[beam_col, angle_col]].dropna()
    df[beam_col] = df[beam_col].astype(int)
    df[angle_col] = df[angle_col].astype(float)
    return dict(zip(df[beam_col], df[angle_col]))


def load_measurements(path, ue_col_hint='UE', bs_col_hint='BS', rss_hint='RSS'):
    """读取测量数据"""
    df = pd.read_excel(path)
    ue_col = [c for c in df.columns if ue_col_hint in c.upper()][0]
    bs_col = [c for c in df.columns if bs_col_hint in c.upper()][0]
    rss_col = [c for c in df.columns if rss_hint in c.upper() or 'POWER' in c.upper()][0]
    df = df[[ue_col, bs_col, rss_col]].copy()
    df[ue_col] = pd.to_numeric(df[ue_col], errors='coerce')
    df[bs_col] = pd.to_numeric(df[bs_col], errors='coerce')
    df[rss_col] = pd.to_numeric(df[rss_col], errors='coerce')
    df.dropna(inplace=True)
    df.columns = ['UE_Beam', 'BS_Beam', 'RSS']
    return df


def map_beams_to_angles(df, beam_map):
    """将波束ID映射为角度"""
    df = df.copy()
    df['AoA_deg'] = df['UE_Beam'].map(lambda x: beam_map.get(int(x), np.nan))
    df['AoD_deg'] = df['BS_Beam'].map(lambda x: beam_map.get(int(x), np.nan))
    df.dropna(subset=['AoA_deg', 'AoD_deg'], inplace=True)
    return df

# -------------------------------
# Interpolation -> initial heatmap
# -------------------------------

def make_heatmap_interpolated(df, aoa_range=None, aod_range=None, grid_res=1.0, smooth=True):
    """
    构建插值后的初始热力图
    返回: AoA_grid, AoD_grid, heat_init
    df: 必须包含 AoA_deg, AoD_deg, RSS
    """
    if aoa_range is None:
        aoa_min, aoa_max = df['AoA_deg'].min(), df['AoA_deg'].max()
    else:
        aoa_min, aoa_max = aoa_range
    if aod_range is None:
        aod_min, aod_max = df['AoD_deg'].min(), df['AoD_deg'].max()
    else:
        aod_min, aod_max = aod_range
    
    aoa_grid = np.arange(aoa_min, aoa_max + grid_res, grid_res)
    aod_grid = np.arange(aod_min, aod_max + grid_res, grid_res)
    AOA, AOD = np.meshgrid(aoa_grid, aod_grid, indexing='xy')
    
    points = df[['AoA_deg', 'AoD_deg']].values
    values = df['RSS'].values
    grid_points = np.vstack([AOA.ravel(), AOD.ravel()]).T
    
    heat_lin = griddata(points, values, grid_points, method='linear', fill_value=np.nan).reshape(AOA.shape)
    heat_near = griddata(points, values, grid_points, method='nearest').reshape(AOA.shape)
    heat = np.where(np.isnan(heat_lin), heat_near, heat_lin)
    
    if smooth:
        for i in range(heat.shape[0]):
            try:
                win = 7 if heat.shape[1] >= 7 else (heat.shape[1] // 2 * 2 + 1)
                if win >= 3:  # Savitzky-Golay需要至少3个点
                    heat[i, :] = savgol_filter(heat[i, :], win, min(2, win-1))
            except Exception:
                pass
    
    return aoa_grid, aod_grid, heat

# -------------------------------
# Peak detection (coarse)
# -------------------------------

def detect_peaks(heat, percentile_thresh=65, neighborhood_size=3):
    """检测热力图中的峰值"""
    local_max = (heat == maximum_filter(heat, size=(neighborhood_size, neighborhood_size))) & \
                (heat > np.nanpercentile(heat, percentile_thresh))
    labeled, n = label(local_max)
    slices = find_objects(labeled)
    peaks = []
    
    if slices is None:
        return peaks
    
    for i, slc in enumerate(slices):
        if slc is None:
            continue
        region = heat[slc]
        local_pos = np.unravel_index(np.argmax(region), region.shape)
        abs_pos = (local_pos[0] + slc[0].start, local_pos[1] + slc[1].start)
        val = heat[abs_pos]
        peaks.append({'label': i + 1, 'idx': abs_pos, 'power': float(val)})
    
    peaks = sorted(peaks, key=lambda x: -x['power'])
    return peaks

# -------------------------------
# Local patch sparse refinement
# -------------------------------

def beam_gain(angle_deg, beam_center_deg, beamwidth_deg=10.0):
    """计算波束增益"""
    sigma = beamwidth_deg / 2.355
    return np.exp(-0.5 * ((angle_deg - beam_center_deg) / sigma) ** 2)


def refine_patches(df_agg, aoa_grid, aod_grid, heat_init, peaks, patch_half=3, beamwidth=10.0, alpha=0.1, max_peaks=20):
    """
    通过局部LASSO精细化峰值
    df_agg: 聚合后的测量数据,包含 AoA_deg AoD_deg RSS (唯一的波束对)
    返回: refined_heat 加性图
    """
    refined = np.zeros_like(heat_init)
    meas_aoa = df_agg['AoA_deg'].values
    meas_aod = df_agg['AoD_deg'].values
    meas_rss = df_agg['RSS'].values
    
    for pk in peaks[:max_peaks]:
        (r0, c0) = pk['idx']
        r1 = max(0, r0 - patch_half)
        r2 = min(heat_init.shape[0] - 1, r0 + patch_half)
        c1 = max(0, c0 - patch_half)
        c2 = min(heat_init.shape[1] - 1, c0 + patch_half)
        
        grid_aod_patch = aod_grid[r1:r2 + 1]
        grid_aoa_patch = aoa_grid[c1:c2 + 1]
        
        G_cols = []
        for aod in grid_aod_patch:
            for aoa in grid_aoa_patch:
                g = beam_gain(meas_aoa, aoa, beamwidth) * beam_gain(meas_aod, aod, beamwidth)
                G_cols.append(g)
        
        G = np.column_stack(G_cols)
        norms = np.linalg.norm(G, axis=0) + 1e-8
        Gn = G / norms
        
        lasso = Lasso(alpha=alpha, max_iter=2000, positive=True)
        lasso.fit(Gn, meas_rss)
        coef = lasso.coef_ / norms
        
        # 放回网格
        k = 0
        for i_r in range(len(grid_aod_patch)):
            for i_c in range(len(grid_aoa_patch)):
                refined[r1 + i_r, c1 + i_c] += coef[k]
                k += 1
    
    return refined

# -------------------------------
# LoS / NLoS simple classification
# -------------------------------

def classify_peaks(peaks_sorted, ratio_thresh=1.5):
    """对峰值进行LoS/NLoS分类"""
    out = []
    if not peaks_sorted:
        return out
    
    top = peaks_sorted[0]
    second = peaks_sorted[1]['power'] if len(peaks_sorted) > 1 else -np.inf
    
    if top['power'] > ratio_thresh * second:
        out.append({**top, 'type': 'Likely LoS'})
        for p in peaks_sorted[1:6]:
            out.append({**p, 'type': 'Likely NLoS'})
    else:
        for i, p in enumerate(peaks_sorted[:6]):
            out.append({**p, 'type': 'Candidate LoS' if i == 0 else 'Candidate NLoS'})
    
    return out

# -------------------------------
# Visualization
# -------------------------------

def plot_heatmap(aoa_grid, aod_grid, heat, classification, save_path=None):
    """
    绘制热力图并标注峰值
    aoa_grid: AoA网格
    aod_grid: AoD网格
    heat: 热力图数据
    classification: 峰值分类结果
    save_path: 保存路径
    """
    plt.figure(figsize=(12, 9))
    
    # 绘制热力图
    im = plt.imshow(heat, extent=[aoa_grid.min(), aoa_grid.max(), 
                                   aod_grid.min(), aod_grid.max()],
                    origin='lower', aspect='auto', cmap='hot', interpolation='bilinear')
    
    plt.colorbar(im, label='RSS (dBm)')
    plt.xlabel('AoA (deg)', fontsize=12)
    plt.ylabel('AoD (deg)', fontsize=12)
    plt.title('AoA-AoD Heatmap with Multipath Components', fontsize=14, fontweight='bold')
    
    # 标注峰值
    colors = {'Likely LoS': 'lime', 'Likely NLoS': 'cyan', 
              'Candidate LoS': 'yellow', 'Candidate NLoS': 'orange'}
    
    for peak in classification:
        idx = peak['idx']
        aoa_val = aoa_grid[idx[1]]
        aod_val = aod_grid[idx[0]]
        ptype = peak['type']
        color = colors.get(ptype, 'white')
        
        plt.plot(aoa_val, aod_val, 'o', color=color, markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5)
        plt.text(aoa_val, aod_val + 2, f"{ptype}\n{peak['power']:.1f}dBm", 
                color='white', fontsize=9, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")
    
    plt.show()

# -------------------------------
# Extract timestamp from filename
# -------------------------------

def extract_timestamp_from_filename(filename):
    """
    从文件名中提取时间戳
    例如: 'Serial Debug 2026-01-27 113221_filtered.xlsx' -> '2026-01-27 113221'
    """
    # 移除扩展名
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 查找时间戳模式: YYYY-MM-DD HHMMSS
    import re
    pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{6})'
    match = re.search(pattern, name_without_ext)
    
    if match:
        return match.group(1)
    else:
        # 如果找不到,返回默认时间戳
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H%M%S')

# -------------------------------
# Main execution
# -------------------------------

if __name__ == '__main__':
    # 指定文件路径
    angle_file = r'D:\桌面\SLAMPro\beam_angle.xlsx'
    rss_file = r'D:\桌面\SLAMPro\debugDoc\Serial Debug 2026-01-27 113221_filtered.xlsx'
    output_dir = r'D:\桌面\SLAMPro\pic_gpt_v2'
    
    # 从文件名提取时间戳
    timestamp = extract_timestamp_from_filename(rss_file)
    output_filename = f"{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"开始处理...")
    print(f"角度映射文件: {angle_file}")
    print(f"测量数据文件: {rss_file}")
    print(f"输出图片路径: {output_path}")
    
    # 加载数据
    print("\n1. 加载波束角度映射...")
    beam_map = load_beam_map(angle_file)
    print(f"   已加载 {len(beam_map)} 个波束映射")
    
    print("\n2. 加载测量数据...")
    meas = load_measurements(rss_file)
    print(f"   已加载 {len(meas)} 条测量记录")
    
    # 映射到角度域
    print("\n3. 映射波束到角度域...")
    meas2 = map_beams_to_angles(meas, beam_map)
    print(f"   成功映射 {len(meas2)} 条记录")
    
    # 聚合重复测量
    print("\n4. 聚合重复测量...")
    meas_agg = meas2.groupby(['UE_Beam', 'BS_Beam', 'AoA_deg', 'AoD_deg']).RSS.mean().reset_index()
    print(f"   聚合后剩余 {len(meas_agg)} 条唯一波束对")
    
    # 构建初始热力图
    print("\n5. 构建初始热力图...")
    aoa_grid, aod_grid, heat_init = make_heatmap_interpolated(meas_agg, grid_res=1.0)
    print(f"   热力图尺寸: {heat_init.shape}")
    
    # 检测峰值
    print("\n6. 检测峰值...")
    peaks = detect_peaks(heat_init)
    print(f"   检测到 {len(peaks)} 个峰值")
    
    # 精细化峰值
    print("\n7. 精细化峰值 (LASSO重构)...")
    refined = refine_patches(meas_agg, aoa_grid, aod_grid, heat_init, peaks)
    
    # 合成最终热力图
    print("\n8. 合成最终热力图...")
    heat_final = 0.6 * refined + 0.4 * heat_init
    
    # 重新检测峰值
    print("\n9. 最终峰值检测...")
    final_peaks = detect_peaks(heat_final)
    print(f"   最终检测到 {len(final_peaks)} 个峰值")
    
    # 分类峰值
    print("\n10. 峰值分类...")
    classification = classify_peaks(final_peaks)
    print(f"   已分类 {len(classification)} 个峰值:")
    for i, peak in enumerate(classification, 1):
        print(f"      峰值{i}: {peak['type']}, 功率={peak['power']:.2f} dBm")
    
    # 绘制并保存热力图
    print("\n11. 绘制热力图...")
    plot_heatmap(aoa_grid, aod_grid, heat_final, classification, save_path=output_path)
    
    print("\n处理完成!")