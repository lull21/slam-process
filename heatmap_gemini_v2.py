import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.linalg import svd
from sklearn.cluster import DBSCAN
import os
import re

# ==========================================
# 模块1: 数据加载与预处理
# ==========================================

def load_data(measure_file, map_file):
    """
    读取Excel/CSV文件并清洗数据
    输入:
        measure_file: 测量数据路径 (UE_Beam, BS_Beam, RSS, CLK...)
        map_file: 角度映射表路径 (BeamID, Angle)
    输出:
        df_meas: 清洗后的测量DataFrame
        angle_map: BeamID到角度的字典
    """
    # 读取角度映射
    df_map = pd.read_excel(map_file)
    # 修复：正确创建angle_map字典，假设列名为'BeamID'和'Angle'
    angle_map = dict(zip(df_map['BeamID'], df_map['Angle']))
    
    # 读取测量数据
    df_meas = pd.read_excel(measure_file)
    
    # 数据清洗：去重，取平均RSS
    # 假设同一波束对的多次测量是时间序列，取均值平滑噪声
    # 修复：添加正确的列名
    df_meas = df_meas.groupby(['UE_Beam', 'BS_Beam'], as_index=False).agg({
        'RSS': 'mean',
        'CLK值': 'mean' # 用于后续ToA分析
    })
    
    return df_meas, angle_map

def build_heatmap_matrix(df_meas, angle_map):
    """
    构建稀疏矩阵并进行插值重构
    """
    # 提取最大波束索引
    # 修复：添加正确的列名
    max_ue = df_meas['UE_Beam'].max()
    max_bs = df_meas['BS_Beam'].max()
    
    # 1. 构建原始稀疏矩阵
    # 初始化为底噪 (假设为数据中的最小值或-100dBm对应的线性值)
    min_rss = df_meas['RSS'].min()
    raw_matrix = np.full((max_ue + 1, max_bs + 1), min_rss)
    
    for _, row in df_meas.iterrows():
        # 修复：添加正确的列名
        u, b, r = int(row['UE_Beam']), int(row['BS_Beam']), row['RSS']
        raw_matrix[u, b] = r
        
    # 2. 构建插值网格
    # 原始坐标
    ue_beams = np.arange(max_ue + 1)
    bs_beams = np.arange(max_bs + 1)
    
    # 目标物理角度坐标 (从angle_map获取范围)
    # 注意：这里假设UE和BS使用相似的角度定义，或者需要根据实际情况调整UE的角度范围
    bs_angles = np.array([angle_map.get(i, 0) for i in bs_beams])
    # 假设UE波束覆盖范围较窄或与BS类似但点数少，这里做线性映射假设作为示例
    ue_angles_raw = np.linspace(bs_angles.min(), bs_angles.max(), len(ue_beams)) 
    
    # 定义高分辨率网格 (例如 0.5度 分辨率)
    grid_bs = np.linspace(bs_angles.min(), bs_angles.max(), 180)
    grid_ue = np.linspace(ue_angles_raw.min(), ue_angles_raw.max(), 90)
    
    # 使用样条插值进行重构
    # 注意：输入必须是严格递增的，需要先排序
    sort_idx_bs = np.argsort(bs_angles)
    sort_idx_ue = np.argsort(ue_angles_raw)
    
    interpolator = RectBivariateSpline(ue_angles_raw[sort_idx_ue], 
                                       bs_angles[sort_idx_bs], 
                                       raw_matrix[sort_idx_ue, :][:, sort_idx_bs])
    
    heatmap_high_res = interpolator(grid_ue, grid_bs)
    
    # 限制插值结果不低于底噪
    heatmap_high_res[heatmap_high_res < min_rss] = min_rss
    
    return heatmap_high_res, grid_ue, grid_bs

# ==========================================
# 模块2: SVD角度估计算法
# ==========================================

def svd_angle_estimator(heatmap, grid_ue, grid_bs, energy_thresh=0.90):
    """
    基于SVD的AoA/AoD联合估计算法
    输入:
        heatmap: 二维RSS功率矩阵 (线性域)
        grid_ue, grid_bs: 对应的角度轴
        energy_thresh: 能量截断阈值 (0.90 - 0.95)
    输出:
        paths: 估计出的路径列表
    """
    # 1. 奇异值分解
    # 确保矩阵为非负 (RSS功率)
    heatmap_pos = np.maximum(heatmap, 0)
    U, S, Vt = svd(heatmap_pos)
    
    # 2. 确定有效秩 K
    total_energy = np.sum(S**2)
    cum_energy = np.cumsum(S**2) / total_energy
    rank_k = np.searchsorted(cum_energy, energy_thresh) + 1
    
    # 修复：初始化为空列表
    detected_paths = []
    
    # 3. 迭代提取秩-1分量
    # 这种方法利用了SVD分离主波束和旁瓣的特性：旁瓣通常与主瓣在同一奇异向量中
    for k in range(rank_k):
        # 重构第k个分量
        component = S[k] * np.outer(U[:, k], Vt[k, :])
        
        # 寻找该分量的全局峰值
        # SVD的一个奇异分量可能包含正负值（数学上的），但在功率谱中我们关注幅值最大处
        idx_ue, idx_bs = np.unravel_index(np.argmax(np.abs(component)), component.shape)
        
        # 提取角度
        aoa_est = grid_ue[idx_ue]
        aod_est = grid_bs[idx_bs]
        power_est = np.abs(component[idx_ue, idx_bs])
        
        # 简单的旁瓣抑制逻辑：如果该峰值位置已经被之前的更强分量覆盖，则可能是旁瓣残留
        # 此处简化处理，直接记录
        detected_paths.append({
            'id': k,
            'AoA': aoa_est,
            'AoD': aod_est,
            'Power': power_est,
            'SingularValue': S[k]
        })
        
    return detected_paths

# ==========================================
# 模块3: 路径分类与可视化
# ==========================================

def classify_and_plot(heatmap, paths, grid_ue, grid_bs, output_path=None):
    """
    可视化热力图并标注LoS/NLoS路径
    """
    plt.figure(figsize=(12, 9))
    
    # 绘制热力图 (使用dB刻度增强对比度)
    heatmap_db = 10 * np.log10(heatmap + 1e-9) 
    extent = [grid_bs.min(), grid_bs.max(), grid_ue.min(), grid_ue.max()]
    
    plt.imshow(heatmap_db, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(label='RSS (dB)')
    
    # 路径分类逻辑
    # 判据1: 功率最强 (SVD第一分量) 且符合几何直觉的通常是LoS (在无严重遮挡下)
    # 注意：实际中应结合CLK数据，此处代码仅演示逻辑
    
    # 按奇异值能量排序
    # 修复：添加正确的字典键名
    sorted_paths = sorted(paths, key=lambda x: x['SingularValue'], reverse=True)
    
    if sorted_paths:
        # 假设第一主成分为LoS
        los_path = sorted_paths[0]
        # 修复：添加正确的字典键名
        plt.scatter(los_path['AoD'], los_path['AoA'], c='white', marker='*', s=300, 
                    label=f"LoS (AoD:{los_path['AoD']:.1f}, AoA:{los_path['AoA']:.1f})")
        
        # 其余为NLoS
        for p in sorted_paths[1:]:
            # 过滤弱路径
            if p['Power'] > los_path['Power'] * 0.1: # 10dB阈值
                # 修复：添加正确的字典键名
                plt.scatter(p['AoD'], p['AoA'], c='red', marker='x', s=150, 
                            label=f"NLoS (Rank-{p['id']})")
    
    plt.xlabel('Base Station AoD (Degree)')
    plt.ylabel('User Equipment AoA (Degree)')
    plt.title('AoA-AoD RSS Heatmap & Identified Multipath Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {output_path}")
    
    plt.show()

def extract_timestamp_from_filename(filename):
    """
    从文件名中提取时间戳部分
    例如: "Serial Debug 2026-01-26 165358_filtered.xlsx" -> "2026-01-26 165358"
    """
    # 使用正则表达式提取日期和时间部分
    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{6})', filename)
    if match:
        return match.group(1)
    return None

# ==========================================
# 主执行流示例
# ==========================================
if __name__ == '__main__':
    # 输入文件路径
    measure_file = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-26 170305_filtered.xlsx'
    map_file = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    
    # 提取时间戳并构建输出路径
    filename = os.path.basename(measure_file)
    timestamp = extract_timestamp_from_filename(filename)
    
    if timestamp:
        output_path = f'D:\\桌面\\SLAMPro\\pic_v2\\{timestamp}.png'
    else:
        # 如果无法提取时间戳，使用默认名称
        output_path = 'D:\\桌面\\SLAMPro\\pic_v2\\heatmap_output.png'
    
    # 执行处理流程
    df_meas, angle_map = load_data(measure_file, map_file)
    heatmap, g_ue, g_bs = build_heatmap_matrix(df_meas, angle_map)
    paths = svd_angle_estimator(heatmap, g_ue, g_bs)
    classify_and_plot(heatmap, paths, g_ue, g_bs, output_path=output_path)