import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.optimize import nnls
import os
import re
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 模块1：数据加载与预处理（融合版）
# ==========================================
class BeamDataProcessor:
    """
    数据处理类：负责加载Excel/CSV数据，清洗并转换为矩阵格式
    融合了v1和v3的数据处理逻辑
    """
    def __init__(self, angle_file, rss_file, output_dir='./output'):
        """
        初始化处理器
        :param angle_file: 波束角度映射文件路径
        :param rss_file: 波束测量数据文件路径
        :param output_dir: 输出目录路径
        """
        self.angle_file = angle_file
        self.rss_file = rss_file
        self.output_dir = output_dir
        
        # 加载数据
        self.angle_map = self._load_angle_map()
        self.rss_data, self.timestamp = self._load_rss_data()
        
        # 初始化矩阵数据
        self.rss_matrix = None
        self.ue_angles = None
        self.bs_angles = None
        self.ue_ids = None
        self.bs_ids = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def _load_angle_map(self):
        """
        加载波束ID到角度的映射表
        融合v1和v3的加载逻辑，增强鲁棒性
        """
        try:
            if self.angle_file.endswith('.csv'):
                df = pd.read_csv(self.angle_file)
            else:
                df = pd.read_excel(self.angle_file, header=None)
            
            # 检查第一行是否为字符串类型的表头
            if isinstance(df.iloc[0, 0], str):
                df = df.iloc[1:]
            
            # 确保数据是数值型
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            
            # 构建字典映射: BeamID -> Angle
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            print(f"Error loading angle map: {e}")
            return {}

    def _load_rss_data(self):
        """
        加载波束测量数据
        融合v1和v3的加载逻辑，同时提取时间戳
        """
        try:
            if self.rss_file.endswith('.csv'):
                df = pd.read_csv(self.rss_file)
            else:
                df = pd.read_excel(self.rss_file)
            
            # 从文件名提取时间戳
            timestamp = self._extract_timestamp()
            
            return df, timestamp
        except Exception as e:
            print(f"Error loading RSS data: {e}")
            return pd.DataFrame(), "unknown_timestamp"

    def _extract_timestamp(self):
        """
        从RSS文件名中提取时间戳
        支持多种格式：YYYY-MM-DD HHMMSS 或 YYYY-MM-DD_HHMMSS
        """
        filename = os.path.basename(self.rss_file)
        
        # 匹配时间戳模式
        pattern = r'(\d{4}-\d{2}-\d{2}[\s_]\d{6})'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1).replace('_', ' ')
        else:
            print(f"Warning: Could not extract timestamp from filename: {filename}")
            from datetime import datetime
            return datetime.now().strftime('%Y-%m-%d %H%M%S')

    def get_output_path(self, suffix=''):
        """
        生成输出文件路径
        :param suffix: 文件名后缀（可选）
        :return: 完整的输出文件路径
        """
        filename = f"{self.timestamp}{suffix}.png"
        return os.path.join(self.output_dir, filename)

    def pivot_data(self):
        """
        将长格式日志转换为矩阵格式，并处理重复测量
        融合v1和v3的透视逻辑
        """
        # 验证必要的列是否存在
        if 'UE_Beam' not in self.rss_data.columns or 'BS_Beam' not in self.rss_data.columns:
            raise ValueError("数据文件中未找到 'UE_Beam' 或 'BS_Beam' 列")
        
        # 对重复的(UE, BS)组合取平均值
        df_avg = self.rss_data.groupby(['UE_Beam', 'BS_Beam'])['RSS'].mean().reset_index()
        
        # 透视表构建矩阵
        pivot = df_avg.pivot(index='UE_Beam', columns='BS_Beam', values='RSS')
        
        # 填充缺失值为背景噪声底噪
        min_rss = df_avg['RSS'].min()
        pivot = pivot.fillna(min_rss)
        
        # 仅保留在 angle_map 中存在的波束ID
        valid_ue_beams = [b for b in pivot.index if b in self.angle_map]
        valid_bs_beams = [b for b in pivot.columns if b in self.angle_map]
        
        pivot = pivot.loc[valid_ue_beams, valid_bs_beams]
        
        # 保存ID信息
        self.ue_ids = pivot.index.values
        self.bs_ids = pivot.columns.values
        
        # 映射物理角度
        self.ue_angles = np.array([self.angle_map[uid] for uid in self.ue_ids])
        self.bs_angles = np.array([self.angle_map[bid] for bid in self.bs_ids])
        
        # 保存矩阵
        self.rss_matrix = pivot.values
        
        return self.rss_matrix, self.ue_angles, self.bs_angles


# ==========================================
# 模块2：LoS径估计器（完整保留v1算法）
# ==========================================
class LoSEstimator:
    """
    LoS径估计器：使用NN-OMP算法（完整保留v1的核心逻辑）
    """
    def __init__(self, ue_angles, bs_angles, rss_matrix):
        self.ue_angles = ue_angles
        self.bs_angles = bs_angles
        self.rss_vector = rss_matrix.flatten()
        self.shape = rss_matrix.shape
        self.aoa_grid = None
        self.aod_grid = None
        self.Phi_RX = None
        self.Phi_TX = None
    
    def _gaussian_beam(self, x, center, width):
        """
        高斯波束模型（v1原始实现）
        """
        sigma = width / 2.355  # FWHM转换
        return np.exp(-((x - center)**2) / (2 * sigma**2))

    def construct_dictionary(self, grid_res=0.1, beam_width=1.4):
        """
        构建功率域字典（v1原始实现）
        :param grid_res: 搜索网格分辨率（度）
        :param beam_width: 波束宽度（度）
        """
        # 定义搜索空间
        self.aoa_grid = np.arange(np.min(self.ue_angles), np.max(self.ue_angles), grid_res)
        self.aod_grid = np.arange(np.min(self.bs_angles), np.max(self.bs_angles), grid_res)
        
        # 构建接收和发射字典
        self.Phi_RX = self._gaussian_beam(self.ue_angles[:, None], self.aoa_grid[None, :], beam_width)
        self.Phi_TX = self._gaussian_beam(self.bs_angles[:, None], self.aod_grid[None, :], beam_width)
        
        return self.aoa_grid, self.aod_grid

    def estimate_los_path(self, max_paths=3):
        """
        执行非负正交匹配追踪 (NN-OMP)（v1原始实现）
        返回识别的LoS径
        """
        residual = self.rss_vector.copy()
        selected_atoms = []
        
        rss_mat_shape = (len(self.ue_angles), len(self.bs_angles))
        
        for k in range(max_paths):
            # 1. 计算相关性 (2D Correlation)
            res_mat = residual.reshape(rss_mat_shape)
            correlation = self.Phi_RX.T @ res_mat @ self.Phi_TX
            
            # 2. 寻找最大相关性位置
            idx_aoa, idx_aod = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # 记录选中的原子索引
            atom_idx = (idx_aoa, idx_aod)
            if atom_idx in selected_atoms:
                break  # 防止重复选择
            selected_atoms.append(atom_idx)
            
            # 3. 构建当前支撑集子字典
            current_atoms_list = []
            for (i_r, i_t) in selected_atoms:
                atom_vec = np.outer(self.Phi_RX[:, i_r], self.Phi_TX[:, i_t]).flatten()
                current_atoms_list.append(atom_vec)
            
            Dict_Active = np.column_stack(current_atoms_list)
            
            # 4. 求解非负最小二乘 (NNLS)
            coeffs, rnorm = nnls(Dict_Active, self.rss_vector)
            
            # 5. 更新残差
            reconstructed = Dict_Active @ coeffs
            residual = self.rss_vector - reconstructed
        
        # 更新路径列表
        path_params = []
        for idx, coeff in enumerate(coeffs):
            if coeff > 0:
                i_r, i_t = selected_atoms[idx]
                path_params.append({
                    'AoA': self.aoa_grid[i_r],
                    'AoD': self.aod_grid[i_t],
                    'Power': coeff,
                    'Is_LoS': False  # 稍后分类
                })
        
        paths_df = pd.DataFrame(path_params)
        
        # LoS分类逻辑（v1原始实现）：最大功率的径为LoS
        if not paths_df.empty:
            max_p_idx = paths_df['Power'].idxmax()
            paths_df.at[max_p_idx, 'Is_LoS'] = True
        
        return paths_df


# ==========================================
# 模块3：NLoS径估计器（完整保留v3算法）
# ==========================================
class NLoSEstimator:
    """
    NLoS径估计器：使用SM-SIC算法（完整保留v3的核心逻辑）
    """
    def __init__(self, beam_width_deg=1.4):
        self.beam_width = beam_width_deg
        # 高斯波束模型参数
        self.sigma = beam_width_deg / 2.355

    def _gaussian_beam(self, query_angles, center_angles):
        """
        构建高斯波束响应矩阵（v3原始实现）
        """
        Q, C = np.meshgrid(query_angles, center_angles)
        diff = Q - C
        return np.exp(- (diff ** 2) / (2 * self.sigma ** 2))

    def construct_dictionary(self, ue_angles, bs_angles, grid_res=0.1):
        """
        构建超分辨率字典（v3原始实现）
        """
        # 定义搜索网格
        self.aoa_grid = np.arange(np.min(ue_angles), np.max(ue_angles) + grid_res, grid_res)
        self.aod_grid = np.arange(np.min(bs_angles), np.max(bs_angles) + grid_res, grid_res)
        
        # 字典矩阵
        self.Phi_RX = self._gaussian_beam(self.aoa_grid, ue_angles)
        self.Phi_TX = self._gaussian_beam(self.aod_grid, bs_angles)
        
        return self.aoa_grid, self.aod_grid

    def estimate_nlos_paths(self, rss_matrix, los_path, max_paths=3, 
                           proximity_mask_radius=5.0, cross_mask_width=10.0):
        """
        核心算法：SM-SIC（空间掩膜连续干扰消除）（v3原始实现）
        
        参数:
            rss_matrix: 测量得到的RSS矩阵
            los_path: LoS径信息（用于设置掩膜）
            max_paths: 最大提取路径数
            proximity_mask_radius: LoS周围的圆形屏蔽半径(度)
            cross_mask_width: 十字屏蔽带的宽度(度)
        """
        # 计算相关性矩阵
        correlation_matrix = self.Phi_RX.T @ rss_matrix @ self.Phi_TX
        
        # 初始化结果列表
        estimated_paths = []
        
        # 初始化空间掩膜
        spatial_mask = np.ones((len(self.aoa_grid), len(self.aod_grid)))
        
        # 如果存在LoS径，首先应用LoS掩膜
        if los_path is not None:
            los_aoa = los_path['AoA']
            los_aod = los_path['AoD']
            
            print(f"检测到 LoS 位于: AoD={los_aod:.1f}°, AoA={los_aoa:.1f}°")
            print(f"正在应用十字掩膜规避旁瓣干扰 (Proximity={proximity_mask_radius}°, CrossWidth={cross_mask_width}°)...")
            
            # 创建网格坐标矩阵
            AOA_G, AOD_G = np.meshgrid(self.aoa_grid, self.aod_grid, indexing='ij')
            
            # 条件1: 邻近区域 (圆形)
            dist_sq = (AOA_G - los_aoa)**2 + (AOD_G - los_aod)**2
            mask_prox = dist_sq > proximity_mask_radius**2
            
            # 条件2: 十字区域 (带状)
            mask_cross_aod = np.abs(AOD_G - los_aod) > (cross_mask_width / 2)
            mask_cross_aoa = np.abs(AOA_G - los_aoa) > (cross_mask_width / 2)
            
            # 更新全局掩膜
            spatial_mask *= mask_prox
            spatial_mask *= mask_cross_aod
            spatial_mask *= mask_cross_aoa
        
        # 搜索NLoS径
        for k in range(max_paths):
            # 1. 应用空间掩膜
            masked_corr = correlation_matrix * spatial_mask
            
            # 2. 搜索最大值
            idx_flat = np.argmax(masked_corr)
            idx_aoa, idx_aod = np.unravel_index(idx_flat, masked_corr.shape)
            
            peak_val = masked_corr[idx_aoa, idx_aod]
            aoa_est = self.aoa_grid[idx_aoa]
            aod_est = self.aod_grid[idx_aod]
            
            # 停止条件：如果峰值过小，则停止
            if k > 0 and len(estimated_paths) > 0 and peak_val < 0.1 * estimated_paths[0]['metric']:
                print(f"路径 {k+1} 信号过弱，停止搜索。")
                break
            
            # 记录路径
            estimated_paths.append({
                'id': k+1,
                'type': 'NLoS',
                'aoa': aoa_est,
                'aod': aod_est,
                'metric': peak_val
            })
            
            # 更新掩膜（防止重复检测同一点）
            AOA_G, AOD_G = np.meshgrid(self.aoa_grid, self.aod_grid, indexing='ij')
            dist_sq = (AOA_G - aoa_est)**2 + (AOD_G - aod_est)**2
            mask_local = dist_sq > (1.0)**2
            spatial_mask *= mask_local
        
        return pd.DataFrame(estimated_paths)


# ==========================================
# 模块4：融合可视化模块
# ==========================================
def visualize_fusion_results(rss_matrix, ue_angles, bs_angles, los_path_df, nlos_paths_df, 
                            processor, save_plot=True, output_suffix=''):
    """
    融合可视化：同时展示LoS和NLoS径
    
    参数:
        rss_matrix: RSS矩阵
        ue_angles: UE角度数组
        bs_angles: BS角度数组
        los_path_df: LoS径数据框（来自v1）
        nlos_paths_df: NLoS径数据框（来自v3）
        processor: 数据处理器
        save_plot: 是否保存图像
        output_suffix: 输出文件名后缀
    """
    # 1. 生成热力图（使用RBF插值）
    grid_res = 100j
    grid_aod, grid_aoa = np.mgrid[min(bs_angles):max(bs_angles):grid_res, 
                                  min(ue_angles):max(ue_angles):grid_res]
    
    raw_aod, raw_aoa = np.meshgrid(bs_angles, ue_angles)
    
    try:
        rbf = Rbf(raw_aod.ravel(), raw_aoa.ravel(), rss_matrix.ravel(), function='linear')
        z_grid = rbf(grid_aod, grid_aoa)
    except:
        z_grid = rss_matrix
        grid_aod = raw_aod
        grid_aoa = raw_aoa
    
    # 2. 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制等高线热力图
    contour = plt.contourf(grid_aod, grid_aoa, z_grid, levels=100, cmap='viridis')
    plt.colorbar(contour, label='Received Signal Strength (RSS)')
    
    # 3. 绘制LoS径（来自v1）
    if not los_path_df.empty:
        los = los_path_df[los_path_df['Is_LoS'] == True]
        if not los.empty:
            for _, row in los.iterrows():
                # LoS: 红色大圆圈
                plt.scatter(row['AoD'], row['AoA'], s=200, c='red', marker='o', 
                           edgecolors='white', linewidth=2, label='LoS Path (v1)', zorder=10)
                # 标注
                label = f"LoS\n({row['AoD']:.1f}, {row['AoA']:.1f})"
                plt.text(row['AoD'] + 1, row['AoA'] + 1, label, color='white', fontweight='bold')
                # 绘制十字参考线
                plt.axvline(x=row['AoD'], color='red', linestyle='--', alpha=0.4)
                plt.axhline(y=row['AoA'], color='red', linestyle='--', alpha=0.4)
    
    # 4. 绘制NLoS径（来自v3）
    if not nlos_paths_df.empty:
        for _, row in nlos_paths_df.iterrows():
            # NLoS: 白色X
            plt.scatter(row['aod'], row['aoa'], s=150, c='white', marker='x', 
                       linewidth=3, label='NLoS Path (v3)', zorder=10)
            # 标注
            label = f"NLoS\n({row['aod']:.1f}, {row['aoa']:.1f})"
            plt.text(row['aod'] + 1, row['aoa'] + 1, label, color='white', 
                    fontsize=9, fontweight='bold')
    
    # 5. 设置图形属性
    plt.xlabel('Angle of Departure (AoD) [deg]', fontsize=12)
    plt.ylabel('Angle of Arrival (AoA) [deg]', fontsize=12)
    plt.title('mmWave Multipath Heatmap - Fusion: LoS (v1) + NLoS (v3)', fontsize=14)
    
    # 去除重复的图例标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', 
               frameon=True, facecolor='black', framealpha=0.6, labelcolor='white')
    plt.grid(True, alpha=0.3)
    
    # 6. 保存图像
    saved_path = None
    if save_plot:
        output_path = processor.get_output_path(suffix=output_suffix)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f"融合热力图已保存至: {output_path}")
    
    plt.show()
    
    return saved_path


# ==========================================
# 主执行流程
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("毫米波多径识别系统 - 融合版本")
    print("LoS识别：基于v1的NN-OMP算法")
    print("NLoS识别：基于v3的SM-SIC算法")
    print("=" * 60)
    
    # ===== 配置文件路径 =====
    # 请根据实际情况修改以下路径
    ANGLE_FILE = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    RSS_FILE = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115200_filtered.xlsx'
    OUTPUT_DIR = 'D:\\桌面\\SLAMPro\\pic_v4'
    
    # 检查文件是否存在
    if not os.path.exists(ANGLE_FILE):
        print(f"错误: 找不到角度映射文件: {ANGLE_FILE}")
        exit(1)
    if not os.path.exists(RSS_FILE):
        print(f"错误: 找不到RSS数据文件: {RSS_FILE}")
        exit(1)
    
    try:
        # ===== 步骤1：数据加载与预处理 =====
        print("\n[步骤1] 数据加载与预处理...")
        processor = BeamDataProcessor(ANGLE_FILE, RSS_FILE, OUTPUT_DIR)
        rss_mat, ue_ang, bs_ang = processor.pivot_data()
        print(f"✓ 数据加载成功")
        print(f"  - 矩阵维度: {rss_mat.shape}")
        print(f"  - UE角度范围: [{min(ue_ang):.1f}°, {max(ue_ang):.1f}°]")
        print(f"  - BS角度范围: [{min(bs_ang):.1f}°, {max(bs_ang):.1f}°]")
        print(f"  - 时间戳: {processor.timestamp}")
        
        # ===== 步骤2：LoS径识别（使用v1算法） =====
        print("\n[步骤2] LoS径识别 (v1 NN-OMP算法)...")
        los_estimator = LoSEstimator(ue_ang, bs_ang, rss_mat)
        los_estimator.construct_dictionary(grid_res=0.1, beam_width=1.4)
        los_paths_df = los_estimator.estimate_los_path(max_paths=3)
        
        print(f"✓ LoS径识别完成")
        if not los_paths_df.empty:
            los_path = los_paths_df[los_paths_df['Is_LoS'] == True]
            if not los_path.empty:
                los_info = los_path.iloc[0]
                print(f"  - 识别到LoS径: AoD={los_info['AoD']:.1f}°, AoA={los_info['AoA']:.1f}°, Power={los_info['Power']:.2f}")
            else:
                print(f"  - 未识别到LoS径")
                los_info = None
        else:
            print(f"  - 未识别到任何径")
            los_info = None
        
        # ===== 步骤3：NLoS径识别（使用v3算法） =====
        print("\n[步骤3] NLoS径识别 (v3 SM-SIC算法)...")
        nlos_estimator = NLoSEstimator(beam_width_deg=1.4)
        nlos_estimator.construct_dictionary(ue_ang, bs_ang, grid_res=0.1)
        
        # 传入LoS信息用于设置掩膜
        nlos_paths_df = nlos_estimator.estimate_nlos_paths(
            rss_mat, 
            los_info if los_info is not None else None,
            max_paths=3,
            proximity_mask_radius=10.0,
            cross_mask_width=10.0
        )
        
        print(f"✓ NLoS径识别完成")
        if not nlos_paths_df.empty:
            print(f"  - 识别到 {len(nlos_paths_df)} 条NLoS径:")
            for _, row in nlos_paths_df.iterrows():
                print(f"    Path {row['id']}: AoD={row['aod']:.1f}°, AoA={row['aoa']:.1f}°, Metric={row['metric']:.2f}")
        else:
            print(f"  - 未识别到NLoS径")
        
        # ===== 步骤4：融合可视化 =====
        print("\n[步骤4] 生成融合可视化结果...")
        saved_path = visualize_fusion_results(
            rss_mat, ue_ang, bs_ang,
            los_paths_df, nlos_paths_df,
            processor,
            save_plot=True,
            output_suffix='_fusion'
        )
        
        print(f"\n{'=' * 60}")
        print(f"处理完成！")
        print(f"输出路径: {saved_path}")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"\n错误: 处理过程中发生异常")
        print(f"详细信息: {e}")
        import traceback
        traceback.print_exc()