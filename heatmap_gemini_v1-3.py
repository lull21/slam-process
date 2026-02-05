import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.optimize import nnls
import seaborn as sns
import os
import re
from pathlib import Path
from matplotlib.colors import PowerNorm, LogNorm, TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 模块1：数据加载与预处理
# ==========================================
class BeamDataProcessor:
    def __init__(self, angle_file, rss_file, output_dir='D:\\桌面\\SLAMPro\\pic'):
        """
        初始化处理器
        :param angle_file: 波束角度映射文件路径
        :param rss_file: 波束测量数据文件路径
        :param output_dir: 输出目录路径
        """
        self.angle_map = self._load_angle_map(angle_file)
        self.rss_data = self._load_rss_data(rss_file)
        self.rss_matrix = None
        self.rss_matrix_processed = None  # 新增：预处理后的矩阵
        self.ue_angles = None
        self.bs_angles = None
        self.ue_ids = None
        self.bs_ids = None
        self.output_dir = output_dir
        self.rss_file = rss_file
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def _load_angle_map(self, path):
        try:
            df = pd.read_excel(path)
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            print(f"Error loading angle map: {e}")
            return {}

    def _load_rss_data(self, path):
        try:
            if path.endswith('.xlsx') or path.endswith('.xls'):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading RSS data: {e}")
            return pd.DataFrame()

    def extract_timestamp_from_filename(self):
        """从RSS文件名中提取时间戳"""
        filename = os.path.basename(self.rss_file)
        pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{6})'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1)
        else:
            pattern2 = r'(\d{4}-\d{2}-\d{2})[_\s]+(\d{6})'
            match2 = re.search(pattern2, filename)
            if match2:
                return f"{match2.group(1)} {match2.group(2)}"
            print(f"Warning: Could not extract timestamp from filename: {filename}")
            return None

    def get_output_path(self, suffix=''):
        """生成输出文件路径"""
        timestamp = self.extract_timestamp_from_filename()
        
        if timestamp:
            filename = f"{timestamp}{suffix}.png"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H%M%S')
            filename = f"{timestamp}{suffix}.png"
        
        return os.path.join(self.output_dir, filename)

    def pivot_data(self):
        """将长格式日志转换为矩阵格式，并处理重复测量"""
        # 对重复的(UE, BS)组合取平均值
        df_avg = self.rss_data.groupby(['UE_Beam', 'BS_Beam'])['RSS'].mean().reset_index()
        
        # 透视表构建矩阵
        pivot = df_avg.pivot(index='UE_Beam', columns='BS_Beam', values='RSS')
        
        # 填充缺失值为背景噪声底噪
        min_rss = df_avg['RSS'].min()
        pivot = pivot.fillna(min_rss)
        
        self.rss_matrix = pivot.values
        self.ue_ids = pivot.index.values
        self.bs_ids = pivot.columns.values
        
        # 映射物理角度
        self.ue_angles = np.array([self.angle_map.get(uid, np.nan) for uid in self.ue_ids])
        self.bs_angles = np.array([self.angle_map.get(bid, np.nan) for bid in self.bs_ids])
        
        # 移除无法映射角度的行/列
        valid_ue = ~np.isnan(self.ue_angles)
        valid_bs = ~np.isnan(self.bs_angles)
        
        self.rss_matrix = self.rss_matrix[valid_ue][:, valid_bs]
        self.ue_angles = self.ue_angles[valid_ue]
        self.bs_angles = self.bs_angles[valid_bs]
        
        # 应用数据预处理
        self.rss_matrix_processed = self._preprocess_power_data(self.rss_matrix)
        
        return self.rss_matrix, self.ue_angles, self.bs_angles

    def _preprocess_power_data(self, rss_matrix, method='adaptive'):
        """
        功率数据预处理，增强强弱信号区分度
        
        :param rss_matrix: 原始RSS矩阵
        :param method: 预处理方法
            - 'log': 对数变换
            - 'power': 幂次变换
            - 'quantile': 分位数归一化
            - 'adaptive': 自适应组合方法（推荐）
        :return: 处理后的矩阵
        """
        data = rss_matrix.copy()
        
        if method == 'log':
            # 对数变换：压缩高值，扩展低值差异
            # 将数据平移到正数域
            data_shifted = data - data.min() + 1
            processed = np.log10(data_shifted)
            
        elif method == 'power':
            # 幂次变换：通过幂指数放大差异
            # 归一化到[0,1]
            data_norm = (data - data.min()) / (data.max() - data.min())
            # 应用幂次变换 (gamma < 1 增强弱信号对比)
            gamma = 0.5
            processed = np.power(data_norm, gamma)
            # 还原到原始范围
            processed = processed * (data.max() - data.min()) + data.min()
            
        elif method == 'quantile':
            # 分位数归一化：使数据分布更均匀
            data_flat = data.flatten()
            sorted_data = np.sort(data_flat)
            ranks = np.searchsorted(sorted_data, data)
            processed = ranks.reshape(data.shape).astype(float)
            
        elif method == 'adaptive':
            # 自适应方法：组合多种技术
            
            # 1. 背景噪声抑制
            # 计算噪声阈值（使用中位数+标准差）
            median_val = np.median(data)
            std_val = np.std(data)
            noise_threshold = median_val + 0.5 * std_val
            
            # 对低于阈值的数据进行压制
            data_suppressed = data.copy()
            mask_noise = data < noise_threshold
            data_suppressed[mask_noise] = data_suppressed[mask_noise] * 0.3
            
            # 2. 动态范围压缩（对数变换）
            data_shifted = data_suppressed - data_suppressed.min() + 1
            data_log = np.log10(data_shifted)
            
            # 3. 对比度增强（直方图均衡化思想）
            data_flat = data_log.flatten()
            # 计算累积分布函数
            hist, bins = np.histogram(data_flat, bins=256)
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            
            # 映射到新的值域
            data_equalized = np.interp(data_log.flatten(), bins[:-1], cdf_normalized)
            processed = data_equalized.reshape(data.shape)
            
            # 4. 强信号进一步增强
            # 识别前10%的强信号点
            threshold_90 = np.percentile(processed, 90)
            mask_strong = processed > threshold_90
            processed[mask_strong] = processed[mask_strong] * 1.5
            
            # 归一化到合理范围
            processed = (processed - processed.min()) / (processed.max() - processed.min())
            processed = processed * (data.max() - data.min()) + data.min()
        
        else:
            processed = data
        
        return processed

# ==========================================
# 模块2：核心算法 - 稀疏正则化非负匹配追踪
# ==========================================
class MultipathEstimator:
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
        """高斯波束模型"""
        sigma = width / 2.355
        return np.exp(-((x - center)**2) / (2 * sigma**2))

    def construct_dictionary(self, grid_res=0.1, beam_width=1.4):
        """构建功率域字典"""
        self.aoa_grid = np.arange(np.min(self.ue_angles), np.max(self.ue_angles), grid_res)
        self.aod_grid = np.arange(np.min(self.bs_angles), np.max(self.bs_angles), grid_res)
        
        self.Phi_RX = self._gaussian_beam(self.ue_angles[:, None], self.aoa_grid[None, :], beam_width)
        self.Phi_TX = self._gaussian_beam(self.bs_angles[:, None], self.aod_grid[None, :], beam_width)
        
        return self.aoa_grid, self.aod_grid

    def estimate_paths_nn_omp(self, max_paths=3):
        """执行非负正交匹配追踪 (NN-OMP)"""
        residual = self.rss_vector.copy()
        selected_atoms = []
        
        rss_mat_shape = (len(self.ue_angles), len(self.bs_angles))
        
        for k in range(max_paths):
            res_mat = residual.reshape(rss_mat_shape)
            correlation = self.Phi_RX.T @ res_mat @ self.Phi_TX
            
            idx_aoa, idx_aod = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            atom_idx = (idx_aoa, idx_aod)
            if atom_idx in selected_atoms:
                break
            selected_atoms.append(atom_idx)
            
            current_atoms_list = []
            for (i_r, i_t) in selected_atoms:
                atom_vec = np.outer(self.Phi_RX[:, i_r], self.Phi_TX[:, i_t]).flatten()
                current_atoms_list.append(atom_vec)
            
            Dict_Active = np.column_stack(current_atoms_list)
            
            coeffs, rnorm = nnls(Dict_Active, self.rss_vector)
            
            reconstructed = Dict_Active @ coeffs
            residual = self.rss_vector - reconstructed
        
        path_params = []
        for idx, coeff in enumerate(coeffs):
            if coeff > 0:
                i_r, i_t = selected_atoms[idx]
                path_params.append({
                    'AoA': self.aoa_grid[i_r],
                    'AoD': self.aod_grid[i_t],
                    'Power': coeff,
                    'Is_LoS': False
                })
                    
        return pd.DataFrame(path_params)

# ==========================================
# 模块3：路径分类与可视化（优化版）
# ==========================================
def classify_and_plot(paths_df, estimator, processor, save_plot=True, output_suffix='', 
                     colormap='hot', norm_type='power', use_processed_data=True):
    """
    路径分类与优化可视化
    
    :param paths_df: 路径数据框
    :param estimator: 多径估计器
    :param processor: 数据处理器
    :param save_plot: 是否保存图像
    :param output_suffix: 输出文件名后缀
    :param colormap: 色图选择
        - 'hot': 黑-红-黄-白，高对比度（推荐）
        - 'inferno': 深紫-红-黄
        - 'plasma': 紫-粉-黄
        - 'turbo': 彩虹增强版
        - 'jet': 传统彩虹
    :param norm_type: 颜色映射归一化类型
        - 'linear': 线性映射
        - 'log': 对数映射
        - 'power': 幂次映射（推荐，gamma=0.5）
        - 'twoslope': 双斜率映射
    :param use_processed_data: 是否使用预处理后的数据
    :return: 保存的文件路径（如果保存）
    """
    # 1. 路径分类
    if not paths_df.empty:
        max_p_idx = paths_df['Power'].idxmax()
        paths_df.at[max_p_idx, 'Is_LoS'] = True
    
    # 2. 选择使用原始数据还是预处理数据
    if use_processed_data and processor.rss_matrix_processed is not None:
        data_to_plot = processor.rss_matrix_processed
    else:
        data_to_plot = processor.rss_matrix
    
    # 3. 生成高分辨率插值网格
    grid_resolution = 200  # 提高分辨率
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(processor.bs_angles), max(processor.bs_angles), grid_resolution),
        np.linspace(min(processor.ue_angles), max(processor.ue_angles), grid_resolution)
    )
    
    # 4. RBF插值
    bs_mesh, ue_mesh = np.meshgrid(processor.bs_angles, processor.ue_angles)
    rbf = Rbf(bs_mesh.flatten(), ue_mesh.flatten(), data_to_plot.flatten(), 
              function='thin_plate', smooth=0.1)  # 使用thin_plate样条，更平滑
    heatmap = rbf(grid_x, grid_y)
    
    # 5. 设置颜色归一化
    vmin, vmax = heatmap.min(), heatmap.max()
    
    if norm_type == 'log':
        # 对数归一化（需要确保数据为正）
        heatmap_shifted = heatmap - heatmap.min() + 1e-6
        norm = LogNorm(vmin=heatmap_shifted.min(), vmax=heatmap_shifted.max())
        heatmap_to_plot = heatmap_shifted
    elif norm_type == 'power':
        # 幂次归一化（gamma < 1 增强低值对比）
        norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        heatmap_to_plot = heatmap
    elif norm_type == 'twoslope':
        # 双斜率归一化（以中位数为中心）
        vcenter = np.median(heatmap)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        heatmap_to_plot = heatmap
    else:  # linear
        norm = None
        heatmap_to_plot = heatmap
    
    # 6. 绘图
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # 绘制热力图背景
    contour = ax.contourf(grid_x, grid_y, heatmap_to_plot, 
                          levels=100,  # 增加颜色级数
                          cmap=colormap, 
                          norm=norm,
                          extend='both')
    
    # 添加颜色条
    cbar = plt.colorbar(contour, ax=ax, label='RSS Power (Optimized Scale)', 
                       fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # 7. 绘制估计路径
    los = paths_df[paths_df['Is_LoS'] == True]
    nlos = paths_df[paths_df['Is_LoS'] == False]
    
    # 可选：绘制NLoS路径
    # if not nlos.empty:
    #     ax.scatter(nlos['AoD'], nlos['AoA'], 
    #               c='cyan', marker='x', s=200, 
    #               label='NLoS Paths', linewidth=3, 
    #               edgecolors='white', zorder=5)
    
    # 绘制LoS路径（更突出）
    if not los.empty:
        # 外圈白色边框
        ax.scatter(los['AoD'], los['AoA'], 
                  c='white', marker='o', s=400, 
                  edgecolors='black', linewidth=3, zorder=6)
        # 内圈红色标记
        ax.scatter(los['AoD'], los['AoA'], 
                  c='red', marker='o', s=250, 
                  edgecolors='yellow', linewidth=2, 
                  label='LoS Path', zorder=7)
    
    # 8. 标注（优化可读性）
    for _, row in paths_df.iterrows():
        if row['Is_LoS']:
            label = f"LoS\n({row['AoD']:.1f}°, {row['AoA']:.1f}°)"
            # 添加文本背景框
            ax.text(row['AoD'] + 2, row['AoA'] + 2, label, 
                   color='white', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='black', alpha=0.7, edgecolor='yellow'),
                   zorder=8)
    
    # 9. 图表装饰
    ax.set_xlabel('Angle of Departure (AoD) [deg]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Angle of Arrival (AoA) [deg]', fontsize=13, fontweight='bold')
    ax.set_title('mmWave Multipath Heatmap (Enhanced Contrast)', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='white')
    
    # 调整坐标轴刻度
    ax.tick_params(labelsize=10)
    
    # 10. 保存图像
    saved_path = None
    if save_plot:
        output_path = processor.get_output_path(suffix=output_suffix)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        saved_path = output_path
        print(f"优化热力图已保存至: {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return saved_path

# ==========================================
# 比较可视化函数（可选）
# ==========================================
def compare_visualizations(paths_df, estimator, processor):
    """
    生成对比图：原始数据 vs 优化数据
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for idx, (use_processed, title_suffix) in enumerate([(False, 'Original'), (True, 'Optimized')]):
        ax = axes[idx]
        
        # 选择数据
        if use_processed and processor.rss_matrix_processed is not None:
            data = processor.rss_matrix_processed
        else:
            data = processor.rss_matrix
        
        # 插值
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(processor.bs_angles), max(processor.bs_angles), 150),
            np.linspace(min(processor.ue_angles), max(processor.ue_angles), 150)
        )
        bs_mesh, ue_mesh = np.meshgrid(processor.bs_angles, processor.ue_angles)
        rbf = Rbf(bs_mesh.flatten(), ue_mesh.flatten(), data.flatten(), function='thin_plate')
        heatmap = rbf(grid_x, grid_y)
        
        # 绘制
        if use_processed:
            norm = PowerNorm(gamma=0.5)
            cmap = 'hot'
        else:
            norm = None
            cmap = 'viridis'
        
        contour = ax.contourf(grid_x, grid_y, heatmap, levels=80, cmap=cmap, norm=norm)
        plt.colorbar(contour, ax=ax, label='RSS Power')
        
        # 标记LoS
        los = paths_df[paths_df['Is_LoS'] == True]
        if not los.empty:
            ax.scatter(los['AoD'], los['AoA'], c='red', marker='o', s=200, 
                      edgecolors='white', linewidth=2, label='LoS', zorder=5)
        
        ax.set_xlabel('AoD [deg]', fontsize=11)
        ax.set_ylabel('AoA [deg]', fontsize=11)
        ax.set_title(f'Heatmap - {title_suffix} Data', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = processor.get_output_path(suffix='_comparison')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {output_path}")
    plt.show()

# ==========================================
# 主执行流程示例
# ==========================================
if __name__ == '__main__':
    # 指定文件路径
    angle_file = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    rss_file = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115200_filtered.xlsx'
    output_dir = 'D:\\桌面\\SLAMPro\\pic_v1-3'
    
    # 初始化处理器
    processor = BeamDataProcessor(angle_file, rss_file, output_dir)
    
    # 数据预处理
    rss_mat, ue_ang, bs_ang = processor.pivot_data()
    
    print(f"数据统计:")
    print(f"  原始数据范围: [{rss_mat.min():.2f}, {rss_mat.max():.2f}]")
    if processor.rss_matrix_processed is not None:
        print(f"  处理后范围: [{processor.rss_matrix_processed.min():.2f}, "
              f"{processor.rss_matrix_processed.max():.2f}]")
    
    # 构建估计器
    estimator = MultipathEstimator(ue_ang, bs_ang, rss_mat)
    estimator.construct_dictionary()
    
    # 估计多径
    paths = estimator.estimate_paths_nn_omp(max_paths=3)
    print(f"\n检测到 {len(paths)} 条路径:")
    print(paths)
    
    # ===== 方案1：使用优化后的单一可视化 =====
    print("\n生成优化热力图...")
    saved_path = classify_and_plot(
        paths, estimator, processor, 
        save_plot=True,
        output_suffix='_optimized',
        colormap='inferno',          # 可选: 'hot', 'inferno', 'plasma', 'turbo'
        norm_type='log',       # 可选: 'linear', 'log对数映射,适合极大动态范围', 'power幂次映射(推荐,gamma=0.5)', 'twoslope双斜率,突出中位数附近变化'
        use_processed_data=True  # 使用预处理数据
    )
    
    # ===== 方案2：生成对比图（可选） =====
    print("\n生成对比图...")
    compare_visualizations(paths, estimator, processor)
    
    print(f"\n处理完成！主输出路径: {saved_path}")