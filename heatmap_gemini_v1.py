import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.optimize import nnls
import seaborn as sns
import os
import re
from pathlib import Path

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
        self.ue_angles = None
        self.bs_angles = None
        self.ue_ids = None
        self.bs_ids = None
        self.output_dir = output_dir
        self.rss_file = rss_file
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def _load_angle_map(self, path):
        # 文件包含 BeamID, Angle 两列
        # 根据Snippet ，数据可能没有表头，需根据实际情况调整
        try:
            df = pd.read_excel(path)  # 或 read_excel
            # 构建字典映射: BeamID -> Angle
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            print(f"Error loading angle map: {e}")
            return {}

    def _load_rss_data(self, path):
        # 加载测量数据，列名为 UE_Beam, BS_Beam, RSS
        try:
            if path.endswith('.xlsx') or path.endswith('.xls'):
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading RSS data: {e}")
            return pd.DataFrame()

    def extract_timestamp_from_filename(self):
        """
        从RSS文件名中提取时间戳
        例如: Serial Debug 2026-01-26 165358_filtered.xlsx -> 2026-01-26 165358
        """
        filename = os.path.basename(self.rss_file)
        
        # 匹配时间戳模式: YYYY-MM-DD HHMMSS
        pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{6})'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1)
        else:
            # 如果没有匹配到，尝试其他可能的格式
            # 例如: YYYY-MM-DD_HHMMSS
            pattern2 = r'(\d{4}-\d{2}-\d{2})[_\s]+(\d{6})'
            match2 = re.search(pattern2, filename)
            if match2:
                return f"{match2.group(1)} {match2.group(2)}"
            
            # 如果仍然没有匹配，返回None
            print(f"Warning: Could not extract timestamp from filename: {filename}")
            return None

    def get_output_path(self, suffix=''):
        """
        生成输出文件路径
        :param suffix: 文件名后缀（可选）
        :return: 完整的输出文件路径
        """
        timestamp = self.extract_timestamp_from_filename()
        
        if timestamp:
            filename = f"{timestamp}{suffix}.png"
        else:
            # 使用当前时间作为备用
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H%M%S')
            filename = f"{timestamp}{suffix}.png"
        
        return os.path.join(self.output_dir, filename)

    def pivot_data(self):
        """
        将长格式日志转换为矩阵格式，并处理重复测量
        """
        # 对重复的(UE, BS)组合取平均值
        df_avg = self.rss_data.groupby(['UE_Beam', 'BS_Beam'])['RSS'].mean().reset_index()
        
        # 透视表构建矩阵
        pivot = df_avg.pivot(index='UE_Beam', columns='BS_Beam', values='RSS')
        
        # 填充缺失值为背景噪声底噪 (例如最小值或0)
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
        
        return self.rss_matrix, self.ue_angles, self.bs_angles

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
        """ 高斯波束模型 """
        sigma = width / 2.355  # FWHM转换
        return np.exp(-((x - center)**2) / (2 * sigma**2))

    def construct_dictionary(self, grid_res=0.1, beam_width=1.4):
        """
        构建功率域字典
        :param grid_res: 搜索网格分辨率（度）
        :param beam_width: 波束宽度（度）
        """
        # 定义搜索空间
        self.aoa_grid = np.arange(np.min(self.ue_angles), np.max(self.ue_angles), grid_res)
        self.aod_grid = np.arange(np.min(self.bs_angles), np.max(self.bs_angles), grid_res)
        
        # 构建接收和发射字典 (Rows=Measured Beams, Cols=Grid Angles)
        # Phi_RX[i, k] = 接收波束i对角度k的响应
        self.Phi_RX = self._gaussian_beam(self.ue_angles[:, None], self.aoa_grid[None, :], beam_width)
        self.Phi_TX = self._gaussian_beam(self.bs_angles[:, None], self.aod_grid[None, :], beam_width)
        
        return self.aoa_grid, self.aod_grid

    def estimate_paths_nn_omp(self, max_paths=3):
        """
        执行非负正交匹配追踪 (NN-OMP)
        """
        residual = self.rss_vector.copy()
        selected_atoms = []
        
        rss_mat_shape = (len(self.ue_angles), len(self.bs_angles))
        
        for k in range(max_paths):
            # 1. 计算相关性 (2D Correlation)
            # 利用克罗内克积结构的性质加速计算: Corr = Phi_RX.T * Residual_Mat * Phi_TX
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
            # Atom_k = vec( phi_rx_i * phi_tx_j.T )
            current_atoms_list = []
            for (i_r, i_t) in selected_atoms:
                atom_vec = np.outer(self.Phi_RX[:, i_r], self.Phi_TX[:, i_t]).flatten()
                current_atoms_list.append(atom_vec)
            
            Dict_Active = np.column_stack(current_atoms_list)
            
            # 4. 求解非负最小二乘 (NNLS)
            # min || y - D*x ||^2 s.t. x >= 0
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
                    
        return pd.DataFrame(path_params)

# ==========================================
# 模块3：路径分类与可视化
# ==========================================
def classify_and_plot(paths_df, estimator, processor, save_plot=True, output_suffix=''):
    """
    路径分类与可视化
    :param paths_df: 路径数据框
    :param estimator: 多径估计器
    :param processor: 数据处理器
    :param save_plot: 是否保存图像
    :param output_suffix: 输出文件名后缀
    :return: 保存的文件路径（如果保存）
    """
    # 1. 路径分类 (简单的功率判据)
    if not paths_df.empty:
        max_p_idx = paths_df['Power'].idxmax()
        paths_df.at[max_p_idx, 'Is_LoS'] = True
    
    # 2. 生成热力图 (RBF插值用于背景显示)
    # 创建网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(processor.bs_angles), max(processor.bs_angles), 100),
        np.linspace(min(processor.ue_angles), max(processor.ue_angles), 100)
    )
    
    # 展平原始数据用于插值
    bs_mesh, ue_mesh = np.meshgrid(processor.bs_angles, processor.ue_angles)
    rbf = Rbf(bs_mesh.flatten(), ue_mesh.flatten(), processor.rss_matrix.flatten(), function='linear')
    heatmap = rbf(grid_x, grid_y)
    
    # 3. 绘图
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图背景
    contour = plt.contourf(grid_x, grid_y, heatmap, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Interpolated RSS Power')
    
    # 绘制估计路径
    los = paths_df[paths_df['Is_LoS'] == True]
    nlos = paths_df[paths_df['Is_LoS'] == False]
    
    # if not nlos.empty:
    #     plt.scatter(nlos['AoD'], nlos['AoA'], c='white', marker='x', s=100, label='NLoS Paths', linewidth=2)
    if not los.empty:
        plt.scatter(los['AoD'], los['AoA'], c='red', marker='o', s=150, edgecolors='black', label='LoS Path', linewidth=2)
    
    # 标注
    for _, row in paths_df.iterrows():
        # label = f"{'LoS' if row['Is_LoS'] else 'NLoS'}\n({row['AoD']:.1f}, {row['AoA']:.1f})"
        # plt.text(row['AoD'] + 1, row['AoA'] + 1, label, color='white', fontweight='bold')
        if row['Is_LoS']:  # 仅标记 LoS
            label = f"LoS\n({row['AoD']:.1f}, {row['AoA']:.1f})"
            plt.text(row['AoD'] + 1, row['AoA'] + 1, label, color='white', fontweight='bold')

    plt.xlabel('Angle of Departure (AoD) [deg]')
    plt.ylabel('Angle of Arrival (AoA) [deg]')
    plt.title('mmWave Multipath Heatmap & Estimation Results')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. 保存图像
    saved_path = None
    if save_plot:
        output_path = processor.get_output_path(suffix=output_suffix)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f"热力图已保存至: {output_path}")
    
    plt.show()
    
    return saved_path

# ==========================================
# 主执行流程示例
# ==========================================
if __name__ == '__main__':
    # 使用示例：
    # 指定文件路径
    angle_file = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    rss_file = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115200_filtered.xlsx'
    output_dir = 'D:\\桌面\\SLAMPro\\pic'
    
    # 初始化处理器
    processor = BeamDataProcessor(angle_file, rss_file, output_dir)
    
    # 数据预处理
    rss_mat, ue_ang, bs_ang = processor.pivot_data()
    
    # 构建估计器
    estimator = MultipathEstimator(ue_ang, bs_ang, rss_mat)
    estimator.construct_dictionary()
    
    # 估计多径
    paths = estimator.estimate_paths_nn_omp(max_paths=3)
    
    # 分类和可视化（自动保存）
    saved_path = classify_and_plot(paths, estimator, processor, save_plot=True)
    
    print(f"处理完成！输出路径: {saved_path}")