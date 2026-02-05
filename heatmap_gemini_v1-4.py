import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.optimize import nnls
import seaborn as sns
import os
import re
from pathlib import Path
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==========================================
# 模块1：数据加载与预处理
# ==========================================
class BeamDataProcessor:
    def __init__(self, angle_file, rss_file, output_dir='./output'):
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
        """
        加载角度映射文件
        文件包含 BeamID, Angle 两列
        """
        try:
            # 尝试读取Excel文件
            if path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
            # 检查列数
            if df.shape[1] < 2:
                raise ValueError(f"角度映射文件应至少包含2列，实际只有{df.shape[1]}列")
            
            # 构建字典映射: BeamID -> Angle
            angle_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            print(f"成功加载角度映射文件，共 {len(angle_dict)} 个波束")
            return angle_dict
        except Exception as e:
            print(f"加载角度映射文件时出错: {e}")
            return {}

    def _load_rss_data(self, path):
        """
        加载测量数据，列名为 UE_Beam, BS_Beam, RSS
        """
        try:
            if path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
            # 检查必需的列
            required_cols = ['UE_Beam', 'BS_Beam', 'RSS']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"警告：缺少列 {missing_cols}，将尝试使用前三列")
                df.columns = required_cols[:len(df.columns)]
            
            print(f"成功加载RSS数据文件，共 {len(df)} 行记录")
            return df
        except Exception as e:
            print(f"加载RSS数据文件时出错: {e}")
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
            print(f"警告：无法从文件名中提取时间戳: {filename}")
            return None

    def get_output_path(self, suffix=''):
        """
        生成输出文件路径
        :param suffix: 文件名后缀（可选）
        :return: 完整的输出文件路径
        """
        timestamp = self.extract_timestamp_from_filename()
        
        if timestamp:
            # 将时间戳中的空格替换为下划线，避免文件名问题
            timestamp = timestamp.replace(' ', '_')
            filename = f"{timestamp}{suffix}.png"
        else:
            # 使用当前时间作为备用
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            filename = f"{timestamp}{suffix}.png"
        
        return os.path.join(self.output_dir, filename)

    def pivot_data(self):
        """
        将长格式日志转换为矩阵格式，并处理重复测量
        """
        if self.rss_data.empty:
            raise ValueError("RSS数据为空，无法进行数据透视")
        
        # 对重复的(UE, BS)组合取平均值
        df_avg = self.rss_data.groupby(['UE_Beam', 'BS_Beam'])['RSS'].mean().reset_index()
        
        # 透视表构建矩阵
        pivot = df_avg.pivot(index='UE_Beam', columns='BS_Beam', values='RSS')
        
        # 填充缺失值为背景噪声底噪 (例如最小值)
        min_rss = df_avg['RSS'].min()
        pivot = pivot.fillna(min_rss)
        
        self.rss_matrix = pivot.values
        self.ue_ids = pivot.index.values
        self.bs_ids = pivot.columns.values
        
        # 映射物理角度
        self.ue_angles = np.array([self.angle_map.get(uid, np.nan) for uid in self.ue_ids])
        self.bs_angles = np.array([self.angle_map.get(bid, np.nan) for bid in self.bs_ids])
        
        # 检查是否有无法映射的角度
        n_invalid_ue = np.sum(np.isnan(self.ue_angles))
        n_invalid_bs = np.sum(np.isnan(self.bs_angles))
        if n_invalid_ue > 0:
            print(f"警告：{n_invalid_ue} 个UE波束无法映射到角度")
        if n_invalid_bs > 0:
            print(f"警告：{n_invalid_bs} 个BS波束无法映射到角度")
        
        # 移除无法映射角度的行/列
        valid_ue = ~np.isnan(self.ue_angles)
        valid_bs = ~np.isnan(self.bs_angles)
        
        self.rss_matrix = self.rss_matrix[valid_ue][:, valid_bs]
        self.ue_angles = self.ue_angles[valid_ue]
        self.bs_angles = self.bs_angles[valid_bs]
        
        print(f"数据透视完成，矩阵大小: {self.rss_matrix.shape}")
        print(f"UE角度范围: [{self.ue_angles.min():.1f}, {self.ue_angles.max():.1f}]")
        print(f"BS角度范围: [{self.bs_angles.min():.1f}, {self.bs_angles.max():.1f}]")
        
        return self.rss_matrix, self.ue_angles, self.bs_angles

# ==========================================
# 模块2：核心算法 - 稀疏正则化非负匹配追踪
# ==========================================
class MultipathEstimator:
    def __init__(self, ue_angles, bs_angles, rss_matrix):
        """
        初始化多径估计器
        :param ue_angles: UE角度数组
        :param bs_angles: BS角度数组
        :param rss_matrix: RSS功率矩阵
        """
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
        高斯波束模型
        :param x: 角度位置
        :param center: 波束中心
        :param width: 波束宽度 (FWHM)
        """
        sigma = width / 2.355  # FWHM转换为标准差
        return np.exp(-((x - center)**2) / (2 * sigma**2))

    def construct_dictionary(self, grid_res=0.1, beam_width=1.4):
        """
        构建功率域字典
        :param grid_res: 搜索网格分辨率（度）
        :param beam_width: 波束宽度（度，FWHM）
        """
        # 定义搜索空间 - 添加边界检查
        ue_min, ue_max = np.min(self.ue_angles), np.max(self.ue_angles)
        bs_min, bs_max = np.min(self.bs_angles), np.max(self.bs_angles)
        
        # 确保至少有一定数量的网格点
        n_aoa_points = max(int((ue_max - ue_min) / grid_res) + 1, 10)
        n_aod_points = max(int((bs_max - bs_min) / grid_res) + 1, 10)
        
        self.aoa_grid = np.linspace(ue_min, ue_max, n_aoa_points)
        self.aod_grid = np.linspace(bs_min, bs_max, n_aod_points)
        
        print(f"构建字典：AoA网格 {len(self.aoa_grid)} 点，AoD网格 {len(self.aod_grid)} 点")
        
        # 构建接收和发射字典 (Rows=Measured Beams, Cols=Grid Angles)
        # Phi_RX[i, k] = 接收波束i对角度k的响应
        self.Phi_RX = self._gaussian_beam(self.ue_angles[:, None], self.aoa_grid[None, :], beam_width)
        self.Phi_TX = self._gaussian_beam(self.bs_angles[:, None], self.aod_grid[None, :], beam_width)
        
        return self.aoa_grid, self.aod_grid

    def estimate_paths_nn_omp(self, max_paths=10, min_power_ratio=0.01):
        """
        执行非负正交匹配追踪 (NN-OMP)
        :param max_paths: 最大路径数
        :param min_power_ratio: 最小功率比（相对于最大值），低于此值的路径将被忽略
        """
        if self.Phi_RX is None or self.Phi_TX is None:
            raise ValueError("请先调用 construct_dictionary() 构建字典")
        
        residual = self.rss_vector.copy()
        selected_atoms = []
        
        rss_mat_shape = (len(self.ue_angles), len(self.bs_angles))
        
        print(f"开始NN-OMP估计，最大路径数: {max_paths}")
        
        for k in range(max_paths):
            # 1. 计算相关性 (2D Correlation)
            res_mat = residual.reshape(rss_mat_shape)
            correlation = self.Phi_RX.T @ res_mat @ self.Phi_TX
            
            # 2. 寻找最大相关性位置
            max_corr = np.max(correlation)
            if max_corr <= 0:
                print(f"迭代 {k}: 相关性为非正值，停止搜索")
                break
                
            idx_aoa, idx_aod = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # 记录选中的原子索引
            atom_idx = (idx_aoa, idx_aod)
            if atom_idx in selected_atoms:
                print(f"迭代 {k}: 检测到重复原子，停止搜索")
                break
            selected_atoms.append(atom_idx)
            
            # 3. 构建当前支撑集子字典
            current_atoms_list = []
            for (i_r, i_t) in selected_atoms:
                atom_vec = np.outer(self.Phi_RX[:, i_r], self.Phi_TX[:, i_t]).flatten()
                current_atoms_list.append(atom_vec)
            
            Dict_Active = np.column_stack(current_atoms_list)
            
            # 4. 求解非负最小二乘 (NNLS)
            try:
                coeffs, rnorm = nnls(Dict_Active, self.rss_vector)
            except Exception as e:
                print(f"迭代 {k}: NNLS求解失败 - {e}")
                break
            
            # 5. 更新残差
            reconstructed = Dict_Active @ coeffs
            residual = self.rss_vector - reconstructed
            
            # 计算残差范数的减少量
            residual_norm = np.linalg.norm(residual)
            print(f"迭代 {k}: AoA={self.aoa_grid[idx_aoa]:.1f}°, AoD={self.aod_grid[idx_aod]:.1f}°, "
                  f"残差范数={residual_norm:.4f}")
        
        # 更新路径列表 - 过滤掉功率过小的路径
        if len(coeffs) > 0:
            max_coeff = np.max(coeffs)
            path_params = []
            for idx, coeff in enumerate(coeffs):
                if coeff > max_coeff * min_power_ratio:  # 只保留相对功率大于阈值的路径
                    i_r, i_t = selected_atoms[idx]
                    path_params.append({
                        'AoA': self.aoa_grid[i_r],
                        'AoD': self.aod_grid[i_t],
                        'Power': coeff,
                        'Is_LoS': False  # 稍后分类
                    })
            
            print(f"估计完成，保留 {len(path_params)} 条有效路径")
            return pd.DataFrame(path_params)
        else:
            print("警告：未找到有效路径")
            return pd.DataFrame(columns=['AoA', 'AoD', 'Power', 'Is_LoS'])

# ==========================================
# 模块3：路径分类与可视化
# ==========================================
def classify_and_plot(paths_df, estimator, processor, 
                      save_plot=True, output_suffix='',
                      power_thresh_db=10.0,
                      angle_thresh_deg=10.0,
                      show_nlos=True):
    """
    路径分类与可视化
    :param paths_df: 路径数据框
    :param estimator: 多径估计器
    :param processor: 数据处理器
    :param save_plot: 是否保存图像
    :param output_suffix: 输出文件后缀
    :param power_thresh_db: 判定弱信号的阈值（dB）
    :param angle_thresh_deg: 判定角度偏离的阈值（度）
    :param show_nlos: 是否在图中显示NLoS路径
    :return: 保存的文件路径
    """
    
    # 1. 路径分类逻辑
    if not paths_df.empty:
        # A. 找到基准：功率最大的路径作为 LoS 参考
        max_idx = paths_df['Power'].idxmax()
        ref_power = paths_df.loc[max_idx, 'Power']
        ref_aoa = paths_df.loc[max_idx, 'AoA']
        ref_aod = paths_df.loc[max_idx, 'AoD']
        
        print(f"\n基准 LoS 路径 -> Power: {ref_power:.4f}, AoA: {ref_aoa:.1f}°, AoD: {ref_aod:.1f}°")
        print("-" * 60)
        
        # B. 遍历所有路径进行判断
        for idx, row in paths_df.iterrows():
            # 计算功率比 (dB) - 添加数值稳定性检查
            power_ratio = row['Power'] / (ref_power + 1e-12)
            if power_ratio > 0:
                p_ratio_db = 10 * np.log10(power_ratio)
            else:
                p_ratio_db = -1000  # 设置一个极小值
            
            is_weak = p_ratio_db < -power_thresh_db
            
            # 计算角度偏差
            diff_aoa = abs(row['AoA'] - ref_aoa)
            diff_aod = abs(row['AoD'] - ref_aod)
            is_far = (diff_aoa > angle_thresh_deg) or (diff_aod > angle_thresh_deg)
            
            # C. 融合判断
            if is_weak and is_far:
                paths_df.at[idx, 'Is_LoS'] = False
                kind = "NLoS (弱且远)"
            else:
                paths_df.at[idx, 'Is_LoS'] = True
                kind = "LoS/显著"
            
            # 打印路径信息
            print(f"路径 {idx}: AoA={row['AoA']:.1f}°, AoD={row['AoD']:.1f}°, "
                  f"功率比={p_ratio_db:.1f}dB, "
                  f"角度偏差=({diff_aoa:.1f}°, {diff_aod:.1f}°) -> {kind}")
        
        print("-" * 60)
    
    # 2. 生成热力图 (RBF插值)
    try:
        # 创建网格
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(processor.bs_angles), max(processor.bs_angles), 100),
            np.linspace(min(processor.ue_angles), max(processor.ue_angles), 100)
        )
        
        # 展平数据用于插值
        bs_mesh, ue_mesh = np.meshgrid(processor.bs_angles, processor.ue_angles)
        
        # 使用RBF插值 - 添加异常处理
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rbf = Rbf(bs_mesh.flatten(), ue_mesh.flatten(), 
                     processor.rss_matrix.flatten(), 
                     function='linear', smooth=0.1)
            heatmap = rbf(grid_x, grid_y)
    except Exception as e:
        print(f"插值警告: {e}")
        heatmap = np.zeros_like(grid_x)
    
    # 3. 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制背景热力图
    contour = ax.contourf(grid_x, grid_y, heatmap, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax, label='插值RSS功率')
    
    # 分离有效和无效路径
    sig_paths = paths_df[paths_df['Is_LoS'] == True]
    nlos_paths = paths_df[paths_df['Is_LoS'] == False]
    
    # 绘制路径点
    if show_nlos and not nlos_paths.empty:
        ax.scatter(nlos_paths['AoD'], nlos_paths['AoA'], 
                  c='white', marker='x', s=100, 
                  label='NLoS (弱且远)', alpha=0.7, linewidths=2)
    
    if not sig_paths.empty:
        ax.scatter(sig_paths['AoD'], sig_paths['AoA'], 
                  c='red', marker='o', s=200, 
                  edgecolors='yellow', label='LoS/显著', 
                  linewidth=2.5, zorder=5)
        
        # 标注有效路径的角度
        for _, row in sig_paths.iterrows():
            label = f"({row['AoD']:.1f}°, {row['AoA']:.1f}°)"
            ax.text(row['AoD'] + 0.5, row['AoA'] + 0.5, label, 
                   color='white', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    ax.set_xlabel('出发角 (AoD) [度]', fontsize=12)
    ax.set_ylabel('到达角 (AoA) [度]', fontsize=12)
    ax.set_title(f'多径估计热力图 (功率阈值: -{power_thresh_db}dB, 角度阈值: ±{angle_thresh_deg}°)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 4. 保存
    saved_path = None
    if save_plot:
        output_path = processor.get_output_path(suffix=output_suffix)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f"\n热力图已保存至: {output_path}")
    
    plt.show()
    
    return saved_path

# ==========================================
# 主执行流程
# ==========================================
def main():
    """
    主执行函数
    """
    # 配置文件路径 - 请根据实际情况修改
    angle_file = 'beam_angle.xlsx'  # 角度映射文件
    rss_file = 'rss_data.xlsx'      # RSS测量数据文件
    output_dir = './output'          # 输出目录
    
    try:
        # 1. 初始化处理器
        print("=" * 60)
        print("开始多径估计处理流程")
        print("=" * 60)
        processor = BeamDataProcessor(angle_file, rss_file, output_dir)
        
        # 2. 数据预处理
        print("\n步骤 1/3: 数据预处理...")
        rss_mat, ue_ang, bs_ang = processor.pivot_data()
        
        # 3. 构建估计器
        print("\n步骤 2/3: 构建多径估计器...")
        estimator = MultipathEstimator(ue_ang, bs_ang, rss_mat)
        estimator.construct_dictionary(grid_res=0.1, beam_width=1.4)
        
        # 4. 估计多径
        print("\n步骤 3/3: 执行多径估计...")
        paths = estimator.estimate_paths_nn_omp(max_paths=5, min_power_ratio=0.01)
        
        # 5. 分类和可视化
        if not paths.empty:
            print("\n生成可视化...")
            saved_path = classify_and_plot(
                paths, estimator, processor,
                save_plot=True,
                power_thresh_db=10.0,   # 功率阈值
                angle_thresh_deg=15.0,  # 角度阈值
                show_nlos=True          # 显示NLoS路径
            )
            
            print("\n" + "=" * 60)
            print(f"处理完成！")
            print(f"输出路径: {saved_path}")
            print(f"检测到 {len(paths)} 条路径")
            print(f"其中 LoS/显著路径: {sum(paths['Is_LoS'])} 条")
            print("=" * 60)
        else:
            print("\n警告：未检测到有效路径")
            
    except Exception as e:
        print(f"\n错误：处理过程中发生异常 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 方式1：使用默认配置运行
    # main()
    
    # 方式2：自定义配置运行（取消注释以使用）
    # """
    angle_file = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    rss_file = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115200_filtered.xlsx'
    output_dir = 'D:\\桌面\\SLAMPro\\pic_v1-4'
    
    processor = BeamDataProcessor(angle_file, rss_file, output_dir)
    rss_mat, ue_ang, bs_ang = processor.pivot_data()
    
    estimator = MultipathEstimator(ue_ang, bs_ang, rss_mat)
    estimator.construct_dictionary()
    
    paths = estimator.estimate_paths_nn_omp(max_paths=30)
    
    saved_path = classify_and_plot(
        paths, estimator, processor,
        save_plot=True,
        power_thresh_db=0.0001,
        angle_thresh_deg=20.0
    )
    # """