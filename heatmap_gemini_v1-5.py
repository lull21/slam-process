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
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        
        os.makedirs(output_dir, exist_ok=True)

    def _load_angle_map(self, path):
        """加载角度映射文件"""
        try:
            if path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
            if df.shape[1] < 2:
                raise ValueError(f"角度映射文件应至少包含2列，实际只有{df.shape[1]}列")
            
            angle_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            print(f"成功加载角度映射文件，共 {len(angle_dict)} 个波束")
            return angle_dict
        except Exception as e:
            print(f"加载角度映射文件时出错: {e}")
            return {}

    def _load_rss_data(self, path):
        """加载测量数据"""
        try:
            if path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            
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
            
            print(f"警告：无法从文件名中提取时间戳: {filename}")
            return None

    def get_output_path(self, suffix=''):
        """生成输出文件路径"""
        timestamp = self.extract_timestamp_from_filename()
        
        if timestamp:
            timestamp = timestamp.replace(' ', '_')
            filename = f"{timestamp}{suffix}.png"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            filename = f"{timestamp}{suffix}.png"
        
        return os.path.join(self.output_dir, filename)

    def pivot_data(self):
        """将长格式日志转换为矩阵格式"""
        if self.rss_data.empty:
            raise ValueError("RSS数据为空，无法进行数据透视")
        
        df_avg = self.rss_data.groupby(['UE_Beam', 'BS_Beam'])['RSS'].mean().reset_index()
        pivot = df_avg.pivot(index='UE_Beam', columns='BS_Beam', values='RSS')
        
        min_rss = df_avg['RSS'].min()
        pivot = pivot.fillna(min_rss)
        
        self.rss_matrix = pivot.values
        self.ue_ids = pivot.index.values
        self.bs_ids = pivot.columns.values
        
        self.ue_angles = np.array([self.angle_map.get(uid, np.nan) for uid in self.ue_ids])
        self.bs_angles = np.array([self.angle_map.get(bid, np.nan) for bid in self.bs_ids])
        
        n_invalid_ue = np.sum(np.isnan(self.ue_angles))
        n_invalid_bs = np.sum(np.isnan(self.bs_angles))
        if n_invalid_ue > 0:
            print(f"警告：{n_invalid_ue} 个UE波束无法映射到角度")
        if n_invalid_bs > 0:
            print(f"警告：{n_invalid_bs} 个BS波束无法映射到角度")
        
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
        """初始化多径估计器"""
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
        ue_min, ue_max = np.min(self.ue_angles), np.max(self.ue_angles)
        bs_min, bs_max = np.min(self.bs_angles), np.max(self.bs_angles)
        
        n_aoa_points = max(int((ue_max - ue_min) / grid_res) + 1, 10)
        n_aod_points = max(int((bs_max - bs_min) / grid_res) + 1, 10)
        
        self.aoa_grid = np.linspace(ue_min, ue_max, n_aoa_points)
        self.aod_grid = np.linspace(bs_min, bs_max, n_aod_points)
        
        print(f"构建字典：AoA网格 {len(self.aoa_grid)} 点，AoD网格 {len(self.aod_grid)} 点")
        
        self.Phi_RX = self._gaussian_beam(self.ue_angles[:, None], self.aoa_grid[None, :], beam_width)
        self.Phi_TX = self._gaussian_beam(self.bs_angles[:, None], self.aod_grid[None, :], beam_width)
        
        return self.aoa_grid, self.aod_grid

    def estimate_paths_nn_omp(self, max_paths=10, min_power_ratio=0.01):
        """执行非负正交匹配追踪 (NN-OMP)"""
        if self.Phi_RX is None or self.Phi_TX is None:
            raise ValueError("请先调用 construct_dictionary() 构建字典")
        
        residual = self.rss_vector.copy()
        selected_atoms = []
        
        rss_mat_shape = (len(self.ue_angles), len(self.bs_angles))
        
        print(f"开始NN-OMP估计，最大路径数: {max_paths}")
        
        for k in range(max_paths):
            res_mat = residual.reshape(rss_mat_shape)
            correlation = self.Phi_RX.T @ res_mat @ self.Phi_TX
            
            max_corr = np.max(correlation)
            if max_corr <= 0:
                print(f"迭代 {k}: 相关性为非正值，停止搜索")
                break
                
            idx_aoa, idx_aod = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            atom_idx = (idx_aoa, idx_aod)
            if atom_idx in selected_atoms:
                print(f"迭代 {k}: 检测到重复原子，停止搜索")
                break
            selected_atoms.append(atom_idx)
            
            current_atoms_list = []
            for (i_r, i_t) in selected_atoms:
                atom_vec = np.outer(self.Phi_RX[:, i_r], self.Phi_TX[:, i_t]).flatten()
                current_atoms_list.append(atom_vec)
            
            Dict_Active = np.column_stack(current_atoms_list)
            
            try:
                coeffs, rnorm = nnls(Dict_Active, self.rss_vector)
            except Exception as e:
                print(f"迭代 {k}: NNLS求解失败 - {e}")
                break
            
            reconstructed = Dict_Active @ coeffs
            residual = self.rss_vector - reconstructed
            
            residual_norm = np.linalg.norm(residual)
            print(f"迭代 {k}: AoA={self.aoa_grid[idx_aoa]:.1f}°, AoD={self.aod_grid[idx_aod]:.1f}°, "
                  f"残差范数={residual_norm:.4f}")
        
        if len(coeffs) > 0:
            max_coeff = np.max(coeffs)
            path_params = []
            for idx, coeff in enumerate(coeffs):
                if coeff > max_coeff * min_power_ratio:
                    i_r, i_t = selected_atoms[idx]
                    path_params.append({
                        'AoA': self.aoa_grid[i_r],
                        'AoD': self.aod_grid[i_t],
                        'Power': coeff,
                        'PathType': 'Unknown'  # 待分类
                    })
            
            print(f"估计完成，保留 {len(path_params)} 条有效路径")
            return pd.DataFrame(path_params)
        else:
            print("警告：未找到有效路径")
            return pd.DataFrame(columns=['AoA', 'AoD', 'Power', 'PathType'])

# ==========================================
# 模块3：高级路径分类器
# ==========================================
class PathClassifier:
    """
    高级路径分类器：严格区分LoS、旁瓣效应、NLoS
    """
    def __init__(self, paths_df, 
                 sidelobe_width_aoa=45.0,     # AoA方向旁瓣宽度
                 sidelobe_width_aod=45.0,     # AoD方向旁瓣宽度
                 nlos_power_thresh_db=10.0,   # NLoS功率阈值
                 nlos_min_angle_sep=20.0):    # NLoS最小角度分离
        """
        初始化路径分类器
        :param paths_df: 初步估计的路径DataFrame
        :param sidelobe_width_aoa: AoA方向旁瓣宽度范围（度）
        :param sidelobe_width_aod: AoD方向旁瓣宽度范围（度）
        :param nlos_power_thresh_db: NLoS判定的功率阈值（dB）
        :param nlos_min_angle_sep: NLoS路径之间的最小角度分离（度）
        """
        self.paths_df = paths_df.copy()
        self.sidelobe_width_aoa = sidelobe_width_aoa
        self.sidelobe_width_aod = sidelobe_width_aod
        self.nlos_power_thresh_db = nlos_power_thresh_db
        self.nlos_min_angle_sep = nlos_min_angle_sep
        
        self.los_path = None
        self.sidelobe_paths = []
        self.nlos_paths = []
        
    def classify_paths(self):
        """
        执行路径分类
        返回分类后的DataFrame
        """
        if self.paths_df.empty:
            print("警告：没有路径可供分类")
            return self.paths_df
        
        print("\n" + "="*70)
        print("开始高级路径分类")
        print("="*70)
        
        # 步骤1：识别唯一LoS径（功率最强）
        self._identify_los_path()
        
        # 步骤2：识别并标记旁瓣效应路径
        self._identify_sidelobe_paths()
        
        # 步骤3：识别并去重NLoS路径
        self._identify_nlos_paths()
        
        # 步骤4：打印分类结果
        self._print_classification_results()
        
        return self.paths_df
    
    def _identify_los_path(self):
        """识别唯一的LoS径（功率最强点）"""
        max_idx = self.paths_df['Power'].idxmax()
        self.los_path = self.paths_df.loc[max_idx].copy()
        
        self.paths_df.at[max_idx, 'PathType'] = 'LoS'
        
        print(f"\n【LoS径识别】")
        print(f"  坐标: AoD={self.los_path['AoD']:.2f}°, AoA={self.los_path['AoA']:.2f}°")
        print(f"  功率: {self.los_path['Power']:.4f}")
        print(f"  → 将此路径作为唯一LoS径")
    
    def _identify_sidelobe_paths(self):
        """
        识别旁瓣效应路径
        旁瓣判定条件：
        1. 与LoS径共享AoD或AoA（在旁瓣宽度范围内）
        2. 功率较强但不是最强
        """
        print(f"\n【旁瓣效应识别】")
        print(f"  旁瓣范围: AoD ±{self.sidelobe_width_aod}°, AoA ±{self.sidelobe_width_aoa}°")
        
        los_aod = self.los_path['AoD']
        los_aoa = self.los_path['AoA']
        
        sidelobe_count = 0
        
        for idx, path in self.paths_df.iterrows():
            if path['PathType'] == 'LoS':
                continue
            
            # 计算角度差异
            diff_aod = abs(path['AoD'] - los_aod)
            diff_aoa = abs(path['AoA'] - los_aoa)
            
            # 旁瓣判定逻辑：
            # 情况1: AoD相近（在旁瓣范围内），AoA任意
            # 情况2: AoA相近（在旁瓣范围内），AoD任意
            is_aod_sidelobe = diff_aod <= self.sidelobe_width_aod and diff_aoa > self.sidelobe_width_aoa
            is_aoa_sidelobe = diff_aoa <= self.sidelobe_width_aoa and diff_aod > self.sidelobe_width_aod
            
            # 额外条件：两个维度都在旁瓣范围内但不是LoS本身
            is_near_los = diff_aod <= self.sidelobe_width_aod and diff_aoa <= self.sidelobe_width_aoa
            
            if is_aod_sidelobe or is_aoa_sidelobe or is_near_los:
                self.paths_df.at[idx, 'PathType'] = 'Sidelobe'
                self.sidelobe_paths.append(path)
                sidelobe_count += 1
                print(f"  路径 {idx}: AoD={path['AoD']:.1f}°, AoA={path['AoA']:.1f}° "
                      f"  功率: {path['Power']:.4f} "
                      f"(Δ AoD={diff_aod:.1f}°, Δ AoA={diff_aoa:.1f}°) → 旁瓣")
        
        print(f"  共识别 {sidelobe_count} 条旁瓣路径")
    
    def _identify_nlos_paths(self):
        """
        识别NLoS路径并去重
        NLoS判定条件（同时满足）：
        1. 功率条件：比LoS径低至少X dB
        2. 角度条件：不在LoS径的旁瓣范围内
        3. 去重条件：不同NLoS路径之间应有足够角度分离
        """
        print(f"\n【NLoS径识别与去重】")
        print(f"  功率阈值: {self.nlos_power_thresh_db} dB")
        print(f"  角度分离: {self.nlos_min_angle_sep}°")
        
        los_power = self.los_path['Power']
        los_aod = self.los_path['AoD']
        los_aoa = self.los_path['AoA']
        
        # 候选NLoS路径（未分类的路径）
        candidate_paths = self.paths_df[self.paths_df['PathType'] == 'Unknown'].copy()
        
        if candidate_paths.empty:
            print("  没有候选NLoS路径")
            return
        
        # 按功率排序（从高到低）
        candidate_paths = candidate_paths.sort_values('Power', ascending=False)
        
        nlos_count = 0
        rejected_count = 0
        
        for idx, path in candidate_paths.iterrows():
            # 条件1：功率检查
            if path['Power'] <= 0 or los_power <= 0:
                power_ratio_db = -100
            else:
                power_ratio_db = 10 * np.log10(path['Power'] / los_power)
            
            is_weak_enough = power_ratio_db < -self.nlos_power_thresh_db
            
            # 条件2：角度检查（不在旁瓣范围内）
            diff_aod = abs(path['AoD'] - los_aod)
            diff_aoa = abs(path['AoA'] - los_aoa)
            
            # 必须同时远离LoS的AoD和AoA
            is_outside_sidelobe = (diff_aod > self.sidelobe_width_aod and 
                                   diff_aoa > self.sidelobe_width_aoa)
            
            # 条件3：与已有NLoS路径的角度分离检查
            is_well_separated = True
            for nlos_path in self.nlos_paths:
                nlos_diff_aod = abs(path['AoD'] - nlos_path['AoD'])
                nlos_diff_aoa = abs(path['AoA'] - nlos_path['AoA'])
                
                # 使用欧氏距离判断
                angle_distance = np.sqrt(nlos_diff_aod**2 + nlos_diff_aoa**2)
                
                if angle_distance < self.nlos_min_angle_sep:
                    is_well_separated = False
                    break
            
            # 综合判定
            if is_weak_enough and is_outside_sidelobe and is_well_separated:
                self.paths_df.at[idx, 'PathType'] = 'NLoS'
                self.nlos_paths.append(path)
                nlos_count += 1
                print(f"  路径 {idx}: AoD={path['AoD']:.1f}°, AoA={path['AoA']:.1f}° "
                      f"(功率={power_ratio_db:.1f}dB, "
                      f"Δ={diff_aod:.1f}°/{diff_aoa:.1f}°) → NLoS ✓")
            else:
                # 标记为噪声/无效
                self.paths_df.at[idx, 'PathType'] = 'Noise'
                rejected_count += 1
                
                # 详细拒绝原因
                reasons = []
                if not is_weak_enough:
                    reasons.append(f"功率过强({power_ratio_db:.1f}dB)")
                if not is_outside_sidelobe:
                    reasons.append("在旁瓣范围内")
                if not is_well_separated:
                    reasons.append("与已有NLoS过近")
                
                print(f"  路径 {idx}: AoD={path['AoD']:.1f}°, AoA={path['AoA']:.1f}° "
                      f"→ 噪声 ✗ ({', '.join(reasons)})")
        
        print(f"  共识别 {nlos_count} 条有效NLoS路径")
        print(f"  拒绝 {rejected_count} 条噪声路径")
    
    def _print_classification_results(self):
        """打印分类统计结果"""
        print("\n" + "="*70)
        print("路径分类汇总")
        print("="*70)
        
        type_counts = self.paths_df['PathType'].value_counts()
        
        print(f"\n总路径数: {len(self.paths_df)}")
        for path_type in ['LoS', 'Sidelobe', 'NLoS', 'Noise']:
            count = type_counts.get(path_type, 0)
            print(f"  {path_type:10s}: {count:3d} 条")
        
        print("\n" + "="*70)

# ==========================================
# 模块4：高级可视化
# ==========================================
def advanced_plot(paths_df, estimator, processor, classifier,
                  save_plot=True, output_suffix='_advanced',
                  show_sidelobe=True, show_noise=False):
    """
    高级可视化：区分LoS、旁瓣、NLoS
    """
    
    # 1. 生成背景热力图
    try:
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(processor.bs_angles), max(processor.bs_angles), 100),
            np.linspace(min(processor.ue_angles), max(processor.ue_angles), 100)
        )
        
        bs_mesh, ue_mesh = np.meshgrid(processor.bs_angles, processor.ue_angles)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rbf = Rbf(bs_mesh.flatten(), ue_mesh.flatten(),
                     processor.rss_matrix.flatten(),
                     function='linear', smooth=0.1)
            heatmap = rbf(grid_x, grid_y)
    except Exception as e:
        print(f"插值警告: {e}")
        heatmap = np.zeros_like(grid_x)
    
    # 2. 创建图形
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # 绘制背景
    contour = ax.contourf(grid_x, grid_y, heatmap, levels=50, cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax, label='插值RSS功率')
    
    # 3. 分离不同类型的路径
    los_paths = paths_df[paths_df['PathType'] == 'LoS']
    sidelobe_paths = paths_df[paths_df['PathType'] == 'Sidelobe']
    nlos_paths = paths_df[paths_df['PathType'] == 'NLoS']
    noise_paths = paths_df[paths_df['PathType'] == 'Noise']
    
    # 4. 绘制LoS旁瓣范围（矩形框）
    if not los_paths.empty and show_sidelobe:
        los_aod = los_paths.iloc[0]['AoD']
        los_aoa = los_paths.iloc[0]['AoA']
        
        # AoD方向的旁瓣区域
        aod_range = classifier.sidelobe_width_aod
        aoa_range = classifier.sidelobe_width_aoa
        
        # 绘制旁瓣范围框
        from matplotlib.patches import Rectangle
        
        # 横向旁瓣区域 (AoD固定，AoA变化)
        rect1 = Rectangle((los_aod - aod_range, min(processor.ue_angles)),
                         2 * aod_range,
                         max(processor.ue_angles) - min(processor.ue_angles),
                         linewidth=2, edgecolor='orange', facecolor='none',
                         linestyle='--', alpha=0.5, label='AoD旁瓣区域')
        ax.add_patch(rect1)
        
        # 纵向旁瓣区域 (AoA固定，AoD变化)
        rect2 = Rectangle((min(processor.bs_angles), los_aoa - aoa_range),
                         max(processor.bs_angles) - min(processor.bs_angles),
                         2 * aoa_range,
                         linewidth=2, edgecolor='cyan', facecolor='none',
                         linestyle='--', alpha=0.5, label='AoA旁瓣区域')
        ax.add_patch(rect2)
    
    # 5. 绘制路径点
    # LoS径 - 红色星形，最大
    if not los_paths.empty:
        ax.scatter(los_paths['AoD'], los_paths['AoA'],
                  c='red', marker='*', s=500, edgecolors='yellow',
                  linewidth=3, label='LoS径 (唯一)', zorder=10)
        
        for _, row in los_paths.iterrows():
            label = f"LoS\n({row['AoD']:.1f}°, {row['AoA']:.1f}°)"
            ax.text(row['AoD'] + 1, row['AoA'] + 1.5, label,
                   color='white', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='red', alpha=0.8),
                   zorder=11)
    
    # 旁瓣路径 - 橙色圆形
    if show_sidelobe and not sidelobe_paths.empty:
        ax.scatter(sidelobe_paths['AoD'], sidelobe_paths['AoA'],
                  c='orange', marker='o', s=150, edgecolors='white',
                  linewidth=2, label='旁瓣效应', alpha=0.7, zorder=8)
    
    # NLoS径 - 绿色菱形
    if not nlos_paths.empty:
        ax.scatter(nlos_paths['AoD'], nlos_paths['AoA'],
                  c='lime', marker='D', s=200, edgecolors='darkgreen',
                  linewidth=2.5, label='NLoS径', zorder=9)
        
        # 标注NLoS路径
        for i, (_, row) in enumerate(nlos_paths.iterrows()):
            label = f"NLoS{i+1}\n({row['AoD']:.1f}°, {row['AoA']:.1f}°)"
            ax.text(row['AoD'] + 1, row['AoA'] - 1.5, label,
                   color='white', fontweight='bold', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='green', alpha=0.7),
                   zorder=9)
    
    # 噪声路径 - 白色叉号（可选）
    if show_noise and not noise_paths.empty:
        ax.scatter(noise_paths['AoD'], noise_paths['AoA'],
                  c='white', marker='x', s=80, linewidths=2,
                  label='噪声/无效', alpha=0.5, zorder=5)
    
    # 6. 设置图表属性
    ax.set_xlabel('出发角 (AoD) [度]', fontsize=13, fontweight='bold')
    ax.set_ylabel('到达角 (AoA) [度]', fontsize=13, fontweight='bold')
    
    title = f'高级多径分类\n'
    title += f'(功率阈值: {classifier.nlos_power_thresh_db}dB, '
    title += f'旁瓣范围: ±{classifier.sidelobe_width_aod}°, '
    title += f'NLoS分离: {classifier.nlos_min_angle_sep}°)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 7. 保存
    saved_path = None
    if save_plot:
        output_path = processor.get_output_path(suffix=output_suffix)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f"\n高级热力图已保存至: {output_path}")
    
    plt.show()
    
    return saved_path

# ==========================================
# 主执行流程
# ==========================================
def main():
    """主执行函数"""
    # 配置文件路径
    angle_file = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    rss_file = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115200_filtered.xlsx'
    output_dir = 'D:\\桌面\\SLAMPro\\pic_v1-5'
    
    try:
        print("=" * 70)
        print("高级多径估计与分类系统")
        print("=" * 70)
        
        # 1. 数据预处理
        print("\n步骤 1/4: 数据预处理...")
        processor = BeamDataProcessor(angle_file, rss_file, output_dir)
        rss_mat, ue_ang, bs_ang = processor.pivot_data()
        
        # 2. 构建估计器
        print("\n步骤 2/4: 构建多径估计器...")
        estimator = MultipathEstimator(ue_ang, bs_ang, rss_mat)
        estimator.construct_dictionary(grid_res=0.1, beam_width=1.4)
        
        # 3. 估计多径
        print("\n步骤 3/4: 执行多径估计...")
        paths = estimator.estimate_paths_nn_omp(max_paths=30, min_power_ratio=0.005)
        
        if paths.empty:
            print("\n警告：未检测到有效路径")
            return
        
        # 4. 高级路径分类
        print("\n步骤 4/4: 高级路径分类...")
        classifier = PathClassifier(
            paths,
            sidelobe_width_aoa=5.0,      # AoA旁瓣宽度
            sidelobe_width_aod=5.0,      # AoD旁瓣宽度
            nlos_power_thresh_db=0.05,       # NLoS功率阈值
            nlos_min_angle_sep=20.0       # NLoS最小角度分离
        )
        
        classified_paths = classifier.classify_paths()
        
        # 5. 高级可视化
        print("\n生成高级可视化...")
        saved_path = advanced_plot(
            classified_paths, estimator, processor, classifier,
            save_plot=True,
            show_sidelobe=True,
            show_noise=False
        )
        
        # 6. 输出最终统计
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"输出路径: {saved_path}")
        print(f"\n路径统计:")
        print(f"  LoS径:    {sum(classified_paths['PathType'] == 'LoS')} 条 (唯一)")
        print(f"  旁瓣:     {sum(classified_paths['PathType'] == 'Sidelobe')} 条")
        print(f"  NLoS径:   {sum(classified_paths['PathType'] == 'NLoS')} 条 (去重后)")
        print(f"  噪声:     {sum(classified_paths['PathType'] == 'Noise')} 条")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误：处理过程中发生异常 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()