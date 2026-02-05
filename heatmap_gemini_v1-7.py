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
        """将长格式日志转换为矩阵格式（包含对数变换）"""
        if self.rss_data.empty:
            raise ValueError("RSS数据为空，无法进行数据透视")
        
        # ==========================================================
        # 修改点：在处理前对所有 RSS 值进行对数变换
        # ==========================================================
        print("\n[预处理] 正在对RSS数据进行对数变换 (np.log)...")
        # 1. 过滤掉非正值，防止对数计算错误 (log(<=0) 无定义)
        original_count = len(self.rss_data)
        self.rss_data = self.rss_data[self.rss_data['RSS'] > 0].copy()
        filtered_count = len(self.rss_data)
        if original_count != filtered_count:
            print(f"  警告：已过滤 {original_count - filtered_count} 条 RSS<=0 的记录")
        
        # 2. 执行对数变换 (这里使用自然对数，如需以10为底请改为 np.log10)
        self.rss_data['RSS'] = np.log(self.rss_data['RSS'])
        print("  对数变换完成，后续所有步骤将基于 Log(RSS) 进行。")
        # ==========================================================

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
                # 注意：如果输入数据经过对数变换后为负值，NNLS 可能无法正常工作，
                # 因为非负组合无法生成负数。前提假设是对数变换后的值仍为正值。
                coeffs, rnorm = nnls(Dict_Active, self.rss_vector)
            except Exception as e:
                print(f"迭代 {k}: NNLS求解失败 - {e}")
                break
            
            reconstructed = Dict_Active @ coeffs
            residual = self.rss_vector - reconstructed
            
            residual_norm = np.linalg.norm(residual)
            print(f"迭代 {k}: AoA={self.aoa_grid[idx_aoa]:.1f}°, AoD={self.aod_grid[idx_aod]:.1f}°, "
                  f"系数={coeffs[-1]:.4f}, 残差={residual_norm:.4f}")
        
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
                        'PathType': 'Unknown'
                    })
            
            print(f"估计完成，保留 {len(path_params)} 条有效路径")
            return pd.DataFrame(path_params)
        else:
            print("警告：未找到有效路径")
            return pd.DataFrame(columns=['AoA', 'AoD', 'Power', 'PathType'])

# ==========================================
# 模块3：改进的路径分类器
# ==========================================
class AdvancedPathClassifier:
    """
    改进的路径分类器：
    核心改进：旁瓣区域内的强信号可以是NLoS，而不是简单地标记为旁瓣干扰
    
    分类逻辑：
    1. LoS: 功率最强的路径（唯一）
    2. NLoS: 满足功率阈值和角度分离的路径（可在旁瓣区域内）
    3. 旁瓣干扰: 在旁瓣区域内但不满足NLoS条件的弱信号
    4. 噪声: 其他无效信号
    """
    def __init__(self, paths_df,
                 sidelobe_width_aoa=45.0,
                 sidelobe_width_aod=45.0,
                 nlos_power_thresh_db=10.0,
                 nlos_angle_separation=20.0,
                 sidelobe_power_ratio_db=15.0):
        """
        初始化路径分类器
        :param paths_df: 初步估计的路径DataFrame
        :param sidelobe_width_aoa: AoA方向旁瓣宽度范围（度）
        :param sidelobe_width_aod: AoD方向旁瓣宽度范围（度）
        :param nlos_power_thresh_db: NLoS判定的最小功率阈值（相对LoS，dB）
        :param nlos_angle_separation: NLoS路径之间的最小角度分离（度）
        :param sidelobe_power_ratio_db: 旁瓣干扰的功率阈值（相对LoS，dB）
                                        比这个值弱的才判定为纯旁瓣干扰
        """
        self.paths_df = paths_df.copy()
        self.sidelobe_width_aoa = sidelobe_width_aoa
        self.sidelobe_width_aod = sidelobe_width_aod
        self.nlos_power_thresh_db = nlos_power_thresh_db
        self.nlos_angle_separation = nlos_angle_separation
        self.sidelobe_power_ratio_db = sidelobe_power_ratio_db
        
        self.los_path = None
        self.nlos_paths = []
        self.sidelobe_paths = []
        
    def classify_paths(self):
        """
        执行路径分类
        返回分类后的DataFrame
        """
        if self.paths_df.empty:
            print("警告：没有路径可供分类")
            return self.paths_df
        
        print("\n" + "="*80)
        print("开始改进的路径分类（旁瓣区域可包含NLoS）")
        print("="*80)
        
        # 步骤1：识别唯一LoS径（功率最强）
        self._identify_los_path()
        
        # 步骤2：识别NLoS径（关键：可以在旁瓣区域内）
        self._identify_nlos_paths()
        
        # 步骤3：剩余路径中识别旁瓣干扰
        self._identify_sidelobe_interference()
        
        # 步骤4：标记噪声
        self._mark_noise()
        
        # 步骤5：打印分类结果
        self._print_classification_results()
        
        return self.paths_df
    
    def _identify_los_path(self):
        """识别唯一的LoS径（功率最强点）"""
        max_idx = self.paths_df['Power'].idxmax()
        self.los_path = self.paths_df.loc[max_idx].copy()
        
        self.paths_df.at[max_idx, 'PathType'] = 'LoS'
        
        print(f"\n【步骤1：LoS径识别】")
        print(f"  坐标: AoD={self.los_path['AoD']:.2f}°, AoA={self.los_path['AoA']:.2f}°")
        print(f"  系数(对数域后): {self.los_path['Power']:.4f}")
        print(f"  ✓ 唯一LoS径已确定")
    
    def _identify_nlos_paths(self):
        """
        识别NLoS径 - 关键改进点
        """
        print(f"\n【步骤2：NLoS径识别】")
        print(f"  NLoS阈值: -{self.nlos_power_thresh_db} dB (相对LoS)")
        print(f"  NLoS角度分离: {self.nlos_angle_separation}°")
        
        los_power = self.los_path['Power']
        los_aod = self.los_path['AoD']
        los_aoa = self.los_path['AoA']
        
        # 候选路径：所有未分类的路径
        candidate_paths = self.paths_df[self.paths_df['PathType'] == 'Unknown'].copy()
        
        if candidate_paths.empty:
            print("  没有候选路径")
            return
        
        # 按功率排序（从高到低）
        candidate_paths = candidate_paths.sort_values('Power', ascending=False)
        
        nlos_count = 0
        
        for idx, path in candidate_paths.iterrows():
            # 条件1：功率检查
            # 注意：由于输入已经取了对数，这里的 Power 实际上已经是 Log 值。
            # 如果这里的 Power 代表的是 Log(Amplitude) 或 Log(Power)，
            # 那么 "dB比值" 计算方式可能需要根据实际物理意义调整。
            # 这里保持原逻辑： 10 * log10(P_path / P_los)。
            # 如果 P_path 本身是对数值，这个计算结果物理意义比较复杂，
            # 但作为相对强度的衡量仍然有效。
            if path['Power'] <= 0 or los_power <= 0:
                power_ratio_db = -100
            else:
                power_ratio_db = 10 * np.log10(path['Power'] / los_power)
            
            # NLoS应该比LoS弱，但不能太弱
            is_valid_power = (-self.sidelobe_power_ratio_db < power_ratio_db < -self.nlos_power_thresh_db)
            
            # 条件2：几何条件 - 与LoS的总角度距离
            diff_aod = abs(path['AoD'] - los_aod)
            diff_aoa = abs(path['AoA'] - los_aoa)
            angle_distance_from_los = np.sqrt(diff_aod**2 + diff_aoa**2)
            
            # 关键改进：不再简单地排除旁瓣区域，而是用角度距离判断
            is_geometrically_distinct = angle_distance_from_los > self.nlos_angle_separation
            
            # 条件3：与已识别NLoS的距离检查（去重）
            is_well_separated_from_nlos = True
            min_distance_to_nlos = float('inf')
            
            for nlos_path in self.nlos_paths:
                nlos_diff_aod = abs(path['AoD'] - nlos_path['AoD'])
                nlos_diff_aoa = abs(path['AoA'] - nlos_path['AoA'])
                distance_to_nlos = np.sqrt(nlos_diff_aod**2 + nlos_diff_aoa**2)
                
                min_distance_to_nlos = min(min_distance_to_nlos, distance_to_nlos)
                
                if distance_to_nlos < self.nlos_angle_separation:
                    is_well_separated_from_nlos = False
                    break
            
            # 综合判定：三个条件同时满足
            if is_valid_power and is_geometrically_distinct and is_well_separated_from_nlos:
                self.paths_df.at[idx, 'PathType'] = 'NLoS'
                self.nlos_paths.append(path)
                nlos_count += 1
                
                # 检查是否在旁瓣区域内
                in_sidelobe_region = (diff_aod <= self.sidelobe_width_aod or 
                                     diff_aoa <= self.sidelobe_width_aoa)
                region_note = " [在旁瓣区域内]" if in_sidelobe_region else " [旁瓣区域外]"
                
                print(f"  路径 {idx}: AoD={path['AoD']:.1f}°, AoA={path['AoA']:.1f}°")
                print(f"    相对比值={power_ratio_db:.1f}, "
                      f"与LoS距离={angle_distance_from_los:.1f}°{region_note}")
                print(f"    → NLoS ✓")
            else:
                # 记录拒绝原因
                reasons = []
                if not is_valid_power:
                    reasons.append(f"强度不符({power_ratio_db:.1f})")
                if not is_geometrically_distinct:
                    reasons.append(f"与LoS过近({angle_distance_from_los:.1f}°)")
                if not is_well_separated_from_nlos:
                    reasons.append(f"与已有NLoS过近({min_distance_to_nlos:.1f}°)")
                
                self.paths_df.at[idx, '_reject_reasons'] = ", ".join(reasons)
        
        print(f"  ✓ 共识别 {nlos_count} 条NLoS路径")
    
    def _identify_sidelobe_interference(self):
        """
        识别旁瓣干扰
        """
        print(f"\n【步骤3：旁瓣干扰识别】")
        print(f"  旁瓣区域: AoD ±{self.sidelobe_width_aod}°, AoA ±{self.sidelobe_width_aoa}°")
        print(f"  旁瓣阈值: <-{self.sidelobe_power_ratio_db}")
        
        los_aod = self.los_path['AoD']
        los_aoa = self.los_path['AoA']
        los_power = self.los_path['Power']
        
        # 候选：未分类的路径
        candidate_paths = self.paths_df[self.paths_df['PathType'] == 'Unknown']
        
        sidelobe_count = 0
        
        for idx, path in candidate_paths.iterrows():
            diff_aod = abs(path['AoD'] - los_aod)
            diff_aoa = abs(path['AoA'] - los_aoa)
            
            # 判断是否在旁瓣区域内
            in_aod_sidelobe = diff_aod <= self.sidelobe_width_aod
            in_aoa_sidelobe = diff_aoa <= self.sidelobe_width_aoa
            in_sidelobe_region = in_aod_sidelobe or in_aoa_sidelobe
            
            # 功率检查
            if path['Power'] > 0 and los_power > 0:
                power_ratio_db = 10 * np.log10(path['Power'] / los_power)
            else:
                power_ratio_db = -100
            
            is_weak = power_ratio_db < -self.sidelobe_power_ratio_db
            
            # 旁瓣判定：在旁瓣区域内 且 功率较弱
            if in_sidelobe_region and is_weak:
                self.paths_df.at[idx, 'PathType'] = 'Sidelobe'
                self.sidelobe_paths.append(path)
                sidelobe_count += 1
                
                direction = []
                if in_aod_sidelobe:
                    direction.append("AoD旁瓣")
                if in_aoa_sidelobe:
                    direction.append("AoA旁瓣")
                
                print(f"  路径 {idx}: AoD={path['AoD']:.1f}°, AoA={path['AoA']:.1f}° "
                      f"({'+'.join(direction)}, {power_ratio_db:.1f}) → 旁瓣干扰")
        
        print(f"  ✓ 共识别 {sidelobe_count} 条旁瓣干扰")
    
    def _mark_noise(self):
        """标记剩余路径为噪声"""
        noise_paths = self.paths_df[self.paths_df['PathType'] == 'Unknown']
        noise_count = len(noise_paths)
        
        if noise_count > 0:
            self.paths_df.loc[noise_paths.index, 'PathType'] = 'Noise'
            print(f"\n【步骤4：噪声标记】")
            print(f"  ✓ 标记 {noise_count} 条噪声路径")
    
    def _print_classification_results(self):
        """打印分类统计结果"""
        print("\n" + "="*80)
        print("路径分类汇总")
        print("="*80)
        
        type_counts = self.paths_df['PathType'].value_counts()
        
        print(f"\n总路径数: {len(self.paths_df)}")
        for path_type in ['LoS', 'NLoS', 'Sidelobe', 'Noise']:
            count = type_counts.get(path_type, 0)
            print(f"  {path_type:10s}: {count:3d} 条")
        
        # 详细的NLoS信息
        nlos_df = self.paths_df[self.paths_df['PathType'] == 'NLoS']
        if not nlos_df.empty:
            print(f"\nNLoS路径详细信息:")
            for i, (idx, path) in enumerate(nlos_df.iterrows()):
                diff_aod = abs(path['AoD'] - self.los_path['AoD'])
                diff_aoa = abs(path['AoA'] - self.los_path['AoA'])
                in_sidelobe = (diff_aod <= self.sidelobe_width_aod or 
                              diff_aoa <= self.sidelobe_width_aoa)
                location = "旁瓣区域内" if in_sidelobe else "旁瓣区域外"
                
                power_db = 10 * np.log10(path['Power'] / self.los_path['Power'])
                print(f"  NLoS{i+1}: ({path['AoD']:.1f}°, {path['AoA']:.1f}°) "
                      f"{power_db:.1f} (rel) [{location}]")
        
        print("\n" + "="*80)

# ==========================================
# 模块4：改进的可视化
# ==========================================
def improved_plot(paths_df, estimator, processor, classifier,
                  save_plot=True, output_suffix='_improved',
                  show_sidelobe=True, show_noise=False):
    """
    改进的可视化：清晰展示NLoS可以在旁瓣区域内
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
            # 这里的 rss_matrix 已经是取过对数的值，因此热力图直接展示 Log(RSS) 分布
            rbf = Rbf(bs_mesh.flatten(), ue_mesh.flatten(),
                     processor.rss_matrix.flatten(),
                     function='linear', smooth=0.1)
            heatmap = rbf(grid_x, grid_y)
    except Exception as e:
        print(f"插值警告: {e}")
        heatmap = np.zeros_like(grid_x)
    
    # 2. 创建图形
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # 绘制背景
    contour = ax.contourf(grid_x, grid_y, heatmap, levels=50, cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax, label='Log(RSS) Power Distribution')
    
    # 3. 分离不同类型的路径
    los_paths = paths_df[paths_df['PathType'] == 'LoS']
    nlos_paths = paths_df[paths_df['PathType'] == 'NLoS']
    sidelobe_paths = paths_df[paths_df['PathType'] == 'Sidelobe']
    noise_paths = paths_df[paths_df['PathType'] == 'Noise']
    
    # 5. 绘制路径点
    # LoS径 - 红色星形
    if not los_paths.empty:
        ax.scatter(los_paths['AoD'], los_paths['AoA'],
                  c='red', marker='*', s=600, edgecolors='black',
                  linewidth=2.5, label='LoS径', zorder=9)
        
        for _, row in los_paths.iterrows():
            label = f"LoS\n({row['AoD']:.1f}°, {row['AoA']:.1f}°)"
            ax.text(row['AoD'] + 1.5, row['AoA'] + 2, label,
                   color='white', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='red', alpha=0.85),
                   zorder=11)
    
    # NLoS径 - 绿色菱形
    if not nlos_paths.empty:
        # 统一绘制所有NLoS，不区分旁瓣区域内外
        for i, (_, row) in enumerate(nlos_paths.iterrows()):
            # 统一使用亮绿色菱形
            ax.scatter(row['AoD'], row['AoA'],
                      c='lime', marker='D', s=250, edgecolors='black',
                      linewidth=2.5, zorder=9)
            
            label = f"NLoS{i+1}\n({row['AoD']:.1f}°, {row['AoA']:.1f}°)"
            bbox_color = 'green'
            
            ax.text(row['AoD'] + 1.5, row['AoA'] - 2, label,
                   color='white', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=bbox_color, alpha=0.8),
                   zorder=9)
        
        # 添加统一的图例说明
        ax.scatter([], [], c='lime', marker='D', s=250, edgecolors='darkgreen',
                  linewidth=2.5, label='NLoS径')
    
    # 6. 设置图表属性
    ax.set_xlabel('出发角 (AoD) [度]', fontsize=14, fontweight='bold')
    ax.set_ylabel('到达角 (AoA) [度]', fontsize=14, fontweight='bold')
    
    title = f'mmWave Multipath Heatmap (Log Scale) & Estimation Results\n'
    ax.set_title(title, fontsize=20, fontweight='bold', pad=3)
    
    # ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    # 修改这里：增加图例大小和调整边框
    ax.legend(
        loc='upper right', 
        fontsize=12,  # 增加字体大小
        framealpha=0.95,
        markerscale=0.8,  # 增加标记符号的大小
        handletextpad=0.5,  # 增加标记和文本之间的间距
        borderpad=1.2,  # 增加边框内部的填充
        labelspacing=1.0,  # 增加标签之间的垂直间距
        handlelength=2.0,  # 增加标记符号的长度
        borderaxespad=1.0,  # 增加图例和坐标轴之间的间距
        fancybox=True,  # 使用圆角边框
        shadow=True  # 添加阴影效果
    )
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 7. 保存
    saved_path = None
    if save_plot:
        output_path = processor.get_output_path(suffix=output_suffix)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        saved_path = output_path
        print(f"\n改进的热力图已保存至: {output_path}")
    
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
    output_dir = 'D:\\桌面\\SLAMPro\\pic_v1-7'
    
    try:
        print("=" * 80)
        print("改进的多径估计与分类系统 v2.0 (Log Input Mode)")
        print("核心改进：NLoS可以在旁瓣区域内 + 预处理对数变换")
        print("=" * 80)
        
        # 1. 数据预处理
        print("\n步骤 1/4: 数据预处理...")
        processor = BeamDataProcessor(angle_file, rss_file, output_dir)
        # 注意：pivot_data 现在包含了对数变换逻辑
        rss_mat, ue_ang, bs_ang = processor.pivot_data()
        
        # 2. 构建估计器
        print("\n步骤 2/4: 构建多径估计器...")
        estimator = MultipathEstimator(ue_ang, bs_ang, rss_mat)
        estimator.construct_dictionary(grid_res=0.1, beam_width=1.4)
        
        # 3. 估计多径
        print("\n步骤 3/4: 执行多径估计 (基于 Log 域数据)...")
        # 由于输入值变小（对数后），min_power_ratio 可能需要根据实际数据的数值范围微调
        paths = estimator.estimate_paths_nn_omp(max_paths=20, min_power_ratio=0.0003)
        
        if paths.empty:
            print("\n警告：未检测到有效路径")
            return
        
        # 4. 改进的路径分类
        print("\n步骤 4/4: 改进的路径分类...")
        classifier = AdvancedPathClassifier(
            paths,
            sidelobe_width_aoa=5,        # 旁瓣宽度
            sidelobe_width_aod=5,
            nlos_power_thresh_db=0.01,       # NLoS最小功率差（相对LoS）
            nlos_angle_separation=15,     # NLoS之间的最小角度分离
            sidelobe_power_ratio_db=0.15    # 旁瓣干扰功率阈值
        )
        
        classified_paths = classifier.classify_paths()
        
        # 5. 改进的可视化
        print("\n生成改进的可视化...")
        saved_path = improved_plot(
            classified_paths, estimator, processor, classifier,
            save_plot=True,
            show_sidelobe=True,
            show_noise=False
        )
        
        # 6. 输出最终统计
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"输出路径: {saved_path}")
        print(f"\n路径统计:")
        print(f"  LoS径:    {sum(classified_paths['PathType'] == 'LoS')} 条 (唯一)")
        print(f"  NLoS径:   {sum(classified_paths['PathType'] == 'NLoS')} 条")
        print(f"  旁瓣干扰: {sum(classified_paths['PathType'] == 'Sidelobe')} 条")
        print(f"  噪声:     {sum(classified_paths['PathType'] == 'Noise')} 条")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误：处理过程中发生异常 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()