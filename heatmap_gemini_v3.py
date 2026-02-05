import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.interpolate import Rbf
import os
import re
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

class BeamDataProcessor:
    """
    数据处理类:负责加载Excel/CSV数据,清洗并转换为矩阵格式。
    """
    def __init__(self, angle_file, rss_data_file):
        self.angle_file = angle_file
        self.rss_data_file = rss_data_file
        self.angle_map = self._load_angle_map()
        self.rss_data, self.timestamp = self._load_rss_data()

    def _load_angle_map(self):
        """加载波束ID到角度的映射表 (beam_angle.xlsx)"""
        try:
            if self.angle_file.endswith('.csv'):
                df = pd.read_csv(self.angle_file)
            else:
                df = pd.read_excel(self.angle_file, header=None)
            
            # 假设第0列是BeamID,第1列是角度(Angle)
            # 转换为字典: {beam_id: angle}
            # 修正: 检查第一行是否为字符串类型的表头
            if isinstance(df.iloc[0, 0], str):  # 如果第一行第一列是字符串(表头)
                df = df.iloc[1:]
            
            # 确保数据是数值型
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        except Exception as e:
            raise ValueError(f"加载角度映射文件失败: {e}")

    def _load_rss_data(self):
        """加载波束测量数据"""
        try:
            if self.rss_data_file.endswith('.csv'):
                df = pd.read_csv(self.rss_data_file)
            else:
                df = pd.read_excel(self.rss_data_file)
            
            # 修正: 从文件名提取时间戳(格式: 2026-01-27 113221)
            # 匹配 YYYY-MM-DD HHMMSS 或 YYYY-MM-DD_HHMMSS 格式
            match = re.search(r'(\d{4}-\d{2}-\d{2}[\s_]\d{6})', self.rss_data_file)
            if match:
                timestamp = match.group(1).replace('_', ' ')  # 统一使用空格
            else:
                timestamp = "unknown_timestamp"
            
            return df, timestamp
        except Exception as e:
            raise ValueError(f"加载RSS数据文件失败: {e}")

    def pivot_data(self):
        """
        数据透视:将长格式日志转换为矩阵格式 (Row=UE_Beam, Col=BS_Beam)。
        并映射为物理角度。
        """
        # 1. 聚合:针对同一波束对的多次测量取平均值
        if 'UE_Beam' not in self.rss_data.columns or 'BS_Beam' not in self.rss_data.columns:
             raise ValueError("数据文件中未找到 'UE_Beam' 或 'BS_Beam' 列")
        
        # 修正: 添加groupby的参数
        df_agg = self.rss_data.groupby(['UE_Beam', 'BS_Beam'])['RSS'].mean().reset_index()
        
        # 2. 透视:构造二维矩阵
        rss_matrix_df = df_agg.pivot(index='UE_Beam', columns='BS_Beam', values='RSS')
        
        # 3. 填充缺失值:使用观测到的最小值作为底噪
        # 修正: 使用RSS列的最小值
        min_rss = df_agg['RSS'].min()
        rss_matrix_df = rss_matrix_df.fillna(min_rss)
        
        # 4. 角度映射与过滤
        # 仅保留在 angle_map 中存在的波束ID
        valid_ue_beams = [b for b in rss_matrix_df.index if b in self.angle_map]
        valid_bs_beams = [b for b in rss_matrix_df.columns if b in self.angle_map]
        
        rss_matrix_df = rss_matrix_df.loc[valid_ue_beams, valid_bs_beams]
        
        # 提取对应的物理角度数组
        ue_angles = np.array([self.angle_map[b] for b in rss_matrix_df.index]) # AoA
        bs_angles = np.array([self.angle_map[b] for b in rss_matrix_df.columns]) # AoD
        
        return rss_matrix_df.values, ue_angles, bs_angles

class SpatialMaskingEstimator:
    """
    SM-SIC 估算器:实现基于空间掩膜的连续干扰消除算法。
    """
    def __init__(self, beam_width_deg=1.4):
        self.beam_width = beam_width_deg
        # 高斯波束模型参数: sigma = FWHM / 2.355
        self.sigma = beam_width_deg / 2.355 

    def _gaussian_beam(self, query_angles, center_angles):
        """构建高斯波束响应矩阵"""
        Q, C = np.meshgrid(query_angles, center_angles)
        diff = Q - C
        # 简单的线性阵列响应近似
        return np.exp(- (diff ** 2) / (2 * self.sigma ** 2))

    def construct_dictionary(self, ue_angles, bs_angles, grid_res=0.1):
        """
        构建超分辨率字典。
        ue_angles: 实际测量的UE波束角度
        bs_angles: 实际测量的BS波束角度
        grid_res: 搜索网格的分辨率(度)
        """
        # 定义搜索网格(覆盖测量范围)
        self.aoa_grid = np.arange(np.min(ue_angles), np.max(ue_angles) + grid_res, grid_res)
        self.aod_grid = np.arange(np.min(bs_angles), np.max(bs_angles) + grid_res, grid_res)
        
        # 字典矩阵:描述每个物理波束如何响应网格上的每个角度
        # Phi_RX: (测量点数 x 网格点数)
        self.Phi_RX = self._gaussian_beam(self.aoa_grid, ue_angles)
        self.Phi_TX = self._gaussian_beam(self.aod_grid, bs_angles)
        
        return self.aoa_grid, self.aod_grid

    def estimate_paths_sm_sic(self, rss_matrix, max_paths=3, 
                              proximity_mask_radius=2.0, cross_mask_width=5.0):
        """
        核心算法:SM-SIC (空间掩膜连续干扰消除)
        
        参数:
            rss_matrix: 测量得到的RSS矩阵
            max_paths: 最大提取路径数
            proximity_mask_radius: LoS周围的圆形屏蔽半径(度)
            cross_mask_width: 十字屏蔽带的宽度(度)
        """
        # 使用相关性矩阵作为搜索平面
        # Correlation = Phi_RX.T * RSS_Matrix * Phi_TX
        # 这相当于在超分辨率网格上进行匹配滤波
        correlation_matrix = self.Phi_RX.T @ rss_matrix @ self.Phi_TX
        
        # 修正: 正确初始化列表
        estimated_paths = []
        
        # 初始化空间掩膜 (全1表示所有区域均可搜索)
        spatial_mask = np.ones((len(self.aoa_grid), len(self.aod_grid)))
        
        for k in range(max_paths):
            # 1. 应用空间掩膜
            masked_corr = correlation_matrix * spatial_mask
            
            # 2. 搜索最大值 (Peak Picking)
            idx_flat = np.argmax(masked_corr)
            idx_aoa, idx_aod = np.unravel_index(idx_flat, masked_corr.shape)
            
            peak_val = masked_corr[idx_aoa, idx_aod]
            aoa_est = self.aoa_grid[idx_aoa]
            aod_est = self.aod_grid[idx_aod]
            
            # 修正: 访问列表中第一个元素的字典
            # 停止条件:如果峰值过小(例如小于LoS的10%),则停止
            if k > 0 and len(estimated_paths) > 0 and peak_val < 0.1 * estimated_paths[0]['metric']:
                print(f"路径 {k+1} 信号过弱,停止搜索。")
                break
            
            path_type = "LoS" if k == 0 else "NLoS"
            
            # 3. 记录路径
            estimated_paths.append({
                'id': k+1,
                'type': path_type,
                'aoa': aoa_est,
                'aod': aod_est,
                'metric': peak_val
            })
            
            # 4. 更新掩膜 (这是满足用户核心约束的关键步骤)
            if path_type == "LoS":
                print(f"检测到 LoS 位于: AoD={aod_est:.1f}°, AoA={aoa_est:.1f}°")
                print(f"正在应用十字掩膜规避旁瓣干扰 (Proximity={proximity_mask_radius}°, CrossWidth={cross_mask_width}°)...")
                
                # 创建网格坐标矩阵
                AOA_G, AOD_G = np.meshgrid(self.aoa_grid, self.aod_grid, indexing='ij')
                
                # 条件1: 邻近区域 (圆形)
                dist_sq = (AOA_G - aoa_est)**2 + (AOD_G - aod_est)**2
                mask_prox = dist_sq > proximity_mask_radius**2
                
                # 条件2: 十字区域 (带状)
                # 屏蔽 AoD = LoS_AoD +/- width/2
                mask_cross_aod = np.abs(AOD_G - aod_est) > (cross_mask_width / 2)
                # 屏蔽 AoA = LoS_AoA +/- width/2
                mask_cross_aoa = np.abs(AOA_G - aoa_est) > (cross_mask_width / 2)
                
                # 更新全局掩膜
                spatial_mask *= mask_prox
                spatial_mask *= mask_cross_aod # 剔除 AoD 轴
                spatial_mask *= mask_cross_aoa # 剔除 AoA 轴
                
            # 如果是 NLoS,通常只需要应用小范围的邻近掩膜,防止重复检测同一点
            else:
                AOA_G, AOD_G = np.meshgrid(self.aoa_grid, self.aod_grid, indexing='ij')
                dist_sq = (AOA_G - aoa_est)**2 + (AOD_G - aod_est)**2
                # NLoS 掩膜半径可以小一点,例如1度
                mask_local = dist_sq > (1.0)**2
                spatial_mask *= mask_local

        return pd.DataFrame(estimated_paths)

def visualize_results(rss_matrix, ue_angles, bs_angles, paths_df, output_path):
    """
    可视化:绘制热力图并标注LoS/NLoS路径及规避区域。
    """
    # 1. 插值平滑热力图 (为了美观)
    # 创建网格
    grid_res = 100j
    grid_aod, grid_aoa = np.mgrid[min(bs_angles):max(bs_angles):grid_res, 
                                  min(ue_angles):max(ue_angles):grid_res]
    
    # 原始数据坐标
    raw_aod, raw_aoa = np.meshgrid(bs_angles, ue_angles)
    
    # RBF 插值
    try:
        rbf = Rbf(raw_aod.ravel(), raw_aoa.ravel(), rss_matrix.ravel(), function='linear')
        z_grid = rbf(grid_aod, grid_aoa)
    except:
        # 如果数据点太少RBF失败,直接使用原始数据绘图
        z_grid = rss_matrix
        grid_aod = raw_aod
        grid_aoa = raw_aoa

    plt.figure(figsize=(12, 9))
    
    # 绘制等高线热力图
    cp = plt.contourf(grid_aod, grid_aoa, z_grid, 100, cmap='viridis')
    plt.colorbar(cp, label='Received Signal Strength (RSS)')
    
    # 绘制路径
    if not paths_df.empty:
        # 提取LoS用于画辅助线
        los_paths = paths_df[paths_df['type'] == 'LoS']
        
        for _, row in paths_df.iterrows():
            if row['type'] == 'LoS':
                # LoS: 红色大圆圈
                plt.scatter(row['aod'], row['aoa'], s=200, c='red', marker='o', 
                           edgecolors='white', linewidth=2, label=f"LoS (Primary)", zorder=10)
                # 绘制十字掩膜的示意线(用户要求的规避区域)
                plt.axvline(x=row['aod'], color='red', linestyle='--', alpha=0.5, label='Sidelobe Ridge (AoD)')
                plt.axhline(y=row['aoa'], color='red', linestyle='--', alpha=0.5, label='Sidelobe Ridge (AoA)')
            else:
                # NLoS: 白色X
                plt.scatter(row['aod'], row['aoa'], s=150, c='white', marker='x', 
                           linewidth=3, label=f"NLoS (Detected)", zorder=10)
                # 标注坐标
                plt.text(row['aod']+1, row['aoa']+1, f"NLoS\n({row['aod']:.1f}, {row['aoa']:.1f})", 
                         color='white', fontsize=9, fontweight='bold')

    plt.xlabel('Angle of Departure (AoD) [deg]', fontsize=12)
    plt.ylabel('Angle of Arrival (AoA) [deg]', fontsize=12)
    plt.title('mmWave Beamspace Heatmap & SM-SIC Path Identification', fontsize=14)
    
    # 去除重复的图例标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', 
               frameon=True, facecolor='black', framealpha=0.6, labelcolor='white')
    plt.grid(True, alpha=0.3)
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"结果图已保存至: {output_path}")
    plt.show()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置输入文件路径(请修改为您实际的文件路径)
    ANGLE_FILE = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx' 
    DATA_FILE = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115303_filtered.xlsx'
    OUTPUT_DIR = 'D:\\桌面\\SLAMPro\\pic_v3'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("--- 开始处理毫米波波束数据 ---")
    
    # 1. 检查文件是否存在
    if not os.path.exists(ANGLE_FILE) or not os.path.exists(DATA_FILE):
        print(f"错误: 找不到输入文件。请确保文件存在。")
        print(f"角度文件: {ANGLE_FILE}")
        print(f"数据文件: {DATA_FILE}")
    else:
        try:
            # 2. 加载与预处理
            processor = BeamDataProcessor(ANGLE_FILE, DATA_FILE)
            rss_mat, ue_ang, bs_ang = processor.pivot_data()
            print(f"数据加载成功: 矩阵维度 {rss_mat.shape}, UE角度范围[{min(ue_ang):.1f}, {max(ue_ang):.1f}]")
            
            # 3. 初始化算法模块
            # beam_width_deg 设定为10度,模拟典型阵列的主瓣宽度
            estimator = SpatialMaskingEstimator(beam_width_deg=10.0)
            
            # 4. 构建字典
            print("构建超分辨率字典...")
            estimator.construct_dictionary(ue_ang, bs_ang, grid_res=0.5)
            
            # 5. 执行 SM-SIC 算法
            print("执行 SM-SIC 算法识别路径...")
            paths = estimator.estimate_paths_sm_sic(rss_mat, max_paths=3, 
                                                  proximity_mask_radius=2.0, 
                                                  cross_mask_width=5.0)
            
            print("\n识别结果:")
            print(paths[['id', 'type', 'aod', 'aoa', 'metric']])
            
            # 6. 可视化并保存
            # 修正: 使用与输入文件名匹配的时间戳格式
            # 从 "Serial Debug 2026-01-27 113221_filtered.xlsx" 提取 "2026-01-27 113221"
            out_name = f"{processor.timestamp}.png"
            output_path = os.path.join(OUTPUT_DIR, out_name)
            
            print(f"\n准备保存图像到: {output_path}")
            visualize_results(rss_mat, ue_ang, bs_ang, paths, output_path)
            
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()