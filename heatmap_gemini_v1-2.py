import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize
import seaborn as sns
import warnings
import os
import re

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数模块 (Configuration)
# ==========================================
class Config:
    # 文件路径
    ANGLE_FILE = 'D:\\桌面\\SLAMPro\\beam_angle.xlsx'
    RSS_FILE = 'D:\\桌面\\SLAMPro\\debugDoc\\Serial Debug 2026-01-27 115303_filtered.xlsx'
    output_dir = 'D:\\桌面\\SLAMPro\\pic_v1-2'
    
    # 算法参数
    GRID_RES = 0.5       # 搜索网格分辨率 (度), 越小精度越高但越慢
    BEAM_WIDTH = 1.4     # 假设的天线波束宽度 (度), 用于构建字典
    N_PATHS = 5          # 预估的最大路径数量 (稀疏度 K)
    
    # 路径鉴别参数
    LOS_POWER_MARGIN = 0.8  # LoS判定阈值 (相对于最大峰值的比例)
    
    # 可视化参数
    PLOT_CMAP = 'viridis'  # 热力图配色: magma, inferno, jet (对比度高)

# ==========================================
# 2. 数据处理类 (Data Processor)
# ==========================================
class BeamDataManager:
    def __init__(self, angle_path, rss_path):
        self.angle_path = angle_path
        self.rss_path = rss_path
        self.beam_map = {}
        self.rss_df = None
        self.rss_matrix = None
        self.aoa_coords = None
        self.aod_coords = None
    
    def load_data(self):
        """加载并清洗数据"""
        print("[-] Loading data files...")
        
        # 1. 加载角度映射表
        try:
            # 尝试读取Excel文件
            angle_df = pd.read_excel(self.angle_path, header=0)
            # 兼容性处理：如果列名不对，按位置读取
            if 'BeamID' not in angle_df.columns and 'Angle' not in angle_df.columns: 
                angle_df = pd.read_excel(self.angle_path, header=None)
                self.beam_map = dict(zip(angle_df.iloc[:,0], angle_df.iloc[:,1]))
            else:
                # 假设第一列是BeamID，第二列是Angle
                col_names = angle_df.columns.tolist()
                self.beam_map = dict(zip(angle_df.iloc[:,0], angle_df.iloc[:,1]))
        except Exception as e:
            print(f"[!] Error loading angle file: {e}")
            return False

        # 2. 加载RSS测量数据
        try:
            self.rss_df = pd.read_excel(self.rss_path)
            # 简单的列名清洗
            self.rss_df.columns = [c.strip() for c in self.rss_df.columns]
        except Exception as e:
            print(f"[!] Error loading RSS file: {e}")
            return False
            
        return True

    def process_matrix(self):
        """将离散日志转换为 (AoA, AoD) 稀疏矩阵"""
        print("[-] Processing RSS matrix...")
        
        # 假设RSS文件包含 'RxBeamID', 'TxBeamID', 'RSS' 列
        # 根据实际列名调整
        required_cols = ['RxBeamID', 'TxBeamID', 'RSS']
        
        # 检查列名是否存在，如果不存在则尝试猜测
        actual_cols = self.rss_df.columns.tolist()
        if not all(col in actual_cols for col in required_cols):
            print(f"[!] Warning: Expected columns {required_cols}, got {actual_cols}")
            print("[!] Attempting to use first 3 columns as RxBeamID, TxBeamID, RSS")
            if len(actual_cols) >= 3:
                self.rss_df.columns = required_cols + actual_cols[3:]
            else:
                print("[!] Not enough columns in RSS file")
                return pd.DataFrame()
        
        # 聚合重复测量取平均
        df_grouped = self.rss_df.groupby(['RxBeamID', 'TxBeamID'])['RSS'].mean().reset_index()
        
        # 映射波束ID到物理角度
        df_grouped['AoA'] = df_grouped['RxBeamID'].map(self.beam_map)
        df_grouped['AoD'] = df_grouped['TxBeamID'].map(self.beam_map)
        
        # 剔除无效角度
        df_clean = df_grouped.dropna(subset=['AoA', 'AoD'])
        
        return df_clean

# ==========================================
# 3. 核心算法类: 稀疏解卷积 (Sparse Deconvolution)
# ==========================================
class SparseChannelEstimator:
    def __init__(self, data_df):
        self.data = data_df
        self.model = None
        self.reconstructed_map = None
        self.path_list = []
        self.path_df = pd.DataFrame()
        
        # 生成搜索网格
        self.aoa_grid = np.arange(data_df['AoA'].min(), data_df['AoA'].max(), Config.GRID_RES)
        self.aod_grid = np.arange(data_df['AoD'].min(), data_df['AoD'].max(), Config.GRID_RES)
    
    def _gaussian_kernel(self, x, mu, fwhm):
        """高斯波束模型 (模拟天线主瓣)"""
        sigma = fwhm / 2.355
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def fit(self):
        """
        核心步骤：构建字典并运行OMP算法
        原理：y = D * x
        y: 观测到的RSS向量
        D: 字典矩阵，每一列代表如果在(AoA_i, AoD_j)有一条路径，测量系统会产生什么样的响应
        x: 我们要求的稀疏向量 (即真实的路径位置和强度)
        """
        print("[-] Running Sparse Recovery (SR-OMP)...")
        
        # 1. 准备观测向量 y
        y = self.data['RSS'].values
        
        # 2. 构建字典矩阵 D (Sensing Matrix)
        n_samples = len(y)
        n_features = len(self.aoa_grid) * len(self.aod_grid)
        
        print(f"[-] Building dictionary: {n_samples} samples x {n_features} features")
        
        Dictionary = np.zeros((n_samples, n_features))
        
        # 观测到的角度坐标
        meas_aoa = self.data['AoA'].values
        meas_aod = self.data['AoD'].values
        
        col_idx = 0
        grid_coords = []  # 记录每一列对应的网格坐标 (grid_aoa, grid_aod)
        
        for g_aoa in self.aoa_grid:
            for g_aod in self.aod_grid:
                # 模拟：如果路径在 (g_aoa, g_aod)，它在各个测量点产生的响应
                # 响应 = Rx增益 * Tx增益
                resp_rx = self._gaussian_kernel(meas_aoa, g_aoa, Config.BEAM_WIDTH)
                resp_tx = self._gaussian_kernel(meas_aod, g_aod, Config.BEAM_WIDTH)
                
                # 这一列原子(Atom)
                Dictionary[:, col_idx] = resp_rx * resp_tx
                grid_coords.append((g_aoa, g_aod))
                col_idx += 1
                
        # 归一化字典 (OMP要求)
        Dictionary = normalize(Dictionary, axis=0)
        
        # 3. 运行OMP算法求解 x
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=Config.N_PATHS, fit_intercept=False)
        omp.fit(Dictionary, y)
        
        # 4. 解析结果
        coefs = omp.coef_
        
        # 提取非零分量作为路径
        active_indices = np.where(coefs > 0)[0]  # 只要正值，功率不能为负
        
        self.path_list = []
        for idx in active_indices:
            angle_pair = grid_coords[idx]
            power = coefs[idx]
            self.path_list.append({
                'AoA': angle_pair[0],
                'AoD': angle_pair[1],
                'Power': power
            })
            
        # 转换为DataFrame方便后续处理
        self.path_df = pd.DataFrame(self.path_list)
        if not self.path_df.empty:
            # 归一化功率用于显示
            self.path_df['NormPower'] = self.path_df['Power'] / self.path_df['Power'].max()
            
            # 鉴别LoS/NLoS
            max_p = self.path_df['Power'].max()
            self.path_df['Type'] = self.path_df['Power'].apply(
                lambda x: 'LoS' if x >= max_p * Config.LOS_POWER_MARGIN else 'NLoS'
            )
        
        return self.path_df

    def generate_clean_heatmap(self):
        """基于稀疏解生成高对比度的'干净'热力图"""
        # 创建空白网格
        heatmap = np.zeros((len(self.aoa_grid), len(self.aod_grid)))
        
        if self.path_list:
            for _, row in self.path_df.iterrows():
                # 找到最近的网格索引
                idx_aoa = (np.abs(self.aoa_grid - row['AoA'])).argmin()
                idx_aod = (np.abs(self.aod_grid - row['AoD'])).argmin()
                
                # 在该位置放置能量
                heatmap[idx_aoa, idx_aod] = row['Power']
                
            # 为了可视化好看，做一个极小的高斯模糊，让点稍微可见，而不是单像素
            # 但保持极高的对比度
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=1.0) 
            
        return heatmap

# ==========================================
# 4. 可视化对比模块 (Visualization)
# ==========================================
def extract_filename_timestamp(filepath):
    """从文件路径中提取时间戳部分"""
    # 从文件名中提取 "2026-01-27 113221" 格式
    filename = os.path.basename(filepath)
    # 匹配 "Serial Debug YYYY-MM-DD HHMMSS" 模式
    match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{6})', filename)
    if match:
        return match.group(1)
    else:
        # 如果没有匹配，返回文件名（去掉扩展名）
        return os.path.splitext(filename)[0]

def plot_comparison(raw_data, estimator, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # --- 图1: 原始插值热力图 (The "Before") ---
    # 使用griddata进行插值
    grid_x, grid_y = np.meshgrid(estimator.aod_grid, estimator.aoa_grid)
    
    # 原始数据
    points = raw_data[['AoD', 'AoA']].values
    values = raw_data['RSS'].values
    
    # 线性插值
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
    
    im1 = axes[0].imshow(grid_z0, extent=[estimator.aod_grid.min(), estimator.aod_grid.max(), 
                                          estimator.aoa_grid.min(), estimator.aoa_grid.max()],
                         origin='lower', aspect='auto', cmap=Config.PLOT_CMAP)
    axes[0].set_title("1. 原始插值热力图 (含旁瓣干扰)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("AoD (出发角)", fontsize=12)
    axes[0].set_ylabel("AoA (到达角)", fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='RSS (Linear Power)')
    axes[0].grid(alpha=0.3)

    # --- 图2: 稀疏重构热力图 (The "After") ---
    clean_map = estimator.generate_clean_heatmap()
    
    im2 = axes[1].imshow(clean_map, extent=[estimator.aod_grid.min(), estimator.aod_grid.max(), 
                                            estimator.aoa_grid.min(), estimator.aoa_grid.max()],
                         origin='lower', aspect='auto', cmap='inferno')  # 使用更黑背景的配色
    axes[1].set_title(f"2. 稀疏重构热力图 (去噪与锐化)\n发现 {len(estimator.path_df)} 条显著路径", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("AoD (出发角)", fontsize=12)
    axes[1].set_ylabel("AoA (到达角)", fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='RSS (Linear Power)')
    
    # 标注路径点
    if not estimator.path_df.empty:
        # 标注LoS
        los_paths = estimator.path_df[estimator.path_df['Type'] == 'LoS']
        if not los_paths.empty:
            axes[1].scatter(los_paths['AoD'], los_paths['AoA'], 
                            s=200, c='red', marker='o', edgecolors='white', linewidth=2, label='视距径 (LoS)')
        
        # 标注NLoS
        nlos_paths = estimator.path_df[estimator.path_df['Type'] == 'NLoS']
        if not nlos_paths.empty:
            axes[1].scatter(nlos_paths['AoD'], nlos_paths['AoA'], 
                            s=100, c='cyan', marker='x', linewidth=2, label='非视距反射 (NLoS)')
        
        # 添加文字标签
        for _, row in estimator.path_df.iterrows():
            axes[1].text(row['AoD']+2, row['AoA']+2, 
                         f"{row['Type']}\n({row['AoD']:.1f}, {row['AoA']:.1f})", 
                         color='white', fontsize=9, fontweight='bold')

    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.2)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[+] Figure saved to: {output_path}")
    
    plt.show()

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    # 初始化
    processor = BeamDataManager(Config.ANGLE_FILE, Config.RSS_FILE)
    
    if processor.load_data():
        # 数据清洗
        df_clean = processor.process_matrix()
        
        if not df_clean.empty:
            # 算法估计
            estimator = SparseChannelEstimator(df_clean)
            estimator.fit()
            
            # 结果输出
            print("\n[+] Estimation Results:")
            if not estimator.path_df.empty:
                print(estimator.path_df[['AoA', 'AoD', 'Power', 'Type']].to_string())
            else:
                print("No paths detected.")
            
            # 生成输出文件名
            timestamp = extract_filename_timestamp(Config.RSS_FILE)
            output_filename = f"{timestamp}.png"
            output_path = os.path.join(Config.output_dir, output_filename)
            
            # 绘图对比并保存
            plot_comparison(df_clean, estimator, output_path)
        else:
            print("[!] Cleaned data is empty. Check mapping files.")
    else:
        print("[!] Failed to load data files.")