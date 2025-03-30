#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统数据处理模块（改进版）

本模块实现了数据加载和预处理功能，
支持中文编码的CSV文件。
改进包括：
1. 智能缺失值处理
2. 高级异常值检测与处理
3. 数据验证与质量报告
4. 特征工程增强
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import logging
import json
from datetime import datetime

# 导入配置
from config import DATA_CONFIG, PATHS

# 设置日志
logger = logging.getLogger(__name__)

def load_data(data_path=None):
    """
    加载数据
    
    Args:
        data_path: 数据文件路径，如果为None则使用默认路径
        
    Returns:
        pandas.DataFrame: 加载的数据
    """
    if data_path is None:
        # 使用默认数据路径
        data_path = os.path.join(PATHS['data_dir'], 'raw', 'CO2气窜原始表.csv')
        
    if not os.path.exists(data_path):
        # 如果默认数据不存在，创建示例数据
        logger.warning(f"未找到数据文件: {data_path}，将创建示例数据")
        return _create_sample_data()
    
    # 加载数据，尝试不同的编码
    encodings = ['gbk', 'utf-8', 'gb18030', 'latin1']
    for encoding in encodings:
        try:
            logger.info(f"尝试使用 {encoding} 编码加载数据: {data_path}")
            df = pd.read_csv(data_path, encoding=encoding)
            logger.info(f"成功使用 {encoding} 编码加载数据")
            
            # 记录数据加载信息
            logger.info(f"加载的数据形状: {df.shape}")
            logger.info(f"列名: {df.columns.tolist()}")
            
            return df
        except UnicodeDecodeError:
            logger.warning(f"{encoding} 编码加载失败，尝试下一种编码")
        except Exception as e:
            logger.error(f"加载数据时发生错误: {e}")
            raise
    
    # 如果所有编码都失败，抛出异常
    raise ValueError(f"无法加载数据文件: {data_path}，请检查文件编码")

def _create_sample_data():
    """
    创建示例数据
    
    Returns:
        pandas.DataFrame: 示例数据
    """
    logger.info("创建示例数据")
    
    # 创建数据目录
    os.makedirs(os.path.join(PATHS['data_dir'], 'raw'), exist_ok=True)
    
    # 示例数据参数
    n_samples = 100
    
    # 创建特征
    np.random.seed(42)
    permeability = np.random.uniform(10, 1000, n_samples)  # 渗透率，单位: mD
    oil_viscosity = np.random.uniform(1, 50, n_samples)    # 油相粘度，单位: mPa·s
    well_spacing = np.random.uniform(100, 500, n_samples)  # 井距，单位: m
    effective_thickness = np.random.uniform(5, 50, n_samples)  # 有效厚度，单位: m
    formation_pressure = np.random.uniform(10, 50, n_samples)  # 地层压力，单位: MPa
    
    # 计算目标变量
    PV_number = 0.01 * permeability / oil_viscosity * (well_spacing / effective_thickness) * (formation_pressure / 20)
    
    # 添加噪声
    PV_number = PV_number + np.random.normal(0, 0.05, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        '区块': ['示例区块'] * n_samples,
        '地层温度℃': np.random.uniform(70, 100, n_samples),
        '地层压力mpa': formation_pressure,
        '注气前地层压力mpa': np.random.uniform(4, 10, n_samples),
        '压力水平': np.random.uniform(0.2, 0.6, n_samples),
        '渗透率md': permeability,
        '地层原油粘度mpas': oil_viscosity,
        '地层原油密度g/cm3': np.random.uniform(0.7, 0.9, n_samples),
        '井组有效厚度m': effective_thickness,
        '注气井压裂': np.random.choice(['是', '否'], n_samples),
        '井距m': well_spacing,
        '孔隙度/%': np.random.uniform(15, 25, n_samples),
        '注入前含油饱和度/%': np.random.uniform(50, 70, n_samples),
        'pv数': PV_number
    })
    
    # 添加一些缺失值以测试缺失值处理
    mask = np.random.random(df.shape) < 0.05  # 5%的缺失率
    df = df.mask(mask)
    
    # 保存示例数据
    sample_data_path = os.path.join(PATHS['data_dir'], 'raw', 'sample_data.csv')
    df.to_csv(sample_data_path, index=False, encoding='gbk')
    logger.info(f"示例数据已保存到 {sample_data_path}")
    
    return df

def validate_data(df):
    """
    验证数据质量并生成报告
    
    Args:
        df: 输入数据
        
    Returns:
        dict: 数据质量报告
    """
    logger.info("开始验证数据质量")
    
    # 基本信息
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': {},
        'categorical_stats': {},
        'correlation': {}
    }
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].count() > 0:  # 确保列不是全部缺失
            report['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'unique_values': df[col].nunique()
            }
    
    # 分类列统计
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].count() > 0:  # 确保列不是全部缺失
            value_counts = df[col].value_counts().to_dict()
            report['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].value_counts().index[0] if df[col].nunique() > 0 else None,
                'value_counts': value_counts if df[col].nunique() <= 10 else "Too many to display"
            }
    
    # 相关性分析
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(3)
        report['correlation'] = corr_matrix.to_dict()
    
    # 保存报告
    report_dir = os.path.join(PATHS['data_dir'], 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    logger.info(f"数据质量报告已保存到 {report_path}")
    
    # 生成可视化报告
    generate_data_quality_visualizations(df, report_dir)
    
    return report

def generate_data_quality_visualizations(df, report_dir):
    """
    生成数据质量可视化报告
    
    Args:
        df: 输入数据
        report_dir: 报告保存目录
    """
    logger.info("生成数据质量可视化报告")
    
    # 创建可视化目录
    viz_dir = os.path.join(report_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 缺失值可视化
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing_percent = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'缺失值数量': missing, '缺失百分比': missing_percent})
        
        ax = missing_df['缺失百分比'].plot(kind='bar', figsize=(12, 6), color='orange')
        ax.set_xlabel('特征')
        ax.set_ylabel('缺失百分比')
        ax.set_title('各特征缺失值百分比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'missing_values.png'))
        plt.close()
    
    # 数值特征分布可视化
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        # 直方图和密度图
        for i, col in enumerate(numeric_cols):
            if df[col].count() > 0:  # 确保列不是全部缺失
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'{col} 分布')
                plt.xlabel(col)
                plt.ylabel('频率')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'distribution_{col}.png'))
                plt.close()
        
        # 箱线图
        plt.figure(figsize=(12, 8))
        df[numeric_cols].boxplot(vert=False, figsize=(12, 8))
        plt.title('数值特征箱线图')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'boxplots.png'))
        plt.close()
        
        # 相关性热图
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
            plt.title('特征相关性热图')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'))
            plt.close()
    
    logger.info(f"数据质量可视化报告已保存到 {viz_dir}")

def handle_missing_values(df):
    """
    智能处理缺失值
    
    Args:
        df: 输入数据
        
    Returns:
        pandas.DataFrame: 处理缺失值后的数据
    """
    logger.info("开始智能处理缺失值")
    
    # 复制数据
    df_processed = df.copy()
    
    # 检查缺失值
    missing_values = df_processed.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"发现缺失值:\n{missing_values[missing_values > 0]}")
        
        # 对分类列使用众数填充
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode()[0]
                df_processed[col] = df_processed[col].fillna(mode_value)
                logger.info(f"列 '{col}' 的缺失值已使用众数 '{mode_value}' 填充")
        
        # 对数值列根据分布特性选择填充方法
        numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                # 检查分布偏度
                if abs(df_processed[col].skew()) > 1:
                    # 偏斜分布使用中位数填充
                    median_value = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_value)
                    logger.info(f"列 '{col}' 的缺失值已使用中位数 {median_value:.4f} 填充（偏斜分布）")
                else:
                    # 正态分布使用均值填充
                    mean_value = df_processed[col].mean()
                    df_processed[col] = df_processed[col].fillna(mean_value)
                    logger.info(f"列 '{col}' 的缺失值已使用均值 {mean_value:.4f} 填充（正态分布）")
        
        # 对于关键特征，如果缺失值比例不高，尝试使用KNN填充
        key_features = ['permeability', 'oil_viscosity', 'well_spacing', 'effective_thickness', 
                        'formation_pressure', 'porosity', 'oil_saturation', 'PV_number']
        
        # 将中文列名映射到英文
        column_mapping = {
            '渗透率md': 'permeability',
            '地层原油粘度mpas': 'oil_viscosity',
            '井距m': 'well_spacing',
            '井组有效厚度m': 'effective_thickness',
            '地层压力mpa': 'formation_pressure',
            '孔隙度/%': 'porosity',
            '注入前含油饱和度/%': 'oil_saturation',
            'pv数': 'PV_number'
        }
        
        # 反向映射，找到中文列名
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # 找到存在于数据中的关键特征（中文列名）
        existing_key_features = [reverse_mapping.get(f, f) for f in key_features if reverse_mapping.get(f, f) in df_processed.columns]
        
        if existing_key_features:
            # 提取数值列用于KNN填充
            numeric_data = df_processed.select_dtypes(include=['float64', 'int64'])
            
            # 检查是否有足够的数据进行KNN填充
            if len(numeric_data) > 5 and numeric_data.isnull().sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1]) < 0.3:
                try:
                    # 使用KNN填充
                    imputer = KNNImputer(n_neighbors=5)
                    numeric_data_imputed = pd.DataFrame(
                        imputer.fit_transform(numeric_data),
                        columns=numeric_data.columns,
                        index=numeric_data.index
                    )
                    
                    # 只更新关键特征的缺失值
                    for col in existing_key_features:
                        if col in numeric_data.columns and df_processed[col].isnull().sum() > 0:
                            df_processed[col] = numeric_data_imputed[col]
                            logger.info(f"列 '{col}' 的缺失值已使用KNN方法填充")
                except Exception as e:
                    logger.warning(f"KNN填充失败: {e}，将使用常规方法")
    
    return df_processed

def handle_outliers(df):
    """
    高级异常值处理
    
    Args:
        df: 输入数据
        
    Returns:
        pandas.DataFrame: 处理异常值后的数据
        dict: 异常值信息
    """
    logger.info("开始高级异常值处理")
    
    # 复制数据
    df_processed = df.copy()
    
    # 存储异常值信息
    outlier_info = {}
    
    # 检查异常值
    for col in df_processed.columns:
        if col not in ['区块', '注气井压裂'] and df_processed[col].dtype in [np.float64, np.int64]:
            # 使用IQR方法检测异常值
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 检测异常值
            outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
            
            if len(outliers) > 0:
                logger.warning(f"列 {col} 中发现 {len(outliers)} 个异常值")
                
                # 记录异常值信息
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df_processed) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'values': df_processed.loc[outliers.index, col].tolist(),
                    'indices': outliers.index.tolist()
                }
                
                # 区分极端异常值和中度异常值
                extreme_outliers = df_processed[(df_processed[col] < Q1 - 3 * IQR) | (df_processed[col] > Q3 + 3 * IQR)]
                moderate_outliers = outliers.drop(extreme_outliers.index, errors='ignore')
                
                # 处理极端异常值 - 截断
                if not extreme_outliers.empty:
                    df_processed.loc[extreme_outliers.index, col] = df_processed.loc[extreme_outliers.index, col].clip(
                        lower_bound, upper_bound
                    )
                    logger.info(f"已将列 {col} 中的 {len(extreme_outliers)} 个极端异常值截断到边界内")
                
                # 处理中度异常值 - Winsorization
                if not moderate_outliers.empty:
                    for idx in moderate_outliers.index:
                        if df_processed.loc[idx, col] < lower_bound:
                            df_processed.loc[idx, col] = lower_bound
                        else:
                            df_processed.loc[idx, col] = upper_bound
                    logger.info(f"已将列 {col} 中的 {len(moderate_outliers)} 个中度异常值进行Winsorization处理")
    
    # 保存异常值信息
    if outlier_info:
        report_dir = os.path.join(PATHS['data_dir'], 'reports')
        os.makedirs(report_dir, exist_ok=True)
        outlier_report_path = os.path.join(report_dir, f"outlier_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(outlier_report_path, 'w', encoding='utf-8') as f:
            json.dump(outlier_info, f, ensure_ascii=False, indent=4)
        
        logger.info(f"异常值报告已保存到 {outlier_report_path}")
    
    return df_processed, outlier_info

def normalize_features(df, method='standard'):
    """
    特征标准化
    
    Args:
        df: 输入数据
        method: 标准化方法，'standard'或'robust'
        
    Returns:
        pandas.DataFrame: 标准化后的数据
        dict: 标准化器，用于后续转换
    """
    logger.info(f"开始特征标准化，使用{method}方法")
    
    # 复制数据
    df_processed = df.copy()
    
    # 选择数值列进行标准化
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 排除目标变量
    if 'PV_number' in numeric_cols:
        numeric_cols.remove('PV_number')
    elif 'pv数' in numeric_cols:
        numeric_cols.remove('pv数')
    
    # 创建标准化器
    scalers = {}
    
    if numeric_cols:
        for col in numeric_cols:
            # 选择标准化方法
            if method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            # 拟合并转换
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
            
            # 保存标准化器
            scalers[col] = scaler
            
            logger.info(f"列 '{col}' 已使用{method}方法标准化")
    
    # 保存标准化器
    scaler_dir = os.path.join(PATHS['data_dir'], 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, f"{method}_scalers.pkl")
    
    import joblib
    joblib.dump(scalers, scaler_path)
    logger.info(f"标准化器已保存到 {scaler_path}")
    
    return df_processed, scalers

def preprocess_data(df, normalize=False, normalization_method='standard'):
    """
    预处理数据
    
    Args:
        df: 输入数据
        normalize: 是否进行标准化
        normalization_method: 标准化方法，'standard'或'robust'
        
    Returns:
        pandas.DataFrame: 预处理后的数据
    """
    logger.info("开始预处理数据")
    
    # 复制数据
    df_processed = df.copy()
    
    # 1. 数据验证
    data_report = validate_data(df_processed)
    
    # 2. 处理缺失值
    df_processed = handle_missing_values(df_processed)
    
    # 3. 处理异常值
    df_processed, outlier_info = handle_outliers(df_processed)
    
    # 4. 处理分类变量
    if '注气井压裂' in df_processed.columns:
        df_processed['注气井压裂'] = df_processed['注气井压裂'].map({'是': 1, '否': 0})
        logger.info("已将'注气井压裂'列转换为数值: 是=1, 否=0")
    
    # 5. 重命名列以便于后续处理
    column_mapping = {
        '区块': 'block',
        '地层温度℃': 'formation_temperature',
        '地层压力mpa': 'formation_pressure',
        '注气前地层压力mpa': 'pre_injection_pressure',
        '压力水平': 'pressure_level',
        '渗透率md': 'permeability',
        '地层原油粘度mpas': 'oil_viscosity',
        '地层原油密度g/cm3': 'oil_density',
        '井组有效厚度m': 'effective_thickness',
        '注气井压裂': 'fracturing',
        '井距m': 'well_spacing',
        '孔隙度/%': 'porosity',
        '注入前含油饱和度/%': 'oil_saturation',
        'pv数': 'PV_number'
    }
    
    df_processed = df_processed.rename(columns=column_mapping)
    logger.info("已将列名从中文转换为英文")
    
    # 6. 标准化特征（可选）
    if normalize:
        df_processed, scalers = normalize_features(df_processed, method=normalization_method)
    
    # 7. 检查数据类型
    for col in df_processed.columns:
        if col != 'block' and not np.issubdtype(df_processed[col].dtype, np.number):
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                logger.info(f"已将列 '{col}' 转换为数值类型")
            except:
                logger.warning(f"无法将列 '{col}' 转换为数值类型")
    
    # 8. 最终检查确保没有NaN或无穷大值
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    if df_processed.isnull().sum().sum() > 0:
        logger.warning("预处理后仍存在缺失值，将使用列均值填充")
        df_processed = df_processed.fillna(df_processed.mean())
    
    logger.info("数据预处理完成")
    return df_processed

def save_processed_data(df, output_path=None):
    """
    保存预处理后的数据
    
    Args:
        df: 预处理后的数据
        output_path: 输出路径，如果为None则使用默认路径
    """
    if output_path is None:
        output_path = os.path.join(PATHS['data_dir'], 'processed', 'processed_data.csv')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存数据
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"预处理后的数据已保存到 {output_path}")
    
    # 同时保存一份Excel格式的数据，方便查看
    excel_path = output_path.replace('.csv', '.xlsx')
    try:
        df.to_excel(excel_path, index=False)
        logger.info(f"预处理后的数据已同时保存为Excel格式: {excel_path}")
    except Exception as e:
        logger.warning(f"保存Excel格式失败: {e}")

def load_and_preprocess_data(data_path=None, save_processed=True, normalize=False, normalization_method='standard'):
    """
    加载并预处理数据
    
    Args:
        data_path: 数据文件路径，如果为None则使用默认路径
        save_processed: 是否保存预处理后的数据
        normalize: 是否进行标准化
        normalization_method: 标准化方法，'standard'或'robust'
        
    Returns:
        pandas.DataFrame: 预处理后的数据
    """
    # 加载数据
    df = load_data(data_path)
    
    # 预处理数据
    df_processed = preprocess_data(df, normalize=normalize, normalization_method=normalization_method)
    
    # 保存预处理后的数据
    if save_processed:
        save_processed_data(df_processed)
    
    return df_processed

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试数据加载和预处理
    df = load_data()
    print("原始数据:")
    print(df.head())
    
    df_processed = preprocess_data(df, normalize=True, normalization_method='robust')
    print("\n预处理后的数据:")
    print(df_processed.head())
    
    # 保存预处理后的数据
    save_processed_data(df_processed)
