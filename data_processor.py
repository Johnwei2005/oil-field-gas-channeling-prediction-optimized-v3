#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统数据处理模块

本模块实现了数据加载和预处理功能，
支持中文编码的CSV文件。
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

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
    
    # 保存示例数据
    sample_data_path = os.path.join(PATHS['data_dir'], 'raw', 'sample_data.csv')
    df.to_csv(sample_data_path, index=False, encoding='gbk')
    logger.info(f"示例数据已保存到 {sample_data_path}")
    
    return df

def preprocess_data(df):
    """
    预处理数据
    
    Args:
        df: 输入数据
        
    Returns:
        pandas.DataFrame: 预处理后的数据
    """
    logger.info("开始预处理数据")
    
    # 复制数据
    df_processed = df.copy()
    
    # 检查缺失值
    missing_values = df_processed.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"发现缺失值:\n{missing_values[missing_values > 0]}")
        
        # 填充缺失值
        df_processed = df_processed.fillna(df_processed.mean())
        logger.info("已使用均值填充缺失值")
    
    # 检查异常值
    for col in df_processed.columns:
        if col not in ['区块', '注气井压裂'] and df_processed[col].dtype in [np.float64, np.int64]:
            # 使用IQR方法检测异常值
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
            
            if len(outliers) > 0:
                logger.warning(f"列 {col} 中发现 {len(outliers)} 个异常值")
                
                # 将异常值限制在边界内
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                logger.info(f"已将列 {col} 中的异常值限制在边界内")
    
    # 处理分类变量
    if '注气井压裂' in df_processed.columns:
        df_processed['注气井压裂'] = df_processed['注气井压裂'].map({'是': 1, '否': 0})
        logger.info("已将'注气井压裂'列转换为数值: 是=1, 否=0")
    
    # 重命名列以便于后续处理
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

def load_and_preprocess_data(data_path=None, save_processed=True):
    """
    加载并预处理数据
    
    Args:
        data_path: 数据文件路径，如果为None则使用默认路径
        save_processed: 是否保存预处理后的数据
        
    Returns:
        pandas.DataFrame: 预处理后的数据
    """
    # 加载数据
    df = load_data(data_path)
    
    # 预处理数据
    df_processed = preprocess_data(df)
    
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
    
    df_processed = preprocess_data(df)
    print("\n预处理后的数据:")
    print(df_processed.head())
    
    # 保存预处理后的数据
    save_processed_data(df_processed)
