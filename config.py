#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统配置文件

本文件包含系统的各种配置参数。
"""

import os

# 数据配置
DATA_CONFIG = {
    'target_column': 'PV_number',  # 目标列名（英文）
    'target_column_cn': 'pv数',    # 目标列名（中文）
    'test_size': 0.2,              # 测试集比例
    'random_state': 42,            # 随机种子
    'encoding': 'gbk'              # 默认编码
}

# 物理模型配置
PHYSICS_CONFIG = {
    'co2_viscosity': 0.05,  # CO2粘度，单位: mPa·s
    'co2_density': 800,     # CO2密度，单位: kg/m³
    'gravity': 9.8,         # 重力加速度，单位: m/s²
    'swi': 0.2,             # 不可动水饱和度
    'sor': 0.3,             # 剩余油饱和度
    'soi': 0.8              # 初始油饱和度
}

# 特征工程配置
FEATURE_CONFIG = {
    'max_features': 10,     # 最大特征数量
    'feature_selection_method': 'hybrid',  # 特征选择方法
    # 特征映射（中文到英文）
    'feature_mapping': {
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
}

# 模型配置
MODEL_CONFIG = {
    'default_model': 'gaussian_process',  # 默认模型类型
    'target_r2_min': 0.9,                 # 目标R²最小值
    'target_r2_max': 0.95                 # 目标R²最大值
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'raw_data_dir': os.path.join(BASE_DIR, 'data', 'raw'),
    'processed_data_dir': os.path.join(BASE_DIR, 'data', 'processed'),
    'model_dir': os.path.join(BASE_DIR, 'models'),
    'results_dir': os.path.join(BASE_DIR, 'results'),
    'log_dir': os.path.join(BASE_DIR, 'logs'),
    'docs_dir': os.path.join(BASE_DIR, 'docs'),
    'notebooks_dir': os.path.join(BASE_DIR, 'notebooks')
}

# 可视化配置
VIZ_CONFIG = {
    'figsize': (10, 6),     # 图形大小
    'dpi': 100,             # 图形DPI
    'cmap': 'viridis',      # 默认颜色映射
    'font_family': 'SimHei' # 中文字体
}
