#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理约束特征工程优化模块

该模块实现了基于物理约束的特征工程方法，包括：
1. 创建物理约束特征
2. 特征选择和优化
3. 特征重要性评估

作者: John Wei
日期: 2025-03-30
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_physics_informed_features(df, target_col='PV_number'):
    """
    基于物理约束创建特征
    
    参数:
        df (DataFrame): 输入数据
        target_col (str): 目标列名
        
    返回:
        DataFrame: 包含物理约束特征的数据框
    """
    # 创建副本避免修改原始数据
    df_physics = df.copy()
    
    # 检查必要的列是否存在
    required_columns = [
        'permeability', 'oil_viscosity', 'well_spacing', 
        'effective_thickness', 'formation_pressure', 'temperature'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"缺少创建物理特征所需的列: {missing_columns}")
        
        # 处理缺失的temperature列
        if 'temperature' in missing_columns:
            if 'formation_temperature' in df.columns:
                df_physics['temperature'] = df_physics['formation_temperature']
                logger.info("使用formation_temperature列替代temperature列")
            else:
                logger.warning("无法找到温度相关列，使用默认值60℃")
                df_physics['temperature'] = 60.0  # 使用默认温度值
    
    # 确保所有列都是数值类型
    for col in df_physics.columns:
        if not np.issubdtype(df_physics[col].dtype, np.number):
            try:
                df_physics[col] = pd.to_numeric(df_physics[col], errors='coerce')
                df_physics[col] = df_physics[col].fillna(df_physics[col].mean())
                logger.info(f"将列 {col} 转换为数值类型")
            except:
                logger.warning(f"无法将列 {col} 转换为数值类型，将其删除")
                df_physics = df_physics.drop(columns=[col])
    
    # 确保数据中没有NaN或无穷大值
    # 先计算每列的均值，忽略NaN值
    column_means = df_physics.mean(skipna=True)
    # 使用计算出的均值填充NaN值
    df_physics = df_physics.fillna(column_means)
    # 替换无穷大值为NaN，然后再次填充
    df_physics = df_physics.replace([np.inf, -np.inf], np.nan)
    df_physics = df_physics.fillna(column_means)
    
    # 再次检查是否还有NaN值，如果有则使用0填充（以防万一）
    if df_physics.isnull().any().any():
        logger.warning("在填充均值后仍然存在NaN值，使用0填充")
        df_physics = df_physics.fillna(0)
    
    try:
        # 1. 迁移性比 (Mobility Ratio)
        if all(col in df_physics.columns for col in ['permeability', 'oil_viscosity']):
            df_physics['mobility_ratio'] = df_physics['permeability'] / df_physics['oil_viscosity']
            
        # 2. 指进系数 (Fingering Index)
        if all(col in df_physics.columns for col in ['permeability', 'oil_viscosity', 'well_spacing']):
            df_physics['fingering_index'] = (df_physics['permeability'] * df_physics['well_spacing']) / df_physics['oil_viscosity']
            
        # 3. 流动能力指数 (Flow Capacity Index)
        if all(col in df_physics.columns for col in ['permeability', 'effective_thickness']):
            df_physics['flow_capacity_index'] = df_physics['permeability'] * df_physics['effective_thickness']
            
        # 4. 重力数 (Gravity Number)
        if all(col in df_physics.columns for col in ['permeability', 'oil_density', 'oil_viscosity']):
            df_physics['gravity_number'] = (df_physics['permeability'] * df_physics['oil_density']) / df_physics['oil_viscosity']
            
        # 5. 压力-粘度比 (Pressure-Viscosity Ratio)
        if all(col in df_physics.columns for col in ['formation_pressure', 'oil_viscosity']):
            df_physics['pressure_viscosity_ratio'] = df_physics['formation_pressure'] / df_physics['oil_viscosity']
            
        # 6. 井距-厚度比 (Well Spacing-Thickness Ratio)
        if all(col in df_physics.columns for col in ['well_spacing', 'effective_thickness']):
            df_physics['spacing_thickness_ratio'] = df_physics['well_spacing'] / df_physics['effective_thickness']
            
        # 7. 渗透率-粘度比 (Permeability-Viscosity Ratio)
        if all(col in df_physics.columns for col in ['permeability', 'oil_viscosity']):
            df_physics['perm_viscosity_ratio'] = np.log(df_physics['permeability'] / df_physics['oil_viscosity'])
            
        # 8. 压力-距离梯度 (Pressure-Distance Gradient)
        if all(col in df_physics.columns for col in ['formation_pressure', 'well_spacing']):
            df_physics['pressure_distance_gradient'] = df_physics['formation_pressure'] / df_physics['well_spacing']
            
        # 9. 流动阻力系数 (Flow Resistance Coefficient)
        if all(col in df_physics.columns for col in ['well_spacing', 'permeability', 'effective_thickness']):
            df_physics['flow_resistance'] = df_physics['well_spacing'] / (df_physics['permeability'] * df_physics['effective_thickness'])
            
        # 10. 温度-粘度比 (Temperature-Viscosity Ratio)
        if all(col in df_physics.columns for col in ['temperature', 'oil_viscosity']):
            df_physics['temp_viscosity_ratio'] = df_physics['temperature'] / df_physics['oil_viscosity']
            
        # 11. 无量纲渗透率 (Dimensionless Permeability)
        if 'permeability' in df_physics.columns:
            mean_perm = df_physics['permeability'].mean()
            if mean_perm > 0:
                df_physics['dimensionless_perm'] = df_physics['permeability'] / mean_perm
        
        # 计算创建的物理特征数量
        original_cols = set(df.columns)
        new_cols = set(df_physics.columns) - original_cols
        logger.info(f"创建了{len(new_cols)}个物理约束特征")
        
    except Exception as e:
        logger.error(f"创建物理特征时出错: {str(e)}")
        # 如果出错，返回原始数据框
        return df
    
    return df_physics

def select_optimal_features_limited(df, target_col, max_features=10):
    """
    选择最优特征，限制特征数量
    
    参数:
        df (DataFrame): 输入数据
        target_col (str): 目标列名
        max_features (int): 最大特征数量
        
    返回:
        list: 选择的特征列表
    """
    try:
        # 确保目标列不在特征中
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 确保数据中没有NaN或无穷大值
        # 先计算每列的均值，忽略NaN值
        column_means = X.mean(skipna=True)
        # 使用计算出的均值填充NaN值
        X = X.fillna(column_means)
        # 替换无穷大值为NaN，然后再次填充
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(column_means)
        
        # 再次检查是否还有NaN值，如果有则使用0填充（以防万一）
        if X.isnull().any().any():
            logger.warning("在特征选择中仍然存在NaN值，使用0填充")
            X = X.fillna(0)
        
        # 确保所有列都是数值类型
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].mean())
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # 方法1: 基于互信息的特征选择
        mi_selector = SelectKBest(mutual_info_regression, k=min(max_features, len(X.columns)))
        mi_selector.fit(X_scaled, y)
        mi_scores = mi_selector.scores_
        mi_features = X.columns[mi_selector.get_support()]
        
        # 方法2: 基于随机森林的特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_importances = rf.feature_importances_
        rf_indices = np.argsort(rf_importances)[::-1][:max_features]
        rf_features = X.columns[rf_indices]
        
        # 结合两种方法的结果
        combined_features = list(set(mi_features) | set(rf_features))
        
        # 如果特征数量超过最大限制，选择重要性最高的特征
        if len(combined_features) > max_features:
            # 计算综合得分
            feature_scores = {}
            for feature in combined_features:
                mi_score = mi_scores[list(X.columns).index(feature)]
                rf_score = rf_importances[list(X.columns).index(feature)]
                # 归一化得分
                feature_scores[feature] = (mi_score / max(mi_scores) + rf_score / max(rf_importances)) / 2
            
            # 选择得分最高的特征
            selected_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
            selected_features = [feature for feature, _ in selected_features]
        else:
            selected_features = combined_features
        
        # 确保包含关键物理特征
        key_features = ['permeability', 'oil_viscosity', 'well_spacing', 'effective_thickness', 
                        'formation_pressure', 'mobility_ratio', 'fingering_index', 
                        'flow_capacity_index', 'gravity_number', 'pressure_viscosity_ratio']
        
        # 找出已存在于数据中的关键特征
        existing_key_features = [f for f in key_features if f in X.columns]
        
        # 确保关键特征被包含，但总数不超过max_features
        for feature in existing_key_features:
            if feature not in selected_features and len(selected_features) < max_features:
                selected_features.append(feature)
            elif feature not in selected_features:
                # 如果已达到最大特征数，替换最不重要的特征
                least_important = min([(f, feature_scores.get(f, 0)) for f in selected_features 
                                      if f not in existing_key_features], key=lambda x: x[1], default=(None, 0))
                if least_important[0] and feature_scores.get(feature, 0) > least_important[1]:
                    selected_features.remove(least_important[0])
                    selected_features.append(feature)
        
        # 限制为最大特征数量
        selected_features = selected_features[:max_features]
        
        logger.info(f"选择了{len(selected_features)}个特征: {selected_features}")
        return selected_features
        
    except Exception as e:
        logger.error(f"特征选择时出错: {str(e)}")
        # 如果出错，返回前max_features个列
        return list(df.drop(columns=[target_col]).columns)[:max_features]

def plot_feature_importance(df, target_col, selected_features):
    """
    绘制特征重要性图
    
    参数:
        df (DataFrame): 输入数据
        target_col (str): 目标列名
        selected_features (list): 选择的特征列表
    """
    try:
        X = df[selected_features]
        y = df[target_col]
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用随机森林计算特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        indices = np.argsort(importances)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        plt.title('特征重要性')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('相对重要性')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()
        
        # 绘制特征相关性热图
        plt.figure(figsize=(12, 10))
        corr = df[selected_features + [target_col]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.savefig('results/feature_correlation.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制特征重要性图时出错: {str(e)}")

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_and_preprocess_data
    
    # 加载数据
    df = load_and_preprocess_data('data/raw/CO2气窜原始表.csv')
    
    # 创建物理约束特征
    df_physics = create_physics_informed_features(df)
    
    # 选择最优特征
    selected_features = select_optimal_features_limited(df_physics, 'PV_number', max_features=10)
    
    # 绘制特征重要性图
    plot_feature_importance(df_physics, 'PV_number', selected_features)
    
    print("特征工程完成")
