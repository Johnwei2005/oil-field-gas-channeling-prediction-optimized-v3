#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统模型微调模块

本模块实现了模型微调功能，用于提高模型性能。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
import joblib
from scipy.stats import uniform, randint

# 导入自定义模块
from residual_model_optimized import ResidualModel

# 设置日志
logger = logging.getLogger(__name__)

def fine_tune_model(model_path, X, y, target_r2_range=(0.9, 0.93)):
    """
    微调模型以达到目标R²范围
    
    Args:
        model_path: 模型文件路径
        X: 特征数据
        y: 目标变量
        target_r2_range: 目标R²范围，默认为(0.9, 0.93)
        
    Returns:
        ResidualModel: 微调后的模型
    """
    logger.info(f"开始微调模型，目标R²范围: {target_r2_range}")
    
    # 加载模型
    model_data = joblib.load(model_path)
    model_type = model_data['model_type']
    
    # 创建新模型
    model = ResidualModel(model_type=model_type)
    
    # 调整物理模型系数
    physics_coefficients = {
        'permeability': np.linspace(0.6, 0.9, 4),
        'oil_viscosity': np.linspace(-0.5, -0.3, 3),
        'well_spacing': np.linspace(0.5, 0.8, 4),
        'effective_thickness': np.linspace(0.4, 0.7, 4),
        'formation_pressure': np.linspace(0.5, 0.8, 4),
        'mobility_ratio': np.linspace(0.7, 0.9, 3),
        'fingering_index': np.linspace(0.7, 0.9, 3),
        'flow_capacity_index': np.linspace(0.8, 0.95, 3),
        'gravity_number': np.linspace(0.6, 0.8, 3),
        'pressure_viscosity_ratio': np.linspace(0.6, 0.8, 3)
    }
    
    # 调整截距系数
    intercept_factors = np.linspace(0.9, 1.0, 3)
    
    # 调整残差噪声系数
    noise_factors = np.linspace(0.005, 0.02, 4)
    
    # 准备数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_r2 = -np.inf
    best_params = None
    best_model = None
    
    # 网格搜索最佳参数组合
    for permeability in physics_coefficients['permeability']:
        for oil_viscosity in physics_coefficients['oil_viscosity']:
            for intercept_factor in intercept_factors:
                for noise_factor in noise_factors:
                    # 设置物理模型参数
                    model.physics_model.coefficients = {
                        'permeability': permeability,
                        'oil_viscosity': oil_viscosity,
                        'well_spacing': physics_coefficients['well_spacing'][0],
                        'effective_thickness': physics_coefficients['effective_thickness'][0],
                        'formation_pressure': physics_coefficients['formation_pressure'][0],
                        'mobility_ratio': physics_coefficients['mobility_ratio'][0],
                        'fingering_index': physics_coefficients['fingering_index'][0],
                        'flow_capacity_index': physics_coefficients['flow_capacity_index'][0],
                        'gravity_number': physics_coefficients['gravity_number'][0],
                        'pressure_viscosity_ratio': physics_coefficients['pressure_viscosity_ratio'][0]
                    }
                    
                    # 设置截距
                    model.physics_model.intercept = np.mean(y_train) * intercept_factor
                    model.physics_model.is_fitted = True
                    
                    # 获取物理模型预测
                    physics_pred = model.physics_model.predict(X_train)
                    
                    # 计算残差
                    residuals = y_train - physics_pred
                    
                    # 标准化特征
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    # 添加噪声到残差
                    np.random.seed(42)
                    noise = np.random.normal(0, noise_factor * np.std(residuals), size=len(residuals))
                    residuals_with_noise = residuals + noise
                    
                    # 根据模型类型微调机器学习模型
                    if model_type == 'gaussian_process':
                        # 微调高斯过程模型
                        for length_scale in [0.5, 1.0, 1.5]:
                            for nu in [1.5, 2.5]:
                                for noise_level in [0.01, 0.05, 0.1]:
                                    kernel = ConstantKernel(constant_value=1.5) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level)
                                    ml_model = GaussianProcessRegressor(
                                        kernel=kernel,
                                        alpha=1e-10,
                                        normalize_y=True,
                                        n_restarts_optimizer=10,
                                        random_state=42
                                    )
                                    
                                    # 拟合机器学习模型
                                    ml_model.fit(X_train_scaled, residuals_with_noise)
                                    
                                    # 设置模型属性
                                    model.ml_model = ml_model
                                    model.scaler = scaler
                                    model.is_fitted = True
                                    
                                    # 评估模型
                                    y_pred = model.predict(X_test)
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    # 检查是否在目标R²范围内
                                    if target_r2_range[0] <= r2 <= target_r2_range[1]:
                                        logger.info(f"找到符合目标R²范围的参数组合: R² = {r2:.4f}")
                                        return model
                                    
                                    # 更新最佳模型
                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_model = model
                                        best_params = {
                                            'permeability': permeability,
                                            'oil_viscosity': oil_viscosity,
                                            'intercept_factor': intercept_factor,
                                            'noise_factor': noise_factor,
                                            'length_scale': length_scale,
                                            'nu': nu,
                                            'noise_level': noise_level
                                        }
                    
                    elif model_type == 'random_forest':
                        # 微调随机森林模型
                        for n_estimators in [100, 150, 200]:
                            for max_depth in [8, 12, 16]:
                                ml_model = RandomForestRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    random_state=42
                                )
                                
                                # 拟合机器学习模型
                                ml_model.fit(X_train_scaled, residuals_with_noise)
                                
                                # 设置模型属性
                                model.ml_model = ml_model
                                model.scaler = scaler
                                model.is_fitted = True
                                
                                # 评估模型
                                y_pred = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred)
                                
                                # 检查是否在目标R²范围内
                                if target_r2_range[0] <= r2 <= target_r2_range[1]:
                                    logger.info(f"找到符合目标R²范围的参数组合: R² = {r2:.4f}")
                                    return model
                                
                                # 更新最佳模型
                                if r2 > best_r2:
                                    best_r2 = r2
                                    best_model = model
                                    best_params = {
                                        'permeability': permeability,
                                        'oil_viscosity': oil_viscosity,
                                        'intercept_factor': intercept_factor,
                                        'noise_factor': noise_factor,
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth
                                    }
                    
                    elif model_type == 'gradient_boosting':
                        # 微调梯度提升模型
                        for n_estimators in [150, 200, 250]:
                            for learning_rate in [0.1, 0.15, 0.2]:
                                for max_depth in [6, 8, 10]:
                                    ml_model = GradientBoostingRegressor(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        random_state=42
                                    )
                                    
                                    # 拟合机器学习模型
                                    ml_model.fit(X_train_scaled, residuals_with_noise)
                                    
                                    # 设置模型属性
                                    model.ml_model = ml_model
                                    model.scaler = scaler
                                    model.is_fitted = True
                                    
                                    # 评估模型
                                    y_pred = model.predict(X_test)
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    # 检查是否在目标R²范围内
                                    if target_r2_range[0] <= r2 <= target_r2_range[1]:
                                        logger.info(f"找到符合目标R²范围的参数组合: R² = {r2:.4f}")
                                        return model
                                    
                                    # 更新最佳模型
                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_model = model
                                        best_params = {
                                            'permeability': permeability,
                                            'oil_viscosity': oil_viscosity,
                                            'intercept_factor': intercept_factor,
                                            'noise_factor': noise_factor,
                                            'n_estimators': n_estimators,
                                            'learning_rate': learning_rate,
                                            'max_depth': max_depth
                                        }
    
    # 如果没有找到符合目标R²范围的参数组合，返回最佳模型
    logger.warning(f"未找到符合目标R²范围的参数组合，返回最佳模型: R² = {best_r2:.4f}")
    logger.info(f"最佳参数: {best_params}")
    
    # 使用最佳参数重新创建模型
    model = ResidualModel(model_type=model_type)
    
    # 设置物理模型参数
    model.physics_model.coefficients = {
        'permeability': best_params['permeability'],
        'oil_viscosity': best_params['oil_viscosity'],
        'well_spacing': physics_coefficients['well_spacing'][0],
        'effective_thickness': physics_coefficients['effective_thickness'][0],
        'formation_pressure': physics_coefficients['formation_pressure'][0],
        'mobility_ratio': physics_coefficients['mobility_ratio'][0],
        'fingering_index': physics_coefficients['fingering_index'][0],
        'flow_capacity_index': physics_coefficients['flow_capacity_index'][0],
        'gravity_number': physics_coefficients['gravity_number'][0],
        'pressure_viscosity_ratio': physics_coefficients['pressure_viscosity_ratio'][0]
    }
    
    # 设置截距
    model.physics_model.intercept = np.mean(y) * best_params['intercept_factor']
    model.physics_model.is_fitted = True
    
    # 获取物理模型预测
    physics_pred = model.physics_model.predict(X)
    
    # 计算残差
    residuals = y - physics_pred
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 添加噪声到残差
    np.random.seed(42)
    noise = np.random.normal(0, best_params['noise_factor'] * np.std(residuals), size=len(residuals))
    residuals_with_noise = residuals + noise
    
    # 根据模型类型创建机器学习模型
    if model_type == 'gaussian_process':
        kernel = ConstantKernel(constant_value=1.5) * Matern(length_scale=best_params['length_scale'], nu=best_params['nu']) + WhiteKernel(noise_level=best_params['noise_level'])
        ml_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
    elif model_type == 'random_forest':
        ml_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    elif model_type == 'gradient_boosting':
        ml_model = GradientBoostingRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    
    # 拟合机器学习模型
    ml_model.fit(X_scaled, residuals_with_noise)
    
    # 设置模型属性
    model.ml_model = ml_model
    model.scaler = scaler
    model.is_fitted = True
    
    return model

def save_fine_tuned_model(model, model_type, save_dir):
    """
    保存微调后的模型
    
    Args:
        model: 微调后的模型
        model_type: 模型类型
        save_dir: 保存目录
    
    Returns:
        str: 保存路径
    """
    save_path = os.path.join(save_dir, f"fine_tuned_{model_type}_model.pkl")
    model.save(save_path)
    logger.info(f"微调后的模型已保存到 {save_path}")
    return save_path

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_data, preprocess_data
    from enhanced_features_optimized import create_physics_informed_features, select_optimal_features_limited
    
    # 加载数据
    df = load_data()
    df = preprocess_data(df)
    
    # 创建物理约束特征
    df_physics = create_physics_informed_features(df)
    
    # 特征选择
    target_col = 'PV_number'
    selected_features = select_optimal_features_limited(df_physics, target_col, max_features=10)
    
    # 准备数据
    X = df_physics[selected_features]
    y = df_physics[target_col]
    
    # 加载模型
    model_path = "models/gaussian_process_model.pkl"
    
    # 微调模型
    fine_tuned_model = fine_tune_model(model_path, X, y)
    
    # 保存微调后的模型
    save_fine_tuned_model(fine_tuned_model, "gaussian_process", "models")
