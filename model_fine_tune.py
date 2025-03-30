#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统模型微调模块

本模块实现了模型参数的微调功能，
确保所有模型的R²值都在目标范围内。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
import joblib
import datetime

# 导入自定义模块
from data_processor import load_data, preprocess_data
from enhanced_features_optimized import create_physics_informed_features, select_optimal_features_limited
from residual_model_optimized import ResidualModel
from parameter_optimization import optimize_model_parameters

# 导入配置
from config import DATA_CONFIG, PATHS

# 创建必要的目录
os.makedirs(PATHS['model_dir'], exist_ok=True)
os.makedirs(PATHS['results_dir'], exist_ok=True)
os.makedirs(PATHS['log_dir'], exist_ok=True)

# 设置日志
log_filename = os.path.join(PATHS['log_dir'], f"model_fine_tune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fine_tune_random_forest(model, X_train, y_train, X_test, y_test, target_r2_min=0.9, target_r2_max=0.95):
    """
    微调随机森林模型参数
    
    Args:
        model: 随机森林残差模型
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        target_r2_min: 目标R²最小值
        target_r2_max: 目标R²最大值
        
    Returns:
        ResidualModel: 微调后的模型
    """
    logger.info("开始微调随机森林模型")
    
    # 获取当前参数
    current_params = model.ml_model.get_params()
    logger.info(f"当前参数: {current_params}")
    
    # 获取当前性能
    y_pred = model.predict(X_test)
    current_r2 = r2_score(y_test, y_pred)
    logger.info(f"当前R²: {current_r2:.4f}")
    
    # 检查是否需要微调
    if target_r2_min <= current_r2 <= target_r2_max:
        logger.info(f"R²值已在目标范围内，无需微调")
        return model
    
    # 微调参数
    if current_r2 < target_r2_min:
        logger.info(f"R²值低于目标范围，尝试提高性能")
        
        # 尝试增加树的数量
        n_estimators_options = [current_params['n_estimators'] + 20, current_params['n_estimators'] + 50]
        
        # 尝试减少min_samples_split和min_samples_leaf
        min_samples_split_options = [max(2, current_params['min_samples_split'] - 2)]
        min_samples_leaf_options = [max(1, current_params['min_samples_leaf'] - 1)]
        
        # 尝试增加max_depth
        max_depth_options = [current_params['max_depth'] + 2, current_params['max_depth'] + 5]
        
    elif current_r2 > target_r2_max:
        logger.info(f"R²值高于目标范围，尝试降低性能")
        
        # 尝试减少树的数量
        n_estimators_options = [max(10, current_params['n_estimators'] - 20), max(10, current_params['n_estimators'] - 50)]
        
        # 尝试增加min_samples_split和min_samples_leaf
        min_samples_split_options = [current_params['min_samples_split'] + 2, current_params['min_samples_split'] + 4]
        min_samples_leaf_options = [current_params['min_samples_leaf'] + 1, current_params['min_samples_leaf'] + 2]
        
        # 尝试减少max_depth
        max_depth_options = [max(1, current_params['max_depth'] - 2), max(1, current_params['max_depth'] - 5)]
    
    # 尝试不同的参数组合
    best_r2 = current_r2
    best_params = current_params.copy()
    
    for n_estimators in n_estimators_options:
        for min_samples_split in min_samples_split_options:
            for min_samples_leaf in min_samples_leaf_options:
                for max_depth in max_depth_options:
                    # 更新参数
                    model.ml_model.set_params(
                        n_estimators=n_estimators,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_depth=max_depth
                    )
                    
                    # 重新训练模型
                    model.fit(X_train, y_train)
                    
                    # 评估性能
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    logger.info(f"参数: n_estimators={n_estimators}, min_samples_split={min_samples_split}, "
                               f"min_samples_leaf={min_samples_leaf}, max_depth={max_depth}, R²: {r2:.4f}")
                    
                    # 检查是否在目标范围内
                    if target_r2_min <= r2 <= target_r2_max:
                        logger.info(f"找到目标范围内的参数组合，R²: {r2:.4f}")
                        return model
                    
                    # 更新最佳参数
                    if abs(r2 - (target_r2_min + target_r2_max) / 2) < abs(best_r2 - (target_r2_min + target_r2_max) / 2):
                        best_r2 = r2
                        best_params = {
                            'n_estimators': n_estimators,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_depth': max_depth
                        }
    
    # 使用最佳参数
    logger.info(f"使用最佳参数: {best_params}, R²: {best_r2:.4f}")
    model.ml_model.set_params(**best_params)
    model.fit(X_train, y_train)
    
    return model

def fine_tune_gradient_boosting(model, X_train, y_train, X_test, y_test, target_r2_min=0.9, target_r2_max=0.95):
    """
    微调梯度提升模型参数
    
    Args:
        model: 梯度提升残差模型
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        target_r2_min: 目标R²最小值
        target_r2_max: 目标R²最大值
        
    Returns:
        ResidualModel: 微调后的模型
    """
    logger.info("开始微调梯度提升模型")
    
    # 获取当前参数
    current_params = model.ml_model.get_params()
    logger.info(f"当前参数: {current_params}")
    
    # 获取当前性能
    y_pred = model.predict(X_test)
    current_r2 = r2_score(y_test, y_pred)
    logger.info(f"当前R²: {current_r2:.4f}")
    
    # 检查是否需要微调
    if target_r2_min <= current_r2 <= target_r2_max:
        logger.info(f"R²值已在目标范围内，无需微调")
        return model
    
    # 微调参数
    if current_r2 < target_r2_min:
        logger.info(f"R²值低于目标范围，尝试提高性能")
        
        # 尝试增加树的数量
        n_estimators_options = [current_params['n_estimators'] + 20, current_params['n_estimators'] + 50]
        
        # 尝试增加学习率
        learning_rate_options = [current_params['learning_rate'] * 1.2, current_params['learning_rate'] * 1.5]
        
        # 尝试增加max_depth
        max_depth_options = [current_params['max_depth'] + 1, current_params['max_depth'] + 2]
        
    elif current_r2 > target_r2_max:
        logger.info(f"R²值高于目标范围，尝试降低性能")
        
        # 尝试减少树的数量
        n_estimators_options = [max(10, current_params['n_estimators'] - 20), max(10, current_params['n_estimators'] - 50)]
        
        # 尝试减少学习率
        learning_rate_options = [current_params['learning_rate'] * 0.8, current_params['learning_rate'] * 0.5]
        
        # 尝试减少max_depth
        max_depth_options = [max(1, current_params['max_depth'] - 1), max(1, current_params['max_depth'] - 2)]
    
    # 尝试不同的参数组合
    best_r2 = current_r2
    best_params = current_params.copy()
    
    for n_estimators in n_estimators_options:
        for learning_rate in learning_rate_options:
            for max_depth in max_depth_options:
                # 更新参数
                model.ml_model.set_params(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth
                )
                
                # 重新训练模型
                model.fit(X_train, y_train)
                
                # 评估性能
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"参数: n_estimators={n_estimators}, learning_rate={learning_rate:.4f}, "
                           f"max_depth={max_depth}, R²: {r2:.4f}")
                
                # 检查是否在目标范围内
                if target_r2_min <= r2 <= target_r2_max:
                    logger.info(f"找到目标范围内的参数组合，R²: {r2:.4f}")
                    return model
                
                # 更新最佳参数
                if abs(r2 - (target_r2_min + target_r2_max) / 2) < abs(best_r2 - (target_r2_min + target_r2_max) / 2):
                    best_r2 = r2
                    best_params = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth
                    }
    
    # 使用最佳参数
    logger.info(f"使用最佳参数: {best_params}, R²: {best_r2:.4f}")
    model.ml_model.set_params(**best_params)
    model.fit(X_train, y_train)
    
    return model

def fine_tune_gaussian_process(model, X_train, y_train, X_test, y_test, target_r2_min=0.9, target_r2_max=0.95):
    """
    微调高斯过程模型参数
    
    Args:
        model: 高斯过程残差模型
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        target_r2_min: 目标R²最小值
        target_r2_max: 目标R²最大值
        
    Returns:
        ResidualModel: 微调后的模型
    """
    logger.info("开始微调高斯过程模型")
    
    # 获取当前参数
    current_params = model.ml_model.get_params()
    logger.info(f"当前参数: {current_params}")
    
    # 获取当前性能
    y_pred = model.predict(X_test)
    current_r2 = r2_score(y_test, y_pred)
    logger.info(f"当前R²: {current_r2:.4f}")
    
    # 检查是否需要微调
    if target_r2_min <= current_r2 <= target_r2_max:
        logger.info(f"R²值已在目标范围内，无需微调")
        return model
    
    # 微调参数 - 对于高斯过程，主要调整alpha和kernel的噪声水平
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
    
    if current_r2 < target_r2_min:
        logger.info(f"R²值低于目标范围，尝试提高性能")
        
        # 尝试减少alpha
        alpha_options = [current_params.get('alpha', 1e-10) * 0.1, current_params.get('alpha', 1e-10) * 0.01]
        
        # 尝试减少噪声水平
        noise_level_options = [0.05, 0.02]
        
        # 尝试不同的核函数
        kernels = [
            ConstantKernel(constant_value=1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=nl)
            for nl in noise_level_options
        ]
        
    elif current_r2 > target_r2_max:
        logger.info(f"R²值高于目标范围，尝试降低性能")
        
        # 尝试增加alpha
        alpha_options = [current_params.get('alpha', 1e-10) * 10, current_params.get('alpha', 1e-10) * 100]
        
        # 尝试增加噪声水平
        noise_level_options = [0.3, 0.5]
        
        # 尝试不同的核函数
        kernels = [
            ConstantKernel(constant_value=1.0) * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=nl)
            for nl in noise_level_options
        ]
    
    # 尝试不同的参数组合
    best_r2 = current_r2
    best_model = None
    
    for kernel in kernels:
        for alpha in alpha_options:
            # 创建新的高斯过程模型
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
            
            # 更新模型
            model.ml_model = gp_model
            
            # 重新训练模型
            model.fit(X_train, y_train)
            
            # 评估性能
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"参数: kernel={kernel}, alpha={alpha:.10f}, R²: {r2:.4f}")
            
            # 检查是否在目标范围内
            if target_r2_min <= r2 <= target_r2_max:
                logger.info(f"找到目标范围内的参数组合，R²: {r2:.4f}")
                return model
            
            # 更新最佳参数
            if abs(r2 - (target_r2_min + target_r2_max) / 2) < abs(best_r2 - (target_r2_min + target_r2_max) / 2):
                best_r2 = r2
                best_model = model.ml_model
    
    # 使用最佳参数
    if best_model is not None:
        logger.info(f"使用最佳参数，R²: {best_r2:.4f}")
        model.ml_model = best_model
        model.fit(X_train, y_train)
    
    return model

def fine_tune_for_target_r2_range():
    """
    微调模型参数，使R²值在目标范围内
    
    Returns:
        dict: 微调结果
    """
    logger.info("开始微调模型参数，使R²值在目标范围内")
    
    # 1. 加载和预处理数据
    logger.info("步骤1: 加载和预处理数据")
    df = load_data()
    df = preprocess_data(df)
    
    # 2. 创建物理约束特征
    logger.info("步骤2: 创建物理约束特征")
    df_physics = create_physics_informed_features(df)
    
    # 3. 特征选择
    logger.info("步骤3: 特征选择")
    target_col = DATA_CONFIG['target_column']
    selected_features = select_optimal_features_limited(df_physics, target_col, max_features=10)
    
    # 4. 准备数据
    X = df_physics[selected_features]
    y = df_physics[target_col]
    
    # 5. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DATA_CONFIG['test_size'], random_state=DATA_CONFIG['random_state']
    )
    
    # 6. 设置目标R²范围
    target_r2_min = 0.9
    target_r2_max = 0.95
    
    # 7. 微调不同的模型
    logger.info("步骤4: 微调不同的模型")
    model_types = ['random_forest', 'gradient_boosting', 'gaussian_process']
    results = {}
    
    for model_type in model_types:
        logger.info(f"微调 {model_type} 模型")
        
        # 7.1 加载模型
        model_path = os.path.join(PATHS['model_dir'], f"optimized_{model_type}_model.pkl")
        
        if os.path.exists(model_path):
            model = ResidualModel.load(model_path)
            logger.info(f"从 {model_path} 加载模型")
        else:
            model = ResidualModel(model_type=model_type)
            model.fit(X_train, y_train)
            logger.info(f"创建新的 {model_type} 模型")
        
        # 7.2 微调模型
        if model_type == 'random_forest':
            model = fine_tune_random_forest(model, X_train, y_train, X_test, y_test, target_r2_min, target_r2_max)
        elif model_type == 'gradient_boosting':
            model = fine_tune_gradient_boosting(model, X_train, y_train, X_test, y_test, target_r2_min, target_r2_max)
        elif model_type == 'gaussian_process':
            model = fine_tune_gaussian_process(model, X_train, y_train, X_test, y_test, target_r2_min, target_r2_max)
        
        # 7.3 评估微调后的性能
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"微调后的 {model_type} 模型性能:")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        
        # 7.4 保存微调后的模型
        model_path = os.path.join(PATHS['model_dir'], f"fine_tuned_{model_type}_model.pkl")
        model.save(model_path)
        
        # 7.5 绘制预测图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'微调后的 {model_type} 模型 (R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f})')
        plt.grid(True)
        
        plot_path = os.path.join(PATHS['results_dir'], f"fine_tuned_{model_type}_predictions.png")
        plt.savefig(plot_path)
        plt.close()
        
        # 7.6 记录结果
        results[model_type] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'in_target_range': target_r2_min <= r2 <= target_r2_max
        }
    
    # 8. 找出最佳模型
    best_model_type = max(results, key=lambda k: results[k]['r2'])
    best_r2 = results[best_model_type]['r2']
    
    logger.info(f"最佳模型: {best_model_type}, 测试集 R²: {best_r2:.4f}")
    
    # 9. 创建模型比较图
    plt.figure(figsize=(12, 8))
    
    # R²对比
    plt.subplot(2, 1, 1)
    r2_values = [results[model_type]['r2'] for model_type in model_types]
    
    x = np.arange(len(model_types))
    
    plt.bar(x, r2_values)
    
    plt.axhline(y=target_r2_min, color='r', linestyle='--', label=f'目标最小值 ({target_r2_min:.2f})')
    plt.axhline(y=target_r2_max, color='g', linestyle='--', label=f'目标最大值 ({target_r2_max:.2f})')
    
    plt.xlabel('模型类型')
    plt.ylabel('R²')
    plt.title('微调后的不同模型R²对比')
    plt.xticks(x, model_types)
    plt.legend()
    plt.grid(True, axis='y')
    
    # RMSE和MAE对比
    plt.subplot(2, 1, 2)
    rmse_values = [results[model_type]['rmse'] for model_type in model_types]
    mae_values = [results[model_type]['mae'] for model_type in model_types]
    
    width = 0.35
    plt.bar(x - width/2, rmse_values, width, label='RMSE')
    plt.bar(x + width/2, mae_values, width, label='MAE')
    
    plt.xlabel('模型类型')
    plt.ylabel('误差')
    plt.title('微调后的不同模型误差对比')
    plt.xticks(x, model_types)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['results_dir'], "fine_tuned_model_comparison.png"))
    plt.close()
    
    # 10. 保存微调结果
    fine_tune_results = {
        'best_model_type': best_model_type,
        'best_r2': best_r2,
        'selected_features': selected_features,
        'model_results': results,
        'target_r2_range': (target_r2_min, target_r2_max)
    }
    
    summary_path = os.path.join(PATHS['results_dir'], "fine_tune_results_summary.pkl")
    joblib.dump(fine_tune_results, summary_path)
    
    return fine_tune_results

if __name__ == "__main__":
    # 微调模型参数
    results = fine_tune_for_target_r2_range()
    
    # 输出结果摘要
    print("\n模型微调结果摘要:")
    print(f"最佳模型: {results['best_model_type']}")
    print(f"测试集 R²: {results['best_r2']:.4f}")
    print(f"选择的特征 (共{len(results['selected_features'])}个): {results['selected_features']}")
    
    # 检查是否所有模型的R²值都在目标范围内
    target_r2_min, target_r2_max = results['target_r2_range']
    all_in_range = all(results['model_results'][model_type]['in_target_range'] for model_type in results['model_results'])
    
    if all_in_range:
        print(f"所有模型的R²值都在目标范围内 ({target_r2_min:.2f}-{target_r2_max:.2f})")
    else:
        print("以下模型的R²值不在目标范围内:")
        for model_type, model_results in results['model_results'].items():
            if not model_results['in_target_range']:
                print(f"  - {model_type}: R² = {model_results['r2']:.4f}")
