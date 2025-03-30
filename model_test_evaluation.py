#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统模型测试与评估模块

本模块实现了模型测试与评估功能，
验证优化后的模型性能是否在目标范围内。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
log_filename = os.path.join(PATHS['log_dir'], f"model_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_prediction_vs_actual(y_true, y_pred, title, output_path):
    """绘制预测值与实际值对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    plt.title(f'{title} (R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f})')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def evaluate_model_performance(model, X_test, y_test):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试目标
        
    Returns:
        dict: 性能指标
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # 计算MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # 计算调整R²
    def adjusted_r2(r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'adj_r2': adj_r2
    }
    
    logger.info(f"模型性能评估:")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MAPE: {mape:.4f}%")
    logger.info(f"调整R²: {adj_r2:.4f}")
    
    return metrics

def test_model_with_cross_validation(model_type, X, y, cv=5):
    """
    使用交叉验证测试模型
    
    Args:
        model_type: 模型类型
        X: 特征数据
        y: 目标变量
        cv: 交叉验证折数
        
    Returns:
        tuple: (平均R², R²标准差)
    """
    # 创建模型
    model = ResidualModel(model_type=model_type)
    
    # 定义自定义评分函数
    def custom_score(estimator, X, y):
        y_pred = estimator.predict(X)
        return r2_score(y, y_pred)
    
    # 执行交叉验证
    cv_scores = []
    for train_idx, test_idx in train_test_split(range(len(X)), test_size=0.2, random_state=42, shuffle=True):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
    
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)
    
    logger.info(f"{model_type} 交叉验证 R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    return mean_r2, std_r2

def test_and_evaluate_model():
    """
    测试和评估模型性能
    
    Returns:
        dict: 测试结果
    """
    logger.info("开始测试和评估模型性能")
    
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
    
    # 6. 测试不同的模型
    logger.info("步骤4: 测试不同的模型")
    model_types = ['random_forest', 'gradient_boosting', 'gaussian_process']
    results = {}
    
    target_r2_min = 0.9
    target_r2_max = 0.95
    
    for model_type in model_types:
        logger.info(f"测试 {model_type} 模型")
        
        # 6.1 优化模型参数
        best_params = optimize_model_parameters(X_train, y_train, model_type, target_r2_min, target_r2_max)
        
        # 6.2 创建并训练模型
        model = ResidualModel(model_type=model_type)
        
        # 设置优化后的参数
        if model_type == 'random_forest':
            model.ml_model.set_params(**best_params)
        elif model_type == 'gradient_boosting':
            model.ml_model.set_params(**best_params)
        elif model_type == 'gaussian_process':
            # 高斯过程模型需要特殊处理，因为kernel参数不能直接设置
            if 'kernel' in best_params:
                kernel_param = best_params.pop('kernel')
                model.ml_model = type(model.ml_model)(
                    kernel=kernel_param,
                    **best_params
                )
            else:
                model.ml_model.set_params(**best_params)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 6.3 评估模型性能
        metrics = evaluate_model_performance(model, X_test, y_test)
        
        # 6.4 交叉验证
        cv_mean_r2, cv_std_r2 = test_model_with_cross_validation(model_type, X, y)
        
        # 6.5 保存模型
        model_path = os.path.join(PATHS['model_dir'], f"optimized_{model_type}_model.pkl")
        model.save(model_path)
        
        # 6.6 绘制预测图
        y_pred = model.predict(X_test)
        plot_path = os.path.join(PATHS['results_dir'], f"optimized_{model_type}_predictions.png")
        plot_prediction_vs_actual(y_test, y_pred, f"优化后的 {model_type} 模型", plot_path)
        
        # 6.7 记录结果
        results[model_type] = {
            'metrics': metrics,
            'cv_mean_r2': cv_mean_r2,
            'cv_std_r2': cv_std_r2,
            'params': best_params
        }
        
        # 6.8 检查R²值是否在目标范围内
        if metrics['r2'] < target_r2_min:
            logger.warning(f"{model_type} 模型R²值 ({metrics['r2']:.4f}) 低于目标范围 ({target_r2_min:.2f}-{target_r2_max:.2f})")
        elif metrics['r2'] > target_r2_max:
            logger.warning(f"{model_type} 模型R²值 ({metrics['r2']:.4f}) 高于目标范围 ({target_r2_min:.2f}-{target_r2_max:.2f})")
        else:
            logger.info(f"{model_type} 模型R²值 ({metrics['r2']:.4f}) 在目标范围内 ({target_r2_min:.2f}-{target_r2_max:.2f})")
    
    # 7. 找出最佳模型
    best_model_type = max(results, key=lambda k: results[k]['metrics']['r2'])
    best_r2 = results[best_model_type]['metrics']['r2']
    
    logger.info(f"最佳模型: {best_model_type}, 测试集 R²: {best_r2:.4f}")
    
    # 8. 创建模型比较图
    plt.figure(figsize=(12, 8))
    
    # R²对比
    plt.subplot(2, 1, 1)
    r2_values = [results[model_type]['metrics']['r2'] for model_type in model_types]
    cv_r2_values = [results[model_type]['cv_mean_r2'] for model_type in model_types]
    
    x = np.arange(len(model_types))
    width = 0.35
    
    plt.bar(x - width/2, r2_values, width, label='测试集 R²')
    plt.bar(x + width/2, cv_r2_values, width, label='交叉验证 R²')
    
    plt.axhline(y=target_r2_min, color='r', linestyle='--', label=f'目标最小值 ({target_r2_min:.2f})')
    plt.axhline(y=target_r2_max, color='g', linestyle='--', label=f'目标最大值 ({target_r2_max:.2f})')
    
    plt.xlabel('模型类型')
    plt.ylabel('R²')
    plt.title('不同模型R²对比')
    plt.xticks(x, model_types)
    plt.legend()
    plt.grid(True, axis='y')
    
    # RMSE和MAE对比
    plt.subplot(2, 1, 2)
    rmse_values = [results[model_type]['metrics']['rmse'] for model_type in model_types]
    mae_values = [results[model_type]['metrics']['mae'] for model_type in model_types]
    
    plt.bar(x - width/2, rmse_values, width, label='RMSE')
    plt.bar(x + width/2, mae_values, width, label='MAE')
    
    plt.xlabel('模型类型')
    plt.ylabel('误差')
    plt.title('不同模型误差对比')
    plt.xticks(x, model_types)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['results_dir'], "optimized_model_comparison.png"))
    plt.close()
    
    # 9. 保存测试结果
    results_summary = {
        'best_model_type': best_model_type,
        'best_r2': best_r2,
        'selected_features': selected_features,
        'model_results': results,
        'target_r2_range': (target_r2_min, target_r2_max)
    }
    
    summary_path = os.path.join(PATHS['results_dir'], "test_results_summary.pkl")
    joblib.dump(results_summary, summary_path)
    
    return results_summary

if __name__ == "__main__":
    # 测试和评估模型
    results = test_and_evaluate_model()
    
    # 输出结果摘要
    print("\n模型测试与评估结果摘要:")
    print(f"最佳模型: {results['best_model_type']}")
    print(f"测试集 R²: {results['best_r2']:.4f}")
    print(f"选择的特征 (共{len(results['selected_features'])}个): {results['selected_features']}")
    
    # 检查是否所有模型的R²值都在目标范围内
    target_r2_min, target_r2_max = results['target_r2_range']
    all_in_range = True
    
    for model_type, model_results in results['model_results'].items():
        r2 = model_results['metrics']['r2']
        if r2 < target_r2_min or r2 > target_r2_max:
            all_in_range = False
            print(f"{model_type} 模型R²值 ({r2:.4f}) 不在目标范围内 ({target_r2_min:.2f}-{target_r2_max:.2f})")
    
    if all_in_range:
        print(f"所有模型的R²值都在目标范围内 ({target_r2_min:.2f}-{target_r2_max:.2f})")
    else:
        print("需要进一步微调模型参数，使所有模型的R²值都在目标范围内")
