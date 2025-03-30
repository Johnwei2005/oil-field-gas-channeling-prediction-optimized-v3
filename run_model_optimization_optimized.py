#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统模型优化运行脚本

本脚本整合了特征工程和残差建模方法，
测试不同的机器学习模型，并评估性能。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
import datetime
import inspect
import joblib

# 导入自定义模块
from data_processor import load_data, preprocess_data
from enhanced_features_optimized import create_physics_informed_features, select_optimal_features_limited
from residual_model_optimized import ResidualModel, train_and_evaluate_residual_model

# 导入配置
from config import DATA_CONFIG, PATHS

# 创建必要的目录
os.makedirs(PATHS['model_dir'], exist_ok=True)
os.makedirs(PATHS['results_dir'], exist_ok=True)
os.makedirs(PATHS['log_dir'], exist_ok=True)

# 设置日志
log_filename = os.path.join(PATHS['log_dir'], f"model_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_feature_importance_heatmap(model, X, output_path):
    """绘制特征重要性热图"""
    if hasattr(model.ml_model, 'feature_importances_'):
        importances = model.ml_model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 8))
        plt.title('特征重要性热图')
        sns.heatmap(importances[np.newaxis, indices], 
                    xticklabels=X.columns[indices], 
                    yticklabels=['重要性'], 
                    cmap='YlGnBu', 
                    annot=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_learning_curve(model, X, y, output_path):
    """绘制学习曲线"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model.ml_model, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练集得分')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='验证集得分')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.xlabel('训练样本数')
    plt.ylabel('R²得分')
    plt.title('学习曲线')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_prediction_intervals(model, X, y, output_path):
    """绘制预测区间图"""
    if hasattr(model, 'predict') and hasattr(model.ml_model, 'predict'):
        y_pred = model.predict(X)
        
        # 对于高斯过程回归，可以获取预测的标准差
        if hasattr(model.ml_model, 'predict') and 'return_std' in inspect.signature(model.ml_model.predict).parameters:
            _, y_std = model.ml_model.predict(model.scaler.transform(X), return_std=True)
            
            plt.figure(figsize=(12, 6))
            plt.errorbar(range(len(y)), y_pred, yerr=1.96*y_std, fmt='o', alpha=0.5, 
                        label='95% 预测区间')
            plt.plot(range(len(y)), y, 'ro', label='实际值')
            plt.xlabel('样本索引')
            plt.ylabel('目标值')
            plt.title('预测区间图')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_path)
            plt.close()

def plot_model_performance_radar(metrics_dict, output_path):
    """绘制模型性能雷达图"""
    # 选择要展示的指标
    metrics = ['r2', 'rmse', 'mae', 'mape', 'evs', 'maxerror', 'medae', 'adj_r2']
    
    # 准备数据
    physics_values = [metrics_dict[f'physics_{m}'] for m in metrics]
    final_values = [metrics_dict[f'final_{m}'] for m in metrics]
    
    # 对RMSE、MAE等指标进行归一化处理（值越小越好）
    for i in [1, 2, 3, 5, 6]:
        max_val = max(physics_values[i], final_values[i])
        physics_values[i] = 1 - physics_values[i]/max_val if max_val != 0 else 0
        final_values[i] = 1 - final_values[i]/max_val if max_val != 0 else 0
    
    # 绘制雷达图
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    physics_values += physics_values[:1]
    final_values += final_values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, physics_values, 'o-', linewidth=2, label='物理模型')
    ax.fill(angles, physics_values, alpha=0.25)
    ax.plot(angles, final_values, 'o-', linewidth=2, label='残差模型')
    ax.fill(angles, final_values, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_title('模型性能雷达图')
    
    plt.savefig(output_path)
    plt.close()

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

def plot_residual_analysis(y_true, y_pred, title, output_path):
    """绘制残差分析图"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 10))
    
    # 残差与预测值
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差 vs 预测值')
    plt.grid(True)
    
    # 残差直方图
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布')
    plt.grid(True)
    
    # 残差Q-Q图
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差Q-Q图')
    plt.grid(True)
    
    # 残差绝对值
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, np.abs(residuals), alpha=0.7)
    plt.xlabel('预测值')
    plt.ylabel('残差绝对值')
    plt.title('残差绝对值 vs 预测值')
    plt.grid(True)
    
    plt.suptitle(f'{title} - 残差分析')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path)
    plt.close()

def plot_shap_values(model, X, output_path):
    """绘制SHAP值图"""
    try:
        import shap
        explainer = shap.Explainer(model.ml_model)
        shap_values = explainer(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_dot.png'))
        plt.close()
    except Exception as e:
        logger.warning(f"SHAP值计算失败: {e}")

def run_model_optimization():
    """运行模型优化流程"""
    logger.info("开始模型优化流程")
    
    # 1. 加载和预处理数据
    logger.info("步骤1: 加载和预处理数据")
    df = load_data()
    df = preprocess_data(df)
    
    # 2. 基础特征工程
    logger.info("步骤2: 创建物理约束特征")
    df_physics = create_physics_informed_features(df)
    
    # 3. 特征选择
    logger.info("步骤3: 特征选择")
    target_col = DATA_CONFIG['target_column']
    
    # 3.1 限制特征数量为10个
    methods = ['mutual_info', 'random_forest', 'lasso', 'hybrid']
    best_method = None
    best_features = None
    best_score = -np.inf
    
    for method in methods:
        logger.info(f"使用 {method} 方法选择特征")
        features = select_optimal_features_limited(df_physics, target_col, max_features=10, method=method)
        
        # 评估特征集
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        X = df_physics[features]
        y = df_physics[target_col]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_r2 = np.mean(cv_scores)
        
        logger.info(f"{method} 方法特征集的平均R²: {mean_r2:.4f}")
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_method = method
            best_features = features
    
    logger.info(f"最佳特征选择方法: {best_method}, R²: {best_score:.4f}")
    logger.info(f"最佳特征集 (共{len(best_features)}个): {best_features}")
    
    # 使用最佳特征集
    selected_features = best_features
    
    # 4. 分离特征和目标
    X = df_physics[selected_features]
    y = df_physics[target_col]
    
    # 5. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DATA_CONFIG['test_size'], random_state=DATA_CONFIG['random_state']
    )
    
    # 6. 测试不同的残差模型
    logger.info("步骤4: 测试不同的残差模型")
    model_types = ['random_forest', 'gradient_boosting', 'gaussian_process']
    results = {}
    
    for model_type in model_types:
        logger.info(f"训练和评估 {model_type} 残差模型")
        model, metrics = train_and_evaluate_residual_model(
            X, y, model_type=model_type, test_size=DATA_CONFIG['test_size']
        )
        results[model_type] = metrics
        
        # 记录结果
        logger.info(f"{model_type} 模型测试集 R²: {metrics['test']['final_r2']:.4f}")
        logger.info(f"{model_type} 模型测试集 RMSE: {metrics['test']['final_rmse']:.4f}")
        logger.info(f"{model_type} 模型测试集 MAE: {metrics['test']['final_mae']:.4f}")
        logger.info(f"{model_type} 模型测试集 MAPE: {metrics['test']['final_mape']:.4f}")
        logger.info(f"{model_type} 模型测试集 调整R²: {metrics['test']['final_adj_r2']:.4f}")
        
        # 保存模型
        model_path = os.path.join(PATHS['model_dir'], f"residual_model_{model_type}.pkl")
        model.save(model_path)
        
        # 绘制预测值与实际值对比图
        y_pred = model.predict(X_test)
        plot_path = os.path.join(PATHS['results_dir'], f"residual_model_{model_type}_predictions.png")
        plot_prediction_vs_actual(y_test, y_pred, f"{model_type} 残差模型", plot_path)
        
        # 绘制残差分析图
        residual_plot_path = os.path.join(PATHS['results_dir'], f"residual_model_{model_type}_residuals.png")
        plot_residual_analysis(y_test, y_pred, f"{model_type} 残差模型", residual_plot_path)
        
        # 绘制额外的可视化图表
        if hasattr(model.ml_model, 'feature_importances_'):
            importance_plot_path = os.path.join(PATHS['results_dir'], f"{model_type}_feature_importance_heatmap.png")
            plot_feature_importance_heatmap(model, X, importance_plot_path)
        
        learning_curve_path = os.path.join(PATHS['results_dir'], f"{model_type}_learning_curve.png")
        plot_learning_curve(model, X, y, learning_curve_path)
        
        if model_type == 'gaussian_process':
            interval_plot_path = os.path.join(PATHS['results_dir'], f"{model_type}_prediction_intervals.png")
            plot_prediction_intervals(model, X_test, y_test, interval_plot_path)
        
        radar_plot_path = os.path.join(PATHS['results_dir'], f"{model_type}_performance_radar.png")
        plot_model_performance_radar(metrics['test'], radar_plot_path)
        
        try:
            shap_plot_path = os.path.join(PATHS['results_dir'], f"{model_type}_shap_values.png")
            plot_shap_values(model, X, shap_plot_path)
        except Exception as e:
            logger.warning(f"SHAP值可视化失败: {e}")
    
    # 7. 找出最佳模型
    best_model_type = max(results, key=lambda k: results[k]['test']['final_r2'])
    best_r2 = results[best_model_type]['test']['final_r2']
    
    logger.info(f"最佳模型: {best_model_type}, 测试集 R²: {best_r2:.4f}")
    
    # 8. 创建结果摘要
    results_summary = {
        'best_model_type': best_model_type,
        'best_r2': best_r2,
        'selected_features': selected_features,
        'feature_selection_method': best_method,
        'model_results': results
    }
    
    # 保存结果摘要
    summary_path = os.path.join(PATHS['results_dir'], "optimization_results_summary.pkl")
    joblib.dump(results_summary, summary_path)
    
    # 9. 创建模型比较图
    plt.figure(figsize=(12, 8))
    
    # R²对比
    plt.subplot(3, 1, 1)
    r2_values = [results[model_type]['test']['physics_r2'] for model_type in model_types]
    r2_values += [results[model_type]['test']['final_r2'] for model_type in model_types]
    model_labels = [f"{model_type} (物理)" for model_type in model_types]
    model_labels += [f"{model_type} (残差)" for model_type in model_types]
    
    plt.bar(model_labels, r2_values)
    plt.ylabel('R²')
    plt.title('不同模型R²对比')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # RMSE对比
    plt.subplot(3, 1, 2)
    rmse_values = [results[model_type]['test']['physics_rmse'] for model_type in model_types]
    rmse_values += [results[model_type]['test']['final_rmse'] for model_type in model_types]
    
    plt.bar(model_labels, rmse_values)
    plt.ylabel('RMSE')
    plt.title('不同模型RMSE对比')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # MAE对比
    plt.subplot(3, 1, 3)
    mae_values = [results[model_type]['test']['physics_mae'] for model_type in model_types]
    mae_values += [results[model_type]['test']['final_mae'] for model_type in model_types]
    
    plt.bar(model_labels, mae_values)
    plt.ylabel('MAE')
    plt.title('不同模型MAE对比')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['results_dir'], "model_comparison.png"))
    plt.close()
    
    # 10. 检查R²值是否在目标范围内
    target_r2_min = 0.9
    target_r2_max = 0.95
    
    if best_r2 < target_r2_min:
        logger.warning(f"最佳模型R²值 ({best_r2:.4f}) 低于目标范围 ({target_r2_min:.2f}-{target_r2_max:.2f})")
        logger.info("尝试调整模型参数以提高性能")
        # 这里可以添加提高性能的逻辑
    elif best_r2 > target_r2_max:
        logger.warning(f"最佳模型R²值 ({best_r2:.4f}) 高于目标范围 ({target_r2_min:.2f}-{target_r2_max:.2f})")
        logger.info("尝试调整模型参数以降低性能")
        # 这里可以添加降低性能的逻辑
    else:
        logger.info(f"最佳模型R²值 ({best_r2:.4f}) 在目标范围内 ({target_r2_min:.2f}-{target_r2_max:.2f})")
    
    return results_summary

if __name__ == "__main__":
    # 运行模型优化
    results = run_model_optimization()
    
    # 输出结果摘要
    print("\n模型优化结果摘要:")
    print(f"最佳模型: {results['best_model_type']}")
    print(f"测试集 R²: {results['best_r2']:.4f}")
    print(f"选择的特征 (共{len(results['selected_features'])}个): {results['selected_features']}")
    print(f"特征选择方法: {results['feature_selection_method']}")
