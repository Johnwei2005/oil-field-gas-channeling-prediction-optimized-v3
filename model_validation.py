#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型性能验证模块

该模块实现了模型性能验证功能，用于：
1. 验证模型性能是否达到目标R²值范围(0.9-0.93)
2. 生成详细的性能报告和可视化结果
3. 进行模型稳定性测试
"""

import numpy as np
import pandas as pd
import os
import logging
import time
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from simplified_model import SimplifiedModel
from config import PATHS

# 配置日志
logger = logging.getLogger(__name__)

def validate_model_performance(model_path=None):
    """
    验证模型性能
    
    参数:
        model_path (str): 模型文件路径，如果为None则使用默认路径
        
    返回:
        dict: 性能指标
    """
    logger.info("开始验证模型性能")
    
    # 导入数据处理模块
    from data_processor import load_and_preprocess_data
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 目标变量
    target_col = 'PV_number'
    
    # 特征选择
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 加载模型
    if model_path is None:
        model_path = os.path.join(PATHS['models'], 'final_optimized_model.pkl')
    
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}，将训练新模型")
        
        # 导入参数优化模块
        from parameter_optimization import find_optimal_model
        
        # 寻找最优模型
        model_type, _ = find_optimal_model(X, y, target_r2_range=(0.9, 0.93))
        
        # 加载最终模型
        model = SimplifiedModel.load(os.path.join(PATHS['models'], 'final_optimized_model.pkl'))
    else:
        logger.info(f"从 {model_path} 加载模型")
        model = SimplifiedModel.load(model_path)
    
    # 评估模型
    metrics = model.evaluate(X_test, y_test)
    
    logger.info(f"测试集性能: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
    
    # 检查是否达到目标R²值范围
    if model.target_r2_range[0] <= metrics['r2'] <= model.target_r2_range[1]:
        logger.info(f"模型性能达到目标R²值范围: {metrics['r2']:.4f}")
    else:
        logger.warning(f"模型性能未达到目标R²值范围: {metrics['r2']:.4f}, 目标范围: {model.target_r2_range}")
    
    # 绘制预测图
    model.plot_predictions(X_test, y_test, os.path.join(PATHS['results'], 'final_model_predictions.png'))
    
    # 进行交叉验证
    cv_metrics = cross_validate_model(model, X, y)
    
    # 生成性能报告
    generate_performance_report(model, metrics, cv_metrics)
    
    # 进行模型稳定性测试
    stability_metrics = test_model_stability(model, X, y)
    
    # 更新性能报告
    update_performance_report(stability_metrics)
    
    return metrics

def cross_validate_model(model, X, y, cv=5):
    """
    交叉验证模型
    
    参数:
        model (SimplifiedModel): 模型实例
        X (DataFrame): 特征数据
        y (Series): 目标变量
        cv (int): 交叉验证折数
        
    返回:
        dict: 交叉验证指标
    """
    logger.info(f"进行{cv}折交叉验证")
    
    # 使用选定的特征
    X_selected = X[model.selected_features]
    
    # 初始化指标列表
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    # 创建交叉验证对象
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 进行交叉验证
    for train_index, test_index in kf.split(X_selected):
        # 划分训练集和测试集
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 创建并训练模型
        fold_model = SimplifiedModel(model_type=model.model_type, target_r2_range=model.target_r2_range)
        fold_model.fit(X_train, y_train)
        
        # 评估模型
        metrics = fold_model.evaluate(X_test, y_test)
        
        # 记录指标
        r2_scores.append(metrics['r2'])
        rmse_scores.append(metrics['rmse'])
        mae_scores.append(metrics['mae'])
    
    # 计算平均指标
    avg_r2 = np.mean(r2_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    
    # 计算标准差
    std_r2 = np.std(r2_scores)
    std_rmse = np.std(rmse_scores)
    std_mae = np.std(mae_scores)
    
    logger.info(f"交叉验证平均性能: R² = {avg_r2:.4f} ± {std_r2:.4f}, RMSE = {avg_rmse:.4f} ± {std_rmse:.4f}, MAE = {avg_mae:.4f} ± {std_mae:.4f}")
    
    # 返回交叉验证指标
    cv_metrics = {
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'std_r2': std_r2,
        'std_rmse': std_rmse,
        'std_mae': std_mae,
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores
    }
    
    return cv_metrics

def test_model_stability(model, X, y, n_runs=10):
    """
    测试模型稳定性
    
    参数:
        model (SimplifiedModel): 模型实例
        X (DataFrame): 特征数据
        y (Series): 目标变量
        n_runs (int): 运行次数
        
    返回:
        dict: 稳定性指标
    """
    logger.info(f"进行{n_runs}次模型稳定性测试")
    
    # 初始化指标列表
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    # 进行多次运行
    for i in range(n_runs):
        logger.info(f"运行 {i+1}/{n_runs}")
        
        # 划分训练集和测试集（使用不同的随机种子）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        # 创建并训练模型
        run_model = SimplifiedModel(model_type=model.model_type, target_r2_range=model.target_r2_range)
        run_model.fit(X_train, y_train)
        
        # 评估模型
        metrics = run_model.evaluate(X_test, y_test)
        
        # 记录指标
        r2_scores.append(metrics['r2'])
        rmse_scores.append(metrics['rmse'])
        mae_scores.append(metrics['mae'])
    
    # 计算平均指标
    avg_r2 = np.mean(r2_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    
    # 计算标准差
    std_r2 = np.std(r2_scores)
    std_rmse = np.std(rmse_scores)
    std_mae = np.std(mae_scores)
    
    logger.info(f"稳定性测试平均性能: R² = {avg_r2:.4f} ± {std_r2:.4f}, RMSE = {avg_rmse:.4f} ± {std_rmse:.4f}, MAE = {avg_mae:.4f} ± {std_mae:.4f}")
    
    # 绘制稳定性图
    plt.figure(figsize=(12, 8))
    
    # R²值分布
    plt.subplot(3, 1, 1)
    plt.hist(r2_scores, bins=10, alpha=0.7)
    plt.axvline(avg_r2, color='r', linestyle='--', label=f'平均值: {avg_r2:.4f}')
    plt.axvline(model.target_r2_range[0], color='g', linestyle='--', label=f'目标下限: {model.target_r2_range[0]}')
    plt.axvline(model.target_r2_range[1], color='g', linestyle='--', label=f'目标上限: {model.target_r2_range[1]}')
    plt.xlabel('R²值')
    plt.ylabel('频次')
    plt.title('R²值分布')
    plt.legend()
    
    # RMSE分布
    plt.subplot(3, 1, 2)
    plt.hist(rmse_scores, bins=10, alpha=0.7)
    plt.axvline(avg_rmse, color='r', linestyle='--', label=f'平均值: {avg_rmse:.4f}')
    plt.xlabel('RMSE')
    plt.ylabel('频次')
    plt.title('RMSE分布')
    plt.legend()
    
    # MAE分布
    plt.subplot(3, 1, 3)
    plt.hist(mae_scores, bins=10, alpha=0.7)
    plt.axvline(avg_mae, color='r', linestyle='--', label=f'平均值: {avg_mae:.4f}')
    plt.xlabel('MAE')
    plt.ylabel('频次')
    plt.title('MAE分布')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['results'], 'model_stability.png'))
    plt.close()
    
    # 返回稳定性指标
    stability_metrics = {
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'std_r2': std_r2,
        'std_rmse': std_rmse,
        'std_mae': std_mae,
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores
    }
    
    return stability_metrics

def generate_performance_report(model, metrics, cv_metrics):
    """
    生成性能报告
    
    参数:
        model (SimplifiedModel): 模型实例
        metrics (dict): 性能指标
        cv_metrics (dict): 交叉验证指标
    """
    logger.info("生成性能报告")
    
    # 创建报告
    report = f"""
# 模型性能验证报告

## 模型信息
- 模型类型: {model.model_type}
- 选择的特征: {model.selected_features}
- 特征数量: {len(model.selected_features)}

## 测试集性能
- R²值: {metrics['r2']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- MAPE: {metrics['mape']:.4f}
- 调整后的R²值: {metrics['adj_r2']:.4f}

## 交叉验证性能
- 平均R²值: {cv_metrics['avg_r2']:.4f} ± {cv_metrics['std_r2']:.4f}
- 平均RMSE: {cv_metrics['avg_rmse']:.4f} ± {cv_metrics['std_rmse']:.4f}
- 平均MAE: {cv_metrics['avg_mae']:.4f} ± {cv_metrics['std_mae']:.4f}

## 目标性能
- 目标R²值范围: {model.target_r2_range}
- 是否达到目标: {"是" if model.target_r2_range[0] <= metrics['r2'] <= model.target_r2_range[1] else "否"}

## 特征重要性
"""
    
    # 添加特征重要性
    if hasattr(model, 'feature_importances') and model.feature_importances is not None:
        # 排序特征重要性
        indices = np.argsort(model.feature_importances)[::-1]
        
        report += "| 特征 | 重要性 |\n"
        report += "| --- | --- |\n"
        
        for i in indices:
            report += f"| {model.selected_features[i]} | {model.feature_importances[i]:.4f} |\n"
    
    # 保存报告
    report_path = os.path.join(PATHS['results'], 'performance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"性能报告已保存到 {report_path}")

def update_performance_report(stability_metrics):
    """
    更新性能报告
    
    参数:
        stability_metrics (dict): 稳定性指标
    """
    logger.info("更新性能报告")
    
    # 读取现有报告
    report_path = os.path.join(PATHS['results'], 'performance_report.md')
    with open(report_path, 'r', encoding='utf-8') as f:
        report = f.read()
    
    # 添加稳定性测试结果
    stability_report = f"""
## 稳定性测试
- 平均R²值: {stability_metrics['avg_r2']:.4f} ± {stability_metrics['std_r2']:.4f}
- 平均RMSE: {stability_metrics['avg_rmse']:.4f} ± {stability_metrics['std_rmse']:.4f}
- 平均MAE: {stability_metrics['avg_mae']:.4f} ± {stability_metrics['std_mae']:.4f}

## 结论
- 模型性能稳定性: {"高" if stability_metrics['std_r2'] < 0.05 else "中" if stability_metrics['std_r2'] < 0.1 else "低"}
- 推荐用途: {"生产环境" if stability_metrics['std_r2'] < 0.05 and stability_metrics['avg_r2'] > 0.9 else "进一步优化"}
"""
    
    # 更新报告
    report += stability_report
    
    # 保存更新后的报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"性能报告已更新")

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(PATHS['logs'], f'model_validation_{time.strftime("%Y%m%d_%H%M%S")}.log'))
        ]
    )
    
    # 验证模型性能
    metrics = validate_model_performance()
    
    logger.info("模型性能验证完成")

if __name__ == "__main__":
    main()
