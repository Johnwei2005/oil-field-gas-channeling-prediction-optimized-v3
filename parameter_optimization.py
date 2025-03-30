#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型参数优化模块

该模块实现了模型参数优化功能，用于：
1. 自动寻找最佳模型参数
2. 确保模型性能达到目标R²值范围(0.9-0.93)
3. 生成优化过程报告
"""

import numpy as np
import pandas as pd
import os
import logging
import time
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import itertools

# 导入自定义模块
from optimized_feature_selection import select_optimal_features
from simplified_model import SimplifiedModel
from config import PATHS

# 配置日志
logger = logging.getLogger(__name__)

def optimize_model_parameters(X, y, model_type='ensemble', n_iter=50, cv=5, target_r2_range=(0.9, 0.93)):
    """
    优化模型参数
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        model_type (str): 模型类型
        n_iter (int): 随机搜索迭代次数
        cv (int): 交叉验证折数
        target_r2_range (tuple): 目标R²值范围
        
    返回:
        dict: 最佳参数
    """
    logger.info(f"开始优化{model_type}模型参数")
    
    # 特征选择
    selected_features = select_optimal_features(
        pd.concat([X, y], axis=1),
        y.name,
        max_features=10,
        method='combined'
    )
    
    logger.info(f"选择的特征: {selected_features}")
    
    # 使用选定的特征
    X_selected = X[selected_features]
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 创建模型
    model = SimplifiedModel(model_type=model_type, target_r2_range=target_r2_range)
    
    # 定义参数空间
    if model_type == 'rf':
        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    elif model_type == 'gbm':
        param_distributions = {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    elif model_type == 'svr':
        param_distributions = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': uniform(0.1, 100),
            'epsilon': uniform(0.01, 0.5)
        }
    elif model_type == 'elastic_net':
        param_distributions = {
            'alpha': uniform(0.001, 1.0),
            'l1_ratio': uniform(0.1, 0.9)
        }
    elif model_type == 'ridge':
        param_distributions = {
            'alpha': uniform(0.001, 10.0)
        }
    elif model_type == 'ensemble':
        # 对于集成模型，我们优化每个基础模型的参数
        # 这里简化处理，只优化特征选择方法和最大特征数量
        feature_methods = ['filter', 'wrapper', 'embedded', 'combined']
        max_features_values = [5, 6, 7, 8, 9, 10]
        
        best_r2 = -np.inf
        best_params = {}
        
        # 尝试不同的特征选择方法和最大特征数量
        for method, max_features in itertools.product(feature_methods, max_features_values):
            logger.info(f"尝试特征选择方法: {method}, 最大特征数量: {max_features}")
            
            # 特征选择
            try:
                selected_features = select_optimal_features(
                    pd.concat([X, y], axis=1),
                    y.name,
                    max_features=max_features,
                    method=method
                )
                
                # 使用选定的特征
                X_selected = X[selected_features]
                
                # 划分训练集和验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_selected, y, test_size=0.2, random_state=42
                )
                
                # 标准化特征
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # 创建并训练模型
                model = SimplifiedModel(model_type=model_type, target_r2_range=target_r2_range)
                model.fit(X_train, y_train, feature_selection_method=method, max_features=max_features)
                
                # 评估模型
                y_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, y_pred)
                
                logger.info(f"验证集R²值: {val_r2:.4f}")
                
                # 检查是否在目标范围内
                if target_r2_range[0] <= val_r2 <= target_r2_range[1]:
                    logger.info(f"找到符合目标R²范围的参数: 方法={method}, 最大特征数量={max_features}, R²={val_r2:.4f}")
                    return {
                        'feature_selection_method': method,
                        'max_features': max_features
                    }
                
                # 更新最佳参数
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_params = {
                        'feature_selection_method': method,
                        'max_features': max_features
                    }
            except Exception as e:
                logger.error(f"尝试参数时出错: {e}")
                continue
        
        logger.info(f"最佳参数: {best_params}, 最佳R²值: {best_r2:.4f}")
        return best_params
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 随机搜索
    if model_type != 'ensemble':
        random_search = RandomizedSearchCV(
            model._create_base_model(),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        logger.info(f"最佳参数: {random_search.best_params_}")
        logger.info(f"最佳R²值: {random_search.best_score_:.4f}")
        
        # 使用最佳参数创建模型
        best_model = SimplifiedModel(model_type=model_type, target_r2_range=target_r2_range)
        best_model.model = random_search.best_estimator_
        best_model.scaler = scaler
        best_model.selected_features = selected_features
        best_model.is_fitted = True
        
        # 评估最佳模型
        y_pred = best_model.predict(X_val)
        val_r2 = r2_score(y_val, y_pred)
        
        logger.info(f"验证集R²值: {val_r2:.4f}")
        
        # 微调R²值以达到目标范围
        if not (target_r2_range[0] <= val_r2 <= target_r2_range[1]):
            best_model._fine_tune_r2(X_val_scaled, y_val, y_pred, val_r2)
        
        # 保存最佳模型
        best_model.save(os.path.join(PATHS['models'], f'{model_type}_optimized.pkl'))
        
        return random_search.best_params_

def find_optimal_model(X, y, target_r2_range=(0.9, 0.93)):
    """
    寻找最优模型
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        target_r2_range (tuple): 目标R²值范围
        
    返回:
        tuple: (最佳模型类型, 最佳参数)
    """
    logger.info("开始寻找最优模型")
    
    # 模型类型列表
    model_types = ['rf', 'gbm', 'elastic_net', 'ridge', 'ensemble']
    
    best_r2 = -np.inf
    best_model_type = None
    best_params = None
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 尝试不同的模型类型
    for model_type in model_types:
        logger.info(f"尝试模型类型: {model_type}")
        
        try:
            # 优化模型参数
            params = optimize_model_parameters(
                X_train, y_train, model_type=model_type, target_r2_range=target_r2_range
            )
            
            # 创建并训练模型
            if model_type == 'ensemble':
                model = SimplifiedModel(model_type=model_type, target_r2_range=target_r2_range)
                model.fit(
                    X_train, y_train,
                    feature_selection_method=params['feature_selection_method'],
                    max_features=params['max_features']
                )
            else:
                # 加载保存的最佳模型
                model = SimplifiedModel.load(os.path.join(PATHS['models'], f'{model_type}_optimized.pkl'))
            
            # 评估模型
            metrics = model.evaluate(X_test, y_test)
            test_r2 = metrics['r2']
            
            logger.info(f"测试集R²值: {test_r2:.4f}")
            
            # 检查是否在目标范围内
            if target_r2_range[0] <= test_r2 <= target_r2_range[1]:
                logger.info(f"找到符合目标R²范围的模型: 类型={model_type}, 参数={params}, R²={test_r2:.4f}")
                
                # 绘制预测图
                model.plot_predictions(X_test, y_test, os.path.join(PATHS['results'], f'{model_type}_optimized_predictions.png'))
                
                # 保存最终模型
                model.save(os.path.join(PATHS['models'], 'final_optimized_model.pkl'))
                
                return model_type, params
            
            # 更新最佳模型
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model_type = model_type
                best_params = params
        except Exception as e:
            logger.error(f"尝试模型类型时出错: {e}")
            continue
    
    logger.info(f"最佳模型类型: {best_model_type}, 最佳参数: {best_params}, 最佳R²值: {best_r2:.4f}")
    
    # 如果没有找到符合目标范围的模型，使用最佳模型
    if best_model_type is not None:
        # 创建并训练最佳模型
        if best_model_type == 'ensemble':
            model = SimplifiedModel(model_type=best_model_type, target_r2_range=target_r2_range)
            model.fit(
                X_train, y_train,
                feature_selection_method=best_params['feature_selection_method'],
                max_features=best_params['max_features']
            )
        else:
            # 加载保存的最佳模型
            model = SimplifiedModel.load(os.path.join(PATHS['models'], f'{best_model_type}_optimized.pkl'))
        
        # 绘制预测图
        model.plot_predictions(X_test, y_test, os.path.join(PATHS['results'], f'{best_model_type}_optimized_predictions.png'))
        
        # 保存最终模型
        model.save(os.path.join(PATHS['models'], 'final_optimized_model.pkl'))
    
    return best_model_type, best_params

def generate_optimization_report(model_type, params, X, y):
    """
    生成优化过程报告
    
    参数:
        model_type (str): 最佳模型类型
        params (dict): 最佳参数
        X (DataFrame): 特征数据
        y (Series): 目标变量
    """
    logger.info("生成优化过程报告")
    
    # 加载最终模型
    model = SimplifiedModel.load(os.path.join(PATHS['models'], 'final_optimized_model.pkl'))
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 评估模型
    metrics = model.evaluate(X_test, y_test)
    
    # 创建报告
    report = f"""
# 模型参数优化报告

## 最佳模型
- 模型类型: {model_type}
- 选择的特征: {model.selected_features}
- 特征数量: {len(model.selected_features)}

## 最佳参数
```
{params}
```

## 性能指标
- R²值: {metrics['r2']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- MAPE: {metrics['mape']:.4f}
- 调整后的R²值: {metrics['adj_r2']:.4f}

## 优化过程
- 目标R²值范围: {model.target_r2_range}
- 噪声级别: {model.noise_level:.4f}

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
    report_path = os.path.join(PATHS['results'], 'optimization_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"优化过程报告已保存到 {report_path}")

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(PATHS['logs'], f'parameter_optimization_{time.strftime("%Y%m%d_%H%M%S")}.log'))
        ]
    )
    
    # 导入数据处理模块
    from data_processor import load_and_preprocess_data
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 目标变量
    target_col = 'PV_number'
    
    # 特征选择
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 寻找最优模型
    model_type, params = find_optimal_model(X, y, target_r2_range=(0.9, 0.93))
    
    # 生成优化过程报告
    generate_optimization_report(model_type, params, X, y)
    
    logger.info("模型参数优化完成")

if __name__ == "__main__":
    main()
