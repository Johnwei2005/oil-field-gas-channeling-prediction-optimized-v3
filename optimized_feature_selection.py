#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化特征选择模块

该模块实现了优化的特征选择方法，确保：
1. 特征数量不超过10个
2. 预测目标不会被加入特征列表
3. 选择最具预测能力的特征组合
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# 配置日志
logger = logging.getLogger(__name__)

def select_optimal_features(df, target_col, max_features=10, method='combined'):
    """
    优化的特征选择方法
    
    参数:
        df (DataFrame): 输入数据
        target_col (str): 目标列名
        max_features (int): 最大特征数量，默认为10
        method (str): 特征选择方法，可选 'filter', 'wrapper', 'embedded', 'combined'
        
    返回:
        list: 选择的特征列表
    """
    logger.info(f"使用 {method} 方法选择最优特征，最大特征数量: {max_features}")
    
    # 确保目标列不在特征中
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 处理缺失值和异常值
    X = handle_missing_and_outliers(X)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 根据选择的方法进行特征选择
    if method == 'filter':
        selected_features = filter_method(X_scaled, y, max_features)
    elif method == 'wrapper':
        selected_features = wrapper_method(X_scaled, y, max_features)
    elif method == 'embedded':
        selected_features = embedded_method(X_scaled, y, max_features)
    elif method == 'combined':
        selected_features = combined_method(X_scaled, y, max_features)
    else:
        raise ValueError(f"不支持的特征选择方法: {method}")
    
    # 确保特征数量不超过最大限制
    if len(selected_features) > max_features:
        selected_features = selected_features[:max_features]
    
    logger.info(f"选择了 {len(selected_features)} 个特征: {selected_features}")
    return selected_features

def handle_missing_and_outliers(X):
    """
    处理缺失值和异常值
    
    参数:
        X (DataFrame): 特征数据
        
    返回:
        DataFrame: 处理后的特征数据
    """
    # 复制数据
    X_processed = X.copy()
    
    # 处理缺失值
    for col in X_processed.columns:
        if X_processed[col].isnull().sum() > 0:
            # 对数值列使用中位数填充
            if np.issubdtype(X_processed[col].dtype, np.number):
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            else:
                # 对非数值列使用众数填充
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
    
    # 处理异常值
    for col in X_processed.columns:
        if np.issubdtype(X_processed[col].dtype, np.number):
            # 使用百分位数方法处理异常值
            lower_bound = X_processed[col].quantile(0.01)
            upper_bound = X_processed[col].quantile(0.99)
            X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)
    
    return X_processed

def filter_method(X, y, max_features):
    """
    基于过滤的特征选择方法
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        max_features (int): 最大特征数量
        
    返回:
        list: 选择的特征列表
    """
    # 使用互信息
    mi_selector = SelectKBest(mutual_info_regression, k=min(max_features, len(X.columns)))
    mi_selector.fit(X, y)
    mi_scores = mi_selector.scores_
    mi_features = X.columns[mi_selector.get_support()]
    
    # 计算相关系数
    corr = pd.DataFrame(X.corrwith(y).abs().sort_values(ascending=False))
    corr.columns = ['correlation']
    corr_features = corr.index[:max_features].tolist()
    
    # 结合两种方法
    combined_features = list(set(mi_features) | set(corr_features))
    
    # 如果特征数量超过最大限制，选择互信息得分最高的特征
    if len(combined_features) > max_features:
        feature_scores = {}
        for feature in combined_features:
            feature_scores[feature] = mi_scores[list(X.columns).index(feature)]
        
        selected_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_features = [feature for feature, _ in selected_features]
    else:
        selected_features = combined_features
    
    return selected_features

def wrapper_method(X, y, max_features):
    """
    基于包装的特征选择方法
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        max_features (int): 最大特征数量
        
    返回:
        list: 选择的特征列表
    """
    # 使用前向特征选择
    selected_features = []
    remaining_features = list(X.columns)
    
    # 交叉验证设置
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 基础模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 前向特征选择
    for _ in range(min(max_features, len(X.columns))):
        best_score = -np.inf
        best_feature = None
        
        for feature in remaining_features:
            # 当前特征集合
            current_features = selected_features + [feature]
            
            # 评估当前特征集合
            scores = cross_val_score(model, X[current_features], y, cv=kf, scoring='r2')
            avg_score = np.mean(scores)
            
            # 更新最佳特征
            if avg_score > best_score:
                best_score = avg_score
                best_feature = feature
        
        # 添加最佳特征
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
    
    return selected_features

def embedded_method(X, y, max_features):
    """
    基于嵌入的特征选择方法
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        max_features (int): 最大特征数量
        
    返回:
        list: 选择的特征列表
    """
    # 使用Lasso回归
    alpha = 0.01
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X, y)
    
    # 获取特征重要性
    lasso_importance = np.abs(lasso.coef_)
    lasso_features = X.columns[lasso_importance > 0]
    
    # 使用Ridge回归
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X, y)
    
    # 获取特征重要性
    ridge_importance = np.abs(ridge.coef_)
    ridge_indices = np.argsort(ridge_importance)[::-1][:max_features]
    ridge_features = X.columns[ridge_indices]
    
    # 结合两种方法
    combined_features = list(set(lasso_features) | set(ridge_features))
    
    # 如果特征数量超过最大限制，选择Lasso系数最大的特征
    if len(combined_features) > max_features:
        feature_scores = {}
        for feature in combined_features:
            feature_scores[feature] = lasso_importance[list(X.columns).index(feature)]
        
        selected_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_features = [feature for feature, _ in selected_features]
    else:
        selected_features = combined_features
    
    return selected_features

def combined_method(X, y, max_features):
    """
    结合多种方法的特征选择
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        max_features (int): 最大特征数量
        
    返回:
        list: 选择的特征列表
    """
    # 获取各种方法的特征
    filter_features = filter_method(X, y, max_features)
    embedded_features = embedded_method(X, y, max_features)
    
    # 结合特征
    combined_features = list(set(filter_features) | set(embedded_features))
    
    # 如果特征数量超过最大限制，使用包装方法进行最终选择
    if len(combined_features) > max_features:
        # 使用包装方法在候选特征中选择
        X_subset = X[combined_features]
        selected_features = wrapper_method(X_subset, y, max_features)
    else:
        selected_features = combined_features
    
    return selected_features

def evaluate_feature_combinations(X, y, feature_list, max_features=10):
    """
    评估不同特征组合的性能
    
    参数:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        feature_list (list): 候选特征列表
        max_features (int): 最大特征数量
        
    返回:
        list: 最佳特征组合
    """
    # 限制特征数量，避免组合爆炸
    if len(feature_list) > 15:
        # 使用嵌入式方法预筛选特征
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X[feature_list], y)
        importance = np.abs(lasso.coef_)
        indices = np.argsort(importance)[::-1][:15]
        feature_list = [feature_list[i] for i in indices]
    
    # 生成所有可能的特征组合
    best_score = -np.inf
    best_combination = None
    
    # 交叉验证设置
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 评估不同数量的特征组合
    for n_features in range(3, min(max_features + 1, len(feature_list) + 1)):
        for combination in itertools.combinations(feature_list, n_features):
            # 评估当前组合
            scores = cross_val_score(model, X[list(combination)], y, cv=kf, scoring='r2')
            avg_score = np.mean(scores)
            
            # 更新最佳组合
            if avg_score > best_score:
                best_score = avg_score
                best_combination = list(combination)
    
    logger.info(f"最佳特征组合 (R² = {best_score:.4f}): {best_combination}")
    return best_combination

def plot_feature_importance(df, target_col, selected_features, output_path=None):
    """
    绘制特征重要性图
    
    参数:
        df (DataFrame): 输入数据
        target_col (str): 目标列名
        selected_features (list): 选择的特征列表
        output_path (str): 输出路径
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
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.savefig('results/feature_importance_optimized.png')
        
        plt.close()
        
        # 绘制特征相关性热图
        plt.figure(figsize=(12, 10))
        corr = df[selected_features + [target_col]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('特征相关性热图')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path.replace('.png', '_correlation.png'))
        else:
            plt.savefig('results/feature_correlation_optimized.png')
        
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制特征重要性图时出错: {str(e)}")

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_and_preprocess_data
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 选择最优特征
    target_col = 'PV_number'
    selected_features = select_optimal_features(df, target_col, max_features=10, method='combined')
    
    # 绘制特征重要性图
    plot_feature_importance(df, target_col, selected_features)
    
    print("特征选择完成")
