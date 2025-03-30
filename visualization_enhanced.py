#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统可视化模块（增强版）

本模块提供了高级可视化功能，包括：
1. 残差分析图
2. 特征重要性可视化
3. 预测结果可视化
4. 交互式可视化
5. 模型性能比较图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
from datetime import datetime
import json
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入配置
from config import PATHS

# 设置日志
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    logger.warning("无法设置中文字体，图表中的中文可能无法正确显示")

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")
    return directory

def plot_prediction_vs_actual(y_true, y_pred, title, output_path=None):
    """
    绘制预测值与实际值对比图
    
    Args:
        y_true: 实际值
        y_pred: 预测值
        title: 图表标题
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"prediction_vs_actual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
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
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"预测值与实际值对比图已保存到 {output_path}")
    return output_path

def plot_residual_analysis(y_true, y_pred, title="残差分析", output_path=None):
    """
    绘制残差分析图
    
    Args:
        y_true: 实际值
        y_pred: 预测值
        title: 图表标题
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"residual_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 10))
    
    # 残差分布图
    plt.subplot(2, 2, 1)
    plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('残差分布')
    plt.xlabel('残差值')
    plt.ylabel('频率')
    
    # 残差与预测值散点图
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.7, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差 vs 预测值')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    
    # Q-Q图
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差Q-Q图')
    
    # 残差自相关图
    plt.subplot(2, 2, 4)
    try:
        plot_acf(residuals, lags=min(20, len(residuals)//5), ax=plt.gca())
        plt.title('残差自相关图')
    except Exception as e:
        logger.warning(f"绘制残差自相关图失败: {e}")
        plt.text(0.5, 0.5, f"自相关图绘制失败: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('残差自相关图（失败）')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"残差分析图已保存到 {output_path}")
    return output_path

def plot_feature_importance(feature_names, importances, title="特征重要性", output_path=None):
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名称列表
        importances: 特征重要性列表
        title: 图表标题
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('重要性', fontsize=14)
    plt.ylabel('特征', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"特征重要性图已保存到 {output_path}")
    return output_path

def plot_correlation_matrix(df, target_col=None, output_path=None):
    """
    绘制相关性矩阵热图
    
    Args:
        df: 数据框
        target_col: 目标列名，如果提供，将按与目标列的相关性排序
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # 计算相关性矩阵
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    
    # 如果提供了目标列，按与目标列的相关性排序
    if target_col is not None and target_col in corr.columns:
        corr_with_target = corr[target_col].abs().sort_values(ascending=False)
        ordered_cols = corr_with_target.index.tolist()
        corr = corr.loc[ordered_cols, ordered_cols]
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    
    plt.title('特征相关性矩阵', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"相关性矩阵热图已保存到 {output_path}")
    return output_path

def plot_model_comparison(model_results, metrics=['r2', 'rmse', 'mae'], output_path=None):
    """
    绘制模型比较图
    
    Args:
        model_results: 字典，键为模型名称，值为包含评估指标的字典
        metrics: 要比较的指标列表
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    model_names = list(model_results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    plt.figure(figsize=(12, 4 * n_metrics))
    
    for i, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, i+1)
        
        values = [model_results[model][metric] for model in model_names]
        
        # 对于R²，值越高越好；对于误差指标，值越低越好
        if metric.lower() == 'r2' or metric.lower() == 'adj_r2':
            colors = ['green' if v >= 0.9 else 'orange' if v >= 0.7 else 'red' for v in values]
        else:
            # 对于误差指标，使用相对比例来确定颜色
            max_val = max(values)
            colors = ['green' if v <= 0.5*max_val else 'orange' if v <= 0.75*max_val else 'red' for v in values]
        
        bars = plt.bar(model_names, values, color=colors)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title(f'模型比较 - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 对于R²，设置y轴范围为0-1
        if metric.lower() == 'r2' or metric.lower() == 'adj_r2':
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"模型比较图已保存到 {output_path}")
    return output_path

def plot_learning_curve(train_sizes, train_scores, test_scores, title="学习曲线", output_path=None):
    """
    绘制学习曲线
    
    Args:
        train_sizes: 训练集大小列表
        train_scores: 训练集得分列表
        test_scores: 测试集得分列表
        title: 图表标题
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"learning_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    plt.figure(figsize=(10, 6))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="测试集得分")
    
    plt.title(title, fontsize=16)
    plt.xlabel("训练样本数", fontsize=14)
    plt.ylabel("得分", fontsize=14)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"学习曲线已保存到 {output_path}")
    return output_path

def create_interactive_visualization(df, target_col, selected_features, output_path=None):
    """
    创建交互式可视化HTML文件
    
    Args:
        df: 数据框
        target_col: 目标列名
        selected_features: 选定的特征列表
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"interactive_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    # 创建子图
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('特征分布', '特征相关性', '特征重要性', '目标变量分布'))
    
    # 特征分布
    for i, feature in enumerate(selected_features[:3]):  # 只展示前3个特征
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature),
            row=1, col=1
        )
    
    # 特征相关性
    corr = df[selected_features + [target_col]].corr()
    fig.add_trace(
        go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu'),
        row=1, col=2
    )
    
    # 特征重要性（假设已经计算好）
    # 这里使用相关性的绝对值作为简单的重要性度量
    importance = corr[target_col].abs().sort_values(ascending=False)
    importance = importance.drop(target_col)
    
    fig.add_trace(
        go.Bar(y=importance.index, x=importance.values, orientation='h'),
        row=2, col=1
    )
    
    # 目标变量分布
    fig.add_trace(
        go.Histogram(x=df[target_col], name=target_col),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(height=800, width=1000, title_text="数据可视化")
    
    # 保存为HTML文件
    fig.write_html(output_path)
    
    logger.info(f"交互式可视化已保存到 {output_path}")
    return output_path

def create_interactive_prediction_visualization(y_true, y_pred, model_name, output_path=None):
    """
    创建交互式预测结果可视化
    
    Args:
        y_true: 实际值
        y_pred: 预测值
        model_name: 模型名称
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"interactive_prediction_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    # 计算评估指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 创建数据框
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'residual': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'index': range(len(y_true))
    })
    
    # 创建子图
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('预测值 vs 实际值', '残差分布', '残差 vs 预测值', '绝对误差'))
    
    # 预测值 vs 实际值
    fig.add_trace(
        go.Scatter(x=df['actual'], y=df['predicted'], mode='markers', 
                  name='数据点', marker=dict(color='blue', opacity=0.7)),
        row=1, col=1
    )
    
    # 添加对角线
    min_val = min(df['actual'].min(), df['predicted'].min())
    max_val = max(df['actual'].max(), df['predicted'].max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='理想线', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # 残差分布
    fig.add_trace(
        go.Histogram(x=df['residual'], name='残差', marker=dict(color='green')),
        row=1, col=2
    )
    
    # 残差 vs 预测值
    fig.add_trace(
        go.Scatter(x=df['predicted'], y=df['residual'], mode='markers', 
                  name='残差', marker=dict(color='orange', opacity=0.7)),
        row=2, col=1
    )
    
    # 添加水平线
    fig.add_trace(
        go.Scatter(x=[df['predicted'].min(), df['predicted'].max()], y=[0, 0], 
                  mode='lines', name='零线', line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # 绝对误差
    fig.add_trace(
        go.Bar(x=df['index'], y=df['abs_error'], name='绝对误差', marker=dict(color='purple')),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(
        height=800, 
        width=1000, 
        title_text=f"{model_name} 模型预测结果 (R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f})"
    )
    
    # 更新坐标轴标签
    fig.update_xaxes(title_text="实际值", row=1, col=1)
    fig.update_yaxes(title_text="预测值", row=1, col=1)
    
    fig.update_xaxes(title_text="残差", row=1, col=2)
    fig.update_yaxes(title_text="频率", row=1, col=2)
    
    fig.update_xaxes(title_text="预测值", row=2, col=1)
    fig.update_yaxes(title_text="残差", row=2, col=1)
    
    fig.update_xaxes(title_text="样本索引", row=2, col=2)
    fig.update_yaxes(title_text="绝对误差", row=2, col=2)
    
    # 保存为HTML文件
    fig.write_html(output_path)
    
    logger.info(f"交互式预测结果可视化已保存到 {output_path}")
    return output_path

def plot_feature_pair_relationships(df, target_col, top_n_features=4, output_path=None):
    """
    绘制特征对关系图
    
    Args:
        df: 数据框
        target_col: 目标列名
        top_n_features: 选择的顶部特征数量
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"feature_pair_relationships_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # 计算与目标变量的相关性
    corr = df.corr()[target_col].abs().sort_values(ascending=False)
    top_features = corr.index[:top_n_features+1].tolist()  # 包括目标变量
    
    if target_col in top_features:
        top_features.remove(target_col)
        top_features = [target_col] + top_features[:top_n_features]
    
    # 创建特征对图
    sns.set(style="ticks")
    sns.pairplot(df[top_features], diag_kind="kde", markers="o", 
                plot_kws=dict(s=50, edgecolor="white", linewidth=1),
                diag_kws=dict(shade=True))
    
    plt.suptitle(f"顶部{top_n_features}个特征的关系图", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"特征对关系图已保存到 {output_path}")
    return output_path

def plot_3d_feature_importance(df, target_col, top_n_features=3, output_path=None):
    """
    绘制3D特征重要性图
    
    Args:
        df: 数据框
        target_col: 目标列名
        top_n_features: 选择的顶部特征数量
        output_path: 输出路径
    """
    if output_path is None:
        results_dir = ensure_dir(PATHS['results_dir'])
        output_path = os.path.join(results_dir, f"3d_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    # 计算与目标变量的相关性
    corr = df.corr()[target_col].abs().sort_values(ascending=False)
    top_features = corr.index[1:top_n_features+1].tolist()  # 排除目标变量自身
    
    if len(top_features) < 2:
        logger.warning(f"特征数量不足，无法创建3D图")
        return None
    
    # 创建3D散点图
    if len(top_features) >= 3:
        # 使用前三个特征
        fig = px.scatter_3d(df, x=top_features[0], y=top_features[1], z=top_features[2],
                          color=target_col, opacity=0.7,
                          title=f"前三个重要特征与{target_col}的关系")
    else:
        # 只有两个特征时，使用目标变量作为z轴
        fig = px.scatter_3d(df, x=top_features[0], y=top_features[1], z=target_col,
                          color=target_col, opacity=0.7,
                          title=f"前两个重要特征与{target_col}的关系")
    
    # 更新布局
    fig.update_layout(
        scene = dict(
            xaxis_title=top_features[0],
            yaxis_title=top_features[1],
            zaxis_title=top_features[2] if len(top_features) >= 3 else target_col
        )
    )
    
    # 保存为HTML文件
    fig.write_html(output_path)
    
    logger.info(f"3D特征重要性图已保存到 {output_path}")
    return output_path

def generate_comprehensive_report(model_results, df, target_col, selected_features, y_true, y_pred, output_dir=None):
    """
    生成综合报告，包含多种可视化
    
    Args:
        model_results: 字典，键为模型名称，值为包含评估指标的字典
        df: 数据框
        target_col: 目标列名
        selected_features: 选定的特征列表
        y_true: 实际值
        y_pred: 预测值
        output_dir: 输出目录
    
    Returns:
        dict: 包含所有生成的图表路径
    """
    if output_dir is None:
        output_dir = ensure_dir(os.path.join(PATHS['results_dir'], f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    else:
        output_dir = ensure_dir(output_dir)
    
    logger.info(f"开始生成综合报告，输出目录: {output_dir}")
    
    # 存储所有生成的图表路径
    report_files = {}
    
    # 1. 预测值与实际值对比图
    report_files['prediction_vs_actual'] = plot_prediction_vs_actual(
        y_true, y_pred, "预测值与实际值对比",
        os.path.join(output_dir, "prediction_vs_actual.png")
    )
    
    # 2. 残差分析图
    report_files['residual_analysis'] = plot_residual_analysis(
        y_true, y_pred, "残差分析",
        os.path.join(output_dir, "residual_analysis.png")
    )
    
    # 3. 特征重要性图（假设已经计算好）
    # 这里使用相关性的绝对值作为简单的重要性度量
    importance = df[selected_features].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    report_files['feature_importance'] = plot_feature_importance(
        importance.index.tolist(), importance.values,
        "特征重要性（基于相关性）",
        os.path.join(output_dir, "feature_importance.png")
    )
    
    # 4. 相关性矩阵热图
    report_files['correlation_matrix'] = plot_correlation_matrix(
        df[selected_features + [target_col]], target_col,
        os.path.join(output_dir, "correlation_matrix.png")
    )
    
    # 5. 模型比较图
    if model_results:
        report_files['model_comparison'] = plot_model_comparison(
            model_results, metrics=['r2', 'rmse', 'mae'],
            os.path.join(output_dir, "model_comparison.png")
        )
    
    # 6. 特征对关系图
    report_files['feature_pair_relationships'] = plot_feature_pair_relationships(
        df[selected_features + [target_col]], target_col, top_n_features=min(4, len(selected_features)),
        os.path.join(output_dir, "feature_pair_relationships.png")
    )
    
    # 7. 交互式可视化
    report_files['interactive_visualization'] = create_interactive_visualization(
        df, target_col, selected_features,
        os.path.join(output_dir, "interactive_visualization.html")
    )
    
    # 8. 交互式预测结果可视化
    report_files['interactive_prediction'] = create_interactive_prediction_visualization(
        y_true, y_pred, "最佳模型",
        os.path.join(output_dir, "interactive_prediction.html")
    )
    
    # 9. 3D特征重要性图
    if len(selected_features) >= 2:
        report_files['3d_feature_importance'] = plot_3d_feature_importance(
            df[selected_features + [target_col]], target_col, top_n_features=min(3, len(selected_features)),
            os.path.join(output_dir, "3d_feature_importance.html")
        )
    
    # 10. 生成报告索引HTML文件
    index_html = create_report_index(report_files, output_dir)
    report_files['index'] = index_html
    
    logger.info(f"综合报告生成完成，索引文件: {index_html}")
    return report_files

def create_report_index(report_files, output_dir):
    """
    创建报告索引HTML文件
    
    Args:
        report_files: 字典，键为报告类型，值为文件路径
        output_dir: 输出目录
    
    Returns:
        str: 索引文件路径
    """
    index_path = os.path.join(output_dir, "index.html")
    
    # 创建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CCUS CO2气窜预测系统分析报告</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
            }}
            .report-section {{
                margin-bottom: 40px;
            }}
            .report-item {{
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            .report-item h3 {{
                margin-top: 0;
                color: #16a085;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin: 10px 0;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 50px;
            }}
        </style>
    </head>
    <body>
        <h1>CCUS CO2气窜预测系统分析报告</h1>
        
        <div class="report-section">
            <h2>1. 模型性能分析</h2>
    """
    
    # 添加预测值与实际值对比图
    if 'prediction_vs_actual' in report_files:
        file_path = os.path.basename(report_files['prediction_vs_actual'])
        html_content += f"""
            <div class="report-item">
                <h3>预测值与实际值对比</h3>
                <p>该图展示了模型预测值与实际值的对比，理想情况下所有点应该落在对角线上。</p>
                <img src="{file_path}" alt="预测值与实际值对比图">
            </div>
        """
    
    # 添加残差分析图
    if 'residual_analysis' in report_files:
        file_path = os.path.basename(report_files['residual_analysis'])
        html_content += f"""
            <div class="report-item">
                <h3>残差分析</h3>
                <p>残差分析帮助我们理解模型的预测误差分布情况，理想情况下残差应该呈正态分布且均值为0。</p>
                <img src="{file_path}" alt="残差分析图">
            </div>
        """
    
    # 添加交互式预测结果可视化
    if 'interactive_prediction' in report_files:
        file_path = os.path.basename(report_files['interactive_prediction'])
        html_content += f"""
            <div class="report-item">
                <h3>交互式预测结果可视化</h3>
                <p>这是一个交互式的预测结果可视化，您可以通过鼠标悬停查看详细信息。</p>
                <p><a href="{file_path}" target="_blank">打开交互式预测结果可视化</a></p>
            </div>
        """
    
    # 添加模型比较图
    if 'model_comparison' in report_files:
        file_path = os.path.basename(report_files['model_comparison'])
        html_content += f"""
            <div class="report-item">
                <h3>模型比较</h3>
                <p>该图比较了不同模型在各项评估指标上的表现。</p>
                <img src="{file_path}" alt="模型比较图">
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="report-section">
            <h2>2. 特征分析</h2>
    """
    
    # 添加特征重要性图
    if 'feature_importance' in report_files:
        file_path = os.path.basename(report_files['feature_importance'])
        html_content += f"""
            <div class="report-item">
                <h3>特征重要性</h3>
                <p>该图展示了各个特征对预测结果的重要性，帮助我们理解哪些因素对气窜预测影响最大。</p>
                <img src="{file_path}" alt="特征重要性图">
            </div>
        """
    
    # 添加相关性矩阵热图
    if 'correlation_matrix' in report_files:
        file_path = os.path.basename(report_files['correlation_matrix'])
        html_content += f"""
            <div class="report-item">
                <h3>特征相关性矩阵</h3>
                <p>该热图展示了各个特征之间的相关性，帮助我们理解特征间的关系。</p>
                <img src="{file_path}" alt="相关性矩阵热图">
            </div>
        """
    
    # 添加特征对关系图
    if 'feature_pair_relationships' in report_files:
        file_path = os.path.basename(report_files['feature_pair_relationships'])
        html_content += f"""
            <div class="report-item">
                <h3>特征对关系图</h3>
                <p>该图展示了重要特征之间的关系，以及它们与目标变量的分布情况。</p>
                <img src="{file_path}" alt="特征对关系图">
            </div>
        """
    
    # 添加3D特征重要性图
    if '3d_feature_importance' in report_files:
        file_path = os.path.basename(report_files['3d_feature_importance'])
        html_content += f"""
            <div class="report-item">
                <h3>3D特征重要性图</h3>
                <p>这是一个交互式的3D图，展示了最重要的几个特征与目标变量的关系。</p>
                <p><a href="{file_path}" target="_blank">打开3D特征重要性图</a></p>
            </div>
        """
    
    # 添加交互式可视化
    if 'interactive_visualization' in report_files:
        file_path = os.path.basename(report_files['interactive_visualization'])
        html_content += f"""
            <div class="report-item">
                <h3>交互式数据可视化</h3>
                <p>这是一个交互式的数据可视化，包含了特征分布、相关性和重要性等信息。</p>
                <p><a href="{file_path}" target="_blank">打开交互式数据可视化</a></p>
            </div>
        """
    
    # 添加时间戳和结束标签
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content += f"""
        </div>
        
        <p class="timestamp">报告生成时间: {timestamp}</p>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"报告索引HTML文件已保存到 {index_path}")
    return index_path

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试代码
    logger.info("测试可视化模块")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    X = np.random.rand(n_samples, 5)
    y_true = 2 * X[:, 0] + 1 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, n_samples)
    y_pred = y_true + np.random.normal(0, 0.2, n_samples)
    
    # 创建数据框
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y_true
    
    # 测试各种可视化函数
    plot_prediction_vs_actual(y_true, y_pred, "测试模型")
    plot_residual_analysis(y_true, y_pred)
    plot_feature_importance(df.columns[:-1], np.array([2, 1, 1.5, 0.5, 0.1]))
    plot_correlation_matrix(df, 'target')
    
    # 测试模型比较
    model_results = {
        'Model A': {'r2': 0.95, 'rmse': 0.12, 'mae': 0.09},
        'Model B': {'r2': 0.92, 'rmse': 0.15, 'mae': 0.11},
        'Model C': {'r2': 0.88, 'rmse': 0.18, 'mae': 0.14}
    }
    plot_model_comparison(model_results)
    
    # 测试交互式可视化
    create_interactive_visualization(df, 'target', df.columns[:-1].tolist())
    create_interactive_prediction_visualization(y_true, y_pred, "测试模型")
    
    logger.info("可视化测试完成")
