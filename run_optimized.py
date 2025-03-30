#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全流程自动化脚本

该脚本整合了整个建模流程，包括：
1. 数据处理
2. 特征选择
3. 模型训练
4. 参数优化
5. 性能验证
"""

import os
import logging
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 导入自定义模块
from data_processor import load_and_preprocess_data
from optimized_feature_selection import select_optimal_features, plot_feature_importance
from simplified_model import SimplifiedModel
from parameter_optimization import optimize_model_parameters, find_optimal_model
from model_validation import validate_model_performance
from config import PATHS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PATHS['logs'], f'run_optimized_{time.strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CCUS CO2气窜预测系统全流程自动化脚本')
    
    parser.add_argument('--skip-steps', type=str, default='',
                        help='要跳过的步骤，用逗号分隔，可选值: data, feature, model, optimize, validate')
    
    parser.add_argument('--model-type', type=str, default='ensemble',
                        help='模型类型，可选值: rf, gbm, svr, elastic_net, ridge, ensemble')
    
    parser.add_argument('--feature-method', type=str, default='combined',
                        help='特征选择方法，可选值: filter, wrapper, embedded, combined')
    
    parser.add_argument('--max-features', type=int, default=10,
                        help='最大特征数量，默认为10')
    
    parser.add_argument('--target-r2-min', type=float, default=0.9,
                        help='目标R²值下限，默认为0.9')
    
    parser.add_argument('--target-r2-max', type=float, default=0.93,
                        help='目标R²值上限，默认为0.93')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 要跳过的步骤
    skip_steps = args.skip_steps.split(',') if args.skip_steps else []
    
    # 目标R²值范围
    target_r2_range = (args.target_r2_min, args.target_r2_max)
    
    logger.info(f"开始全流程自动化，模型类型: {args.model_type}, 特征选择方法: {args.feature_method}, 最大特征数量: {args.max_features}")
    logger.info(f"目标R²值范围: {target_r2_range}")
    logger.info(f"跳过步骤: {skip_steps}")
    
    # 步骤1: 数据处理
    if 'data' not in skip_steps:
        logger.info("步骤1: 数据处理")
        df = load_and_preprocess_data()
    else:
        logger.info("跳过步骤1: 数据处理")
        # 尝试加载已处理的数据
        processed_data_path = os.path.join(PATHS['data_dir'], 'processed', 'processed_data.csv')
        if os.path.exists(processed_data_path):
            df = pd.read_csv(processed_data_path)
            logger.info(f"已加载处理过的数据: {processed_data_path}")
        else:
            logger.warning(f"未找到处理过的数据: {processed_data_path}，将进行数据处理")
            df = load_and_preprocess_data()
    
    # 目标变量
    target_col = 'PV_number'
    
    # 步骤2: 特征选择
    if 'feature' not in skip_steps:
        logger.info("步骤2: 特征选择")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        selected_features = select_optimal_features(
            df, target_col, max_features=args.max_features, method=args.feature_method
        )
        
        # 绘制特征重要性图
        plot_feature_importance(df, target_col, selected_features)
        
        logger.info(f"选择的特征: {selected_features}")
    else:
        logger.info("跳过步骤2: 特征选择")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 尝试加载已保存的模型以获取特征
        model_path = os.path.join(PATHS['models'], f'{args.model_type}_model_optimized.pkl')
        if os.path.exists(model_path):
            model = SimplifiedModel.load(model_path)
            selected_features = model.selected_features
            logger.info(f"从模型加载特征: {selected_features}")
        else:
            logger.warning(f"未找到模型: {model_path}，将进行特征选择")
            selected_features = select_optimal_features(
                df, target_col, max_features=args.max_features, method=args.feature_method
            )
    
    # 步骤3: 模型训练
    if 'model' not in skip_steps:
        logger.info("步骤3: 模型训练")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建并训练模型
        model = SimplifiedModel(model_type=args.model_type, target_r2_range=target_r2_range)
        model.fit(X_train, y_train, feature_selection_method=args.feature_method, max_features=args.max_features)
        
        # 评估模型
        metrics = model.evaluate(X_test, y_test)
        
        logger.info(f"模型训练完成，测试集性能: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
        
        # 绘制预测图
        model.plot_predictions(X_test, y_test)
        
        # 保存模型
        model.save(os.path.join(PATHS['models'], f'{args.model_type}_model_optimized.pkl'))
    else:
        logger.info("跳过步骤3: 模型训练")
    
    # 步骤4: 参数优化
    if 'optimize' not in skip_steps:
        logger.info("步骤4: 参数优化")
        
        # 寻找最优模型
        model_type, params = find_optimal_model(X, y, target_r2_range=target_r2_range)
        
        logger.info(f"参数优化完成，最佳模型类型: {model_type}, 最佳参数: {params}")
    else:
        logger.info("跳过步骤4: 参数优化")
    
    # 步骤5: 性能验证
    if 'validate' not in skip_steps:
        logger.info("步骤5: 性能验证")
        
        # 验证模型性能
        metrics = validate_model_performance()
        
        logger.info(f"性能验证完成，测试集性能: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
    else:
        logger.info("跳过步骤5: 性能验证")
    
    logger.info("全流程自动化完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}", exc_info=True)
