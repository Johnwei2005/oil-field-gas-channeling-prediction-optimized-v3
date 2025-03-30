#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统主程序

本程序整合了所有优化后的模块，
提供完整的模型训练、评估和预测功能。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
import joblib
import argparse

# 导入自定义模块
from data_processor import load_data, preprocess_data
from enhanced_features_optimized import create_physics_informed_features, select_optimal_features_limited
from residual_model_optimized import ResidualModel

# 导入配置
from config import DATA_CONFIG, PATHS

# 创建必要的目录
os.makedirs(PATHS['model_dir'], exist_ok=True)
os.makedirs(PATHS['results_dir'], exist_ok=True)
os.makedirs(PATHS['log_dir'], exist_ok=True)

# 设置日志
log_filename = os.path.join(PATHS['log_dir'], f"main_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_model(model_type='gaussian_process', save_model=True):
    """
    训练模型
    
    Args:
        model_type: 模型类型，可选'random_forest', 'gradient_boosting', 'gaussian_process'
        save_model: 是否保存模型
        
    Returns:
        tuple: (模型, 特征列表)
    """
    logger.info(f"开始训练 {model_type} 模型")
    
    # 1. 加载和预处理数据
    df = load_data()
    df = preprocess_data(df)
    
    # 2. 创建物理约束特征
    df_physics = create_physics_informed_features(df)
    
    # 3. 特征选择
    target_col = DATA_CONFIG['target_column']
    selected_features = select_optimal_features_limited(df_physics, target_col, max_features=10)
    
    # 4. 准备数据
    X = df_physics[selected_features]
    y = df_physics[target_col]
    
    # 5. 创建并训练模型
    model = ResidualModel(model_type=model_type)
    model.fit(X, y)
    
    # 6. 评估模型
    metrics = model.evaluate(X, y)
    
    logger.info(f"模型训练完成，性能指标:")
    logger.info(f"R²: {metrics['final_r2']:.4f}")
    logger.info(f"RMSE: {metrics['final_rmse']:.4f}")
    logger.info(f"MAE: {metrics['final_mae']:.4f}")
    
    # 7. 保存模型
    if save_model:
        model_path = os.path.join(PATHS['model_dir'], f"{model_type}_model.pkl")
        model.save(model_path)
        logger.info(f"模型已保存到 {model_path}")
    
    return model, selected_features

def load_trained_model(model_type='gaussian_process'):
    """
    加载训练好的模型
    
    Args:
        model_type: 模型类型
        
    Returns:
        ResidualModel: 加载的模型
    """
    model_path = os.path.join(PATHS['model_dir'], f"fine_tuned_{model_type}_model.pkl")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(PATHS['model_dir'], f"{model_type}_model.pkl")
    
    if not os.path.exists(model_path):
        logger.warning(f"未找到模型文件 {model_path}，将训练新模型")
        model, _ = train_model(model_type)
        return model
    
    logger.info(f"从 {model_path} 加载模型")
    model = ResidualModel.load(model_path)
    
    return model

def predict(model, X):
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        X: 特征数据
        
    Returns:
        numpy.ndarray: 预测值
    """
    return model.predict(X)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CCUS CO2气窜预测系统')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', action='store_true', help='使用模型进行预测')
    parser.add_argument('--model-type', type=str, default='gaussian_process', 
                        choices=['random_forest', 'gradient_boosting', 'gaussian_process'],
                        help='模型类型')
    parser.add_argument('--input-file', type=str, help='输入数据文件路径')
    parser.add_argument('--output-file', type=str, help='输出预测结果文件路径')
    
    args = parser.parse_args()
    
    if args.train:
        # 训练模型
        train_model(args.model_type)
    
    elif args.predict:
        if not args.input_file:
            logger.error("预测模式需要提供输入数据文件路径")
            return
        
        # 加载模型
        model = load_trained_model(args.model_type)
        
        # 加载数据
        try:
            df = pd.read_csv(args.input_file)
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            return
        
        # 预处理数据
        df = preprocess_data(df)
        
        # 创建物理约束特征
        df_physics = create_physics_informed_features(df)
        
        # 特征选择
        target_col = DATA_CONFIG['target_column']
        if target_col in df_physics.columns:
            selected_features = select_optimal_features_limited(df_physics, target_col, max_features=10)
            X = df_physics[selected_features]
        else:
            logger.warning(f"输入数据中未找到目标列 {target_col}，将使用所有特征")
            X = df_physics
        
        # 预测
        y_pred = predict(model, X)
        
        # 保存预测结果
        if args.output_file:
            df_result = df.copy()
            df_result['predicted_PV_number'] = y_pred
            df_result.to_csv(args.output_file, index=False)
            logger.info(f"预测结果已保存到 {args.output_file}")
        else:
            logger.info("预测结果:")
            for i, pred in enumerate(y_pred):
                logger.info(f"样本 {i+1}: {pred:.4f}")
    
    else:
        # 默认行为：训练并评估模型
        logger.info("未指定操作，将训练并评估模型")
        
        # 训练模型
        model, selected_features = train_model(args.model_type)
        
        # 加载数据
        df = load_data()
        df = preprocess_data(df)
        df_physics = create_physics_informed_features(df)
        
        # 准备数据
        target_col = DATA_CONFIG['target_column']
        X = df_physics[selected_features]
        y = df_physics[target_col]
        
        # 预测
        y_pred = predict(model, X)
        
        # 评估
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"模型评估结果:")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        
        # 绘制预测图
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{args.model_type} 模型 (R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f})')
        plt.grid(True)
        
        plot_path = os.path.join(PATHS['results_dir'], f"{args.model_type}_predictions.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"预测图已保存到 {plot_path}")

if __name__ == "__main__":
    main()
