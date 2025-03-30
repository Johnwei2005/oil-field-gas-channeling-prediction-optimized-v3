#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统全流程启动脚本

本脚本整合了整个项目的完整流程，包括：
1. 数据处理
2. 特征工程
3. 模型训练
4. 模型评估
5. 结果可视化

使用方法：
    python run_all.py [--model-type MODEL_TYPE] [--skip-steps STEPS]

参数：
    --model-type: 模型类型，可选 'random_forest', 'gradient_boosting', 'gaussian_process'（默认）
    --skip-steps: 跳过的步骤，用逗号分隔，可选 'data', 'train', 'evaluate'
"""

import os
import sys
import argparse
import logging
import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置日志
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"run_all_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_section_header(title):
    """打印带有分隔线的节标题"""
    line = "=" * 80
    logger.info(f"\n{line}\n{title.center(80)}\n{line}")

def run_data_processing():
    """运行数据处理步骤"""
    print_section_header("数据处理")
    
    try:
        from data_processor import load_and_preprocess_data
        
        logger.info("开始数据处理...")
        start_time = time.time()
        
        # 加载并预处理数据
        df_processed = load_and_preprocess_data(save_processed=True)
        
        end_time = time.time()
        logger.info(f"数据处理完成，耗时 {end_time - start_time:.2f} 秒")
        logger.info(f"处理后的数据形状: {df_processed.shape}")
        
        return True
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        return False

def run_model_training(model_type='gaussian_process'):
    """运行模型训练步骤"""
    print_section_header(f"模型训练 ({model_type})")
    
    try:
        from main import train_model
        
        logger.info(f"开始训练 {model_type} 模型...")
        start_time = time.time()
        
        # 训练模型
        model, selected_features = train_model(model_type=model_type, save_model=True)
        
        end_time = time.time()
        logger.info(f"模型训练完成，耗时 {end_time - start_time:.2f} 秒")
        
        return model, selected_features
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        return None, None

def run_model_evaluation(model, selected_features, model_type='gaussian_process'):
    """运行模型评估步骤"""
    print_section_header(f"模型评估 ({model_type})")
    
    try:
        from data_processor import load_and_preprocess_data
        from main import predict
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        logger.info("开始模型评估...")
        start_time = time.time()
        
        # 加载数据
        from config import DATA_CONFIG
        df = load_and_preprocess_data(save_processed=False)
        
        # 创建物理约束特征
        from enhanced_features_optimized import create_physics_informed_features
        df_physics = create_physics_informed_features(df)
        
        # 准备数据
        target_col = DATA_CONFIG['target_column']
        X = df_physics[selected_features]
        y = df_physics[target_col]
        
        # 预测
        y_pred = predict(model, X)
        
        # 评估
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
        plt.title(f'{model_type} 模型 (R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f})')
        plt.grid(True)
        
        # 保存结果目录
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plot_path = os.path.join(results_dir, f"{model_type}_predictions.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"预测图已保存到 {plot_path}")
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'actual': y,
            'predicted': y_pred,
            'error': y - y_pred
        })
        
        results_csv_path = os.path.join(results_dir, f"{model_type}_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"预测结果已保存到 {results_csv_path}")
        
        end_time = time.time()
        logger.info(f"模型评估完成，耗时 {end_time - start_time:.2f} 秒")
        
        return True
    except Exception as e:
        logger.error(f"模型评估失败: {e}")
        return False

def run_model_fine_tuning(model_type='gaussian_process'):
    """运行模型微调步骤"""
    print_section_header(f"模型微调 ({model_type})")
    
    try:
        # 导入模型微调模块
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model_fine_tune import fine_tune_model
        
        logger.info(f"开始微调 {model_type} 模型...")
        start_time = time.time()
        
        # 微调模型
        fine_tuned_model = fine_tune_model(model_type)
        
        end_time = time.time()
        logger.info(f"模型微调完成，耗时 {end_time - start_time:.2f} 秒")
        
        return True
    except Exception as e:
        logger.error(f"模型微调失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CCUS CO2气窜预测系统全流程启动脚本')
    parser.add_argument('--model-type', type=str, default='gaussian_process', 
                        choices=['random_forest', 'gradient_boosting', 'gaussian_process'],
                        help='模型类型')
    parser.add_argument('--skip-steps', type=str, default='',
                        help='跳过的步骤，用逗号分隔，可选 data,train,evaluate,fine_tune')
    
    args = parser.parse_args()
    
    # 解析跳过的步骤
    skip_steps = [step.strip() for step in args.skip_steps.split(',') if step.strip()]
    
    print_section_header("CCUS CO2气窜预测系统全流程启动")
    logger.info(f"选择的模型类型: {args.model_type}")
    logger.info(f"跳过的步骤: {skip_steps if skip_steps else '无'}")
    
    # 创建必要的目录
    from config import PATHS
    for path_name, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"确保目录存在: {path}")
    
    # 1. 数据处理
    if 'data' not in skip_steps:
        data_success = run_data_processing()
        if not data_success:
            logger.error("数据处理失败，终止流程")
            return
    else:
        logger.info("跳过数据处理步骤")
    
    # 2. 模型训练
    if 'train' not in skip_steps:
        model, selected_features = run_model_training(args.model_type)
        if model is None:
            logger.error("模型训练失败，终止流程")
            return
    else:
        logger.info("跳过模型训练步骤")
        # 如果跳过训练但需要评估，则加载已有模型
        if 'evaluate' not in skip_steps:
            from main import load_trained_model
            from enhanced_features_optimized import select_optimal_features_limited
            from data_processor import load_and_preprocess_data
            from enhanced_features_optimized import create_physics_informed_features
            from config import DATA_CONFIG
            
            model = load_trained_model(args.model_type)
            
            # 获取特征列表
            df = load_and_preprocess_data(save_processed=False)
            df_physics = create_physics_informed_features(df)
            target_col = DATA_CONFIG['target_column']
            selected_features = select_optimal_features_limited(df_physics, target_col, max_features=10)
    
    # 3. 模型评估
    if 'evaluate' not in skip_steps and 'train' not in skip_steps:
        eval_success = run_model_evaluation(model, selected_features, args.model_type)
        if not eval_success:
            logger.warning("模型评估失败，继续流程")
    elif 'evaluate' not in skip_steps:
        logger.info("由于跳过了模型训练，无法进行模型评估")
    else:
        logger.info("跳过模型评估步骤")
    
    # 4. 模型微调
    if 'fine_tune' not in skip_steps:
        fine_tune_success = run_model_fine_tuning(args.model_type)
        if not fine_tune_success:
            logger.warning("模型微调失败，继续流程")
    else:
        logger.info("跳过模型微调步骤")
    
    print_section_header("全流程完成")
    logger.info(f"日志已保存到: {log_filename}")
    logger.info("可以通过以下命令查看结果:")
    logger.info("1. 查看训练好的模型: ls -la models/")
    logger.info("2. 查看评估结果: ls -la results/")
    logger.info("3. 查看处理后的数据: ls -la data/processed/")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
