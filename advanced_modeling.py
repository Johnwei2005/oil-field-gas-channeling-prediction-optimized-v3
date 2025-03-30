#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CCUS CO2气窜预测系统高级建模模块

本模块实现了多种高级建模技术，包括：
1. 迁移学习框架
2. 集成学习方法
3. 不确定性量化
4. 自动超参数优化
5. 物理信息神经网络
6. 域分解方法
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import optuna
from optuna.integration import TFKerasPruningCallback
from scipy.stats import norm
import warnings

# 导入配置
from config import DATA_CONFIG, PATHS

# 设置日志
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class PhysicsInformedResidualModel(BaseEstimator, RegressorMixin):
    """
    物理信息残差模型
    
    结合物理模型和机器学习模型，使用机器学习模型预测物理模型的残差
    """
    
    def __init__(self, ml_model_type='random_forest', physics_model=None, ml_model_params=None):
        """
        初始化
        
        Args:
            ml_model_type: 机器学习模型类型，可选'random_forest', 'gradient_boosting', 'gaussian_process', 'neural_network'
            physics_model: 物理模型函数，接受X作为输入，返回物理预测值
            ml_model_params: 机器学习模型参数字典
        """
        self.ml_model_type = ml_model_type
        self.physics_model = physics_model
        self.ml_model_params = ml_model_params or {}
        self.ml_model = None
        self.scaler = StandardScaler()
        
        # 初始化机器学习模型
        self._init_ml_model()
    
    def _init_ml_model(self):
        """初始化机器学习模型"""
        if self.ml_model_type == 'random_forest':
            self.ml_model = RandomForestRegressor(
                n_estimators=self.ml_model_params.get('n_estimators', 100),
                max_depth=self.ml_model_params.get('max_depth', None),
                min_samples_split=self.ml_model_params.get('min_samples_split', 2),
                min_samples_leaf=self.ml_model_params.get('min_samples_leaf', 1),
                random_state=RANDOM_SEED
            )
        elif self.ml_model_type == 'gradient_boosting':
            self.ml_model = GradientBoostingRegressor(
                n_estimators=self.ml_model_params.get('n_estimators', 100),
                learning_rate=self.ml_model_params.get('learning_rate', 0.1),
                max_depth=self.ml_model_params.get('max_depth', 3),
                random_state=RANDOM_SEED
            )
        elif self.ml_model_type == 'gaussian_process':
            kernel = self.ml_model_params.get('kernel', 
                     ConstantKernel() * RBF() + ConstantKernel() * Matern(nu=1.5))
            self.ml_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.ml_model_params.get('alpha', 1e-10),
                normalize_y=self.ml_model_params.get('normalize_y', True),
                random_state=RANDOM_SEED
            )
        elif self.ml_model_type == 'neural_network':
            self.ml_model = MLPRegressor(
                hidden_layer_sizes=self.ml_model_params.get('hidden_layer_sizes', (100, 50)),
                activation=self.ml_model_params.get('activation', 'relu'),
                solver=self.ml_model_params.get('solver', 'adam'),
                alpha=self.ml_model_params.get('alpha', 0.0001),
                learning_rate=self.ml_model_params.get('learning_rate', 'adaptive'),
                max_iter=self.ml_model_params.get('max_iter', 1000),
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"不支持的机器学习模型类型: {self.ml_model_type}")
    
    def _default_physics_model(self, X):
        """
        默认物理模型，基于简化的物理原理
        
        Args:
            X: 特征矩阵，DataFrame或numpy数组
        
        Returns:
            numpy.ndarray: 物理模型预测值
        """
        # 将X转换为DataFrame以便于访问列
        if not isinstance(X, pd.DataFrame):
            # 假设X的列顺序与训练时相同
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        # 检查必要的列是否存在
        required_columns = ['permeability', 'oil_viscosity', 'well_spacing', 'effective_thickness', 'formation_pressure']
        missing_columns = [col for col in required_columns if col not in X.columns]
        
        if missing_columns:
            logger.warning(f"物理模型缺少必要的列: {missing_columns}，将返回零预测")
            return np.zeros(X.shape[0])
        
        # 简化的物理模型：基于达西定律和流体力学原理
        # PV数 ≈ k * (Δp * L) / (μ * h)
        # 其中：k为渗透率，Δp为压力差，L为井距，μ为油相粘度，h为有效厚度
        
        # 计算物理预测值
        physics_pred = (
            0.01 * X['permeability'] / X['oil_viscosity'] * 
            (X['well_spacing'] / X['effective_thickness']) * 
            (X['formation_pressure'] / 20)
        )
        
        return physics_pred.values
    
    def fit(self, X, y):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            self: 训练后的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 使用物理模型进行预测
        if self.physics_model is None:
            logger.info("使用默认物理模型")
            self.physics_model = self._default_physics_model
        
        physics_pred = self.physics_model(X)
        
        # 计算残差
        residuals = y - physics_pred
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练机器学习模型预测残差
        self.ml_model.fit(X_scaled, residuals)
        
        # 计算训练指标
        ml_pred = self.ml_model.predict(X_scaled)
        final_pred = physics_pred + ml_pred
        
        r2 = r2_score(y, final_pred)
        rmse = np.sqrt(mean_squared_error(y, final_pred))
        mae = mean_absolute_error(y, final_pred)
        
        logger.info(f"训练完成，训练集指标 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵
        
        Returns:
            numpy.ndarray: 预测值
        """
        # 使用物理模型进行预测
        physics_pred = self.physics_model(X)
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 使用机器学习模型预测残差
        ml_pred = self.ml_model.predict(X_scaled)
        
        # 最终预测 = 物理预测 + 残差预测
        final_pred = physics_pred + ml_pred
        
        return final_pred
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            dict: 评估指标
        """
        # 使用物理模型进行预测
        physics_pred = self.physics_model(X)
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 使用机器学习模型预测残差
        ml_pred = self.ml_model.predict(X_scaled)
        
        # 最终预测 = 物理预测 + 残差预测
        final_pred = physics_pred + ml_pred
        
        # 计算指标
        physics_r2 = r2_score(y, physics_pred)
        physics_rmse = np.sqrt(mean_squared_error(y, physics_pred))
        physics_mae = mean_absolute_error(y, physics_pred)
        
        final_r2 = r2_score(y, final_pred)
        final_rmse = np.sqrt(mean_squared_error(y, final_pred))
        final_mae = mean_absolute_error(y, final_pred)
        
        improvement_r2 = final_r2 - physics_r2
        improvement_rmse = physics_rmse - final_rmse
        improvement_mae = physics_mae - final_mae
        
        metrics = {
            'physics_r2': physics_r2,
            'physics_rmse': physics_rmse,
            'physics_mae': physics_mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'improvement_r2': improvement_r2,
            'improvement_rmse': improvement_rmse,
            'improvement_mae': improvement_mae
        }
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        joblib.dump(self, filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            PhysicsInformedResidualModel: 加载的模型
        """
        model = joblib.load(filepath)
        logger.info(f"从 {filepath} 加载模型")
        return model

class TransferLearningModel(BaseEstimator, RegressorMixin):
    """
    迁移学习模型
    
    使用预训练模型作为基础，针对目标数据集进行微调
    """
    
    def __init__(self, base_model=None, transfer_method='fine_tune', freeze_layers=True, learning_rate=0.001):
        """
        初始化
        
        Args:
            base_model: 预训练的基础模型
            transfer_method: 迁移学习方法，可选'fine_tune'或'feature_transfer'
            freeze_layers: 是否冻结基础模型的层
            learning_rate: 学习率
        """
        self.base_model = base_model
        self.transfer_method = transfer_method
        self.freeze_layers = freeze_layers
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted_ = False
    
    def _build_model(self, input_shape):
        """
        构建模型
        
        Args:
            input_shape: 输入形状
        """
        if self.base_model is None:
            logger.warning("未提供基础模型，将创建新模型")
            # 创建新模型
            self.model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
        else:
            if self.transfer_method == 'fine_tune':
                # 微调方法：使用预训练模型的权重，但允许更新
                self.model = clone_keras_model(self.base_model)
                
                # 冻结部分层
                if self.freeze_layers:
                    for layer in self.model.layers[:-2]:  # 冻结除最后两层外的所有层
                        layer.trainable = False
            
            elif self.transfer_method == 'feature_transfer':
                # 特征迁移方法：使用预训练模型作为特征提取器
                # 获取预训练模型的倒数第二层输出
                feature_extractor = Model(
                    inputs=self.base_model.input,
                    outputs=self.base_model.layers[-2].output
                )
                
                # 冻结特征提取器
                feature_extractor.trainable = False
                
                # 创建新模型
                inputs = Input(shape=(input_shape,))
                features = feature_extractor(inputs)
                x = Dense(32, activation='relu')(features)
                outputs = Dense(1)(x)
                
                self.model = Model(inputs=inputs, outputs=outputs)
            
            else:
                raise ValueError(f"不支持的迁移学习方法: {self.transfer_method}")
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=None, verbose=1):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批量大小
            callbacks: 回调函数列表
            verbose: 详细程度
        
        Returns:
            self: 训练后的模型
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        self._build_model(X_scaled.shape[1])
        
        # 设置回调函数
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10)
            ]
        
        # 训练模型
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted_ = True
        self.history_ = history.history
        
        # 计算训练指标
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"训练完成，训练集指标 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵
        
        Returns:
            numpy.ndarray: 预测值
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_scaled)
        
        # 如果预测结果是二维的，取第一列
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            dict: 评估指标
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred = self.predict(X)
        
        # 计算指标
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练，无法保存")
        
        # 保存Keras模型
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # 保存缩放器和其他属性
        model_data = {
            'scaler': self.scaler,
            'transfer_method': self.transfer_method,
            'freeze_layers': self.freeze_layers,
            'learning_rate': self.learning_rate,
            'is_fitted_': self.is_fitted_,
            'model_path': model_path
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到 {filepath} 和 {model_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            TransferLearningModel: 加载的模型
        """
        # 加载模型数据
        model_data = joblib.load(filepath)
        
        # 加载Keras模型
        model_path = model_data['model_path']
        keras_model = load_model(model_path)
        
        # 创建新实例
        instance = cls(
            base_model=None,  # 不需要基础模型，因为我们直接加载训练好的模型
            transfer_method=model_data['transfer_method'],
            freeze_layers=model_data['freeze_layers'],
            learning_rate=model_data['learning_rate']
        )
        
        # 设置属性
        instance.model = keras_model
        instance.scaler = model_data['scaler']
        instance.is_fitted_ = model_data['is_fitted_']
        
        logger.info(f"从 {filepath} 和 {model_path} 加载模型")
        return instance

class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    集成学习模型
    
    结合多个基础模型的预测结果
    """
    
    def __init__(self, base_models=None, ensemble_method='voting', weights=None, meta_model=None):
        """
        初始化
        
        Args:
            base_models: 基础模型列表，每个元素为(名称, 模型)元组
            ensemble_method: 集成方法，可选'voting'或'stacking'
            weights: 投票权重，仅在ensemble_method='voting'时使用
            meta_model: 元模型，仅在ensemble_method='stacking'时使用
        """
        self.base_models = base_models or []
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_model = meta_model
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            self: 训练后的模型
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建集成模型
        if self.ensemble_method == 'voting':
            self.model = VotingRegressor(
                estimators=self.base_models,
                weights=self.weights
            )
        elif self.ensemble_method == 'stacking':
            meta_model = self.meta_model or Ridge()
            self.model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=meta_model
            )
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        # 计算训练指标
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"训练完成，训练集指标 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵
        
        Returns:
            numpy.ndarray: 预测值
        """
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """
        带不确定性的预测
        
        Args:
            X: 特征矩阵
            n_samples: 采样次数
        
        Returns:
            tuple: (预测均值, 预测标准差)
        """
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 收集每个基础模型的预测
        predictions = []
        for name, model in self.base_models:
            if hasattr(model, 'predict'):
                pred = model.predict(X_scaled)
                predictions.append(pred)
        
        # 转换为数组
        predictions = np.array(predictions)
        
        # 计算均值和标准差
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            dict: 评估指标
        """
        # 预测
        y_pred = self.predict(X)
        
        # 计算指标
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # 计算带不确定性的预测
        mean_pred, std_pred = self.predict_with_uncertainty(X)
        
        # 计算预测区间覆盖率
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        coverage = np.mean((y >= lower_bound) & (y <= upper_bound))
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'uncertainty_mean': np.mean(std_pred),
            'uncertainty_max': np.max(std_pred),
            'coverage_95': coverage
        }
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        joblib.dump(self, filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            EnsembleModel: 加载的模型
        """
        model = joblib.load(filepath)
        logger.info(f"从 {filepath} 加载模型")
        return model

class DomainDecompositionModel(BaseEstimator, RegressorMixin):
    """
    域分解模型
    
    将问题域分解为多个子域，每个子域使用专门的模型
    """
    
    def __init__(self, domain_criteria=None, domain_models=None):
        """
        初始化
        
        Args:
            domain_criteria: 域划分标准，函数或字典
            domain_models: 域模型字典，键为域名称，值为模型
        """
        self.domain_criteria = domain_criteria
        self.domain_models = domain_models or {}
        self.default_model = None
        self.domains_ = None
        self.scaler = StandardScaler()
    
    def _default_domain_criteria(self, X):
        """
        默认域划分标准
        
        Args:
            X: 特征矩阵
        
        Returns:
            dict: 域划分结果，键为域名称，值为布尔索引数组
        """
        # 将X转换为DataFrame以便于访问列
        if not isinstance(X, pd.DataFrame):
            # 假设X的列顺序与训练时相同
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        # 基于渗透率和油相粘度划分域
        domains = {}
        
        # 高渗透率域
        if 'permeability' in X.columns:
            domains['high_permeability'] = X['permeability'] > X['permeability'].median()
        
        # 高粘度域
        if 'oil_viscosity' in X.columns:
            domains['high_viscosity'] = X['oil_viscosity'] > X['oil_viscosity'].median()
        
        # 如果没有找到合适的列，使用简单的随机划分
        if not domains:
            logger.warning("未找到用于域划分的列，使用随机划分")
            n_samples = X.shape[0]
            domains['domain_1'] = np.random.choice([True, False], size=n_samples, p=[0.5, 0.5])
            domains['domain_2'] = ~domains['domain_1']
        
        return domains
    
    def fit(self, X, y):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            self: 训练后的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names_)
        
        # 确定域划分标准
        if self.domain_criteria is None:
            logger.info("使用默认域划分标准")
            self.domain_criteria = self._default_domain_criteria
        
        # 划分域
        if callable(self.domain_criteria):
            domains = self.domain_criteria(X)
        else:
            domains = self.domain_criteria
        
        self.domains_ = domains
        
        # 为每个域训练模型
        for domain_name, domain_mask in domains.items():
            if domain_mask.sum() < 10:
                logger.warning(f"域 {domain_name} 的样本数量过少 ({domain_mask.sum()})，跳过训练")
                continue
            
            logger.info(f"训练域 {domain_name} 的模型，样本数量: {domain_mask.sum()}")
            
            # 获取域的数据
            X_domain = X_scaled_df[domain_mask]
            y_domain = y[domain_mask]
            
            # 创建并训练域模型
            if domain_name in self.domain_models:
                model = clone(self.domain_models[domain_name])
            else:
                model = RandomForestRegressor(random_state=RANDOM_SEED)
            
            model.fit(X_domain, y_domain)
            self.domain_models[domain_name] = model
        
        # 训练默认模型（用于未覆盖的样本）
        self.default_model = RandomForestRegressor(random_state=RANDOM_SEED)
        self.default_model.fit(X_scaled_df, y)
        
        # 计算训练指标
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"训练完成，训练集指标 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵
        
        Returns:
            numpy.ndarray: 预测值
        """
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names_)
        
        # 划分域
        if callable(self.domain_criteria):
            domains = self.domain_criteria(X)
        else:
            domains = self.domains_
        
        # 初始化预测结果
        predictions = np.zeros(X.shape[0])
        covered = np.zeros(X.shape[0], dtype=bool)
        
        # 对每个域进行预测
        for domain_name, domain_mask in domains.items():
            if domain_name not in self.domain_models:
                continue
            
            # 获取域的数据
            X_domain = X_scaled_df[domain_mask]
            
            if X_domain.shape[0] == 0:
                continue
            
            # 预测
            domain_pred = self.domain_models[domain_name].predict(X_domain)
            
            # 更新预测结果
            predictions[domain_mask] = domain_pred
            covered[domain_mask] = True
        
        # 对未覆盖的样本使用默认模型
        if not np.all(covered) and self.default_model is not None:
            uncovered = ~covered
            X_uncovered = X_scaled_df[uncovered]
            
            if X_uncovered.shape[0] > 0:
                default_pred = self.default_model.predict(X_uncovered)
                predictions[uncovered] = default_pred
        
        return predictions
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            dict: 评估指标
        """
        # 预测
        y_pred = self.predict(X)
        
        # 计算总体指标
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        metrics = {
            'overall_r2': r2,
            'overall_rmse': rmse,
            'overall_mae': mae,
            'domain_metrics': {}
        }
        
        # 计算每个域的指标
        if callable(self.domain_criteria):
            domains = self.domain_criteria(X)
        else:
            domains = self.domains_
        
        for domain_name, domain_mask in domains.items():
            if domain_mask.sum() == 0 or domain_name not in self.domain_models:
                continue
            
            # 获取域的数据
            y_domain = y[domain_mask]
            y_pred_domain = y_pred[domain_mask]
            
            # 计算指标
            domain_r2 = r2_score(y_domain, y_pred_domain)
            domain_rmse = np.sqrt(mean_squared_error(y_domain, y_pred_domain))
            domain_mae = mean_absolute_error(y_domain, y_pred_domain)
            
            metrics['domain_metrics'][domain_name] = {
                'r2': domain_r2,
                'rmse': domain_rmse,
                'mae': domain_mae,
                'samples': domain_mask.sum()
            }
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        joblib.dump(self, filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            DomainDecompositionModel: 加载的模型
        """
        model = joblib.load(filepath)
        logger.info(f"从 {filepath} 加载模型")
        return model

class PhysicsInformedNeuralNetwork:
    """
    物理信息神经网络
    
    结合物理约束的神经网络模型
    """
    
    def __init__(self, input_dim, hidden_layers=[64, 32], activation='relu', 
                 physics_weight=0.5, learning_rate=0.001):
        """
        初始化
        
        Args:
            input_dim: 输入维度
            hidden_layers: 隐藏层节点数列表
            activation: 激活函数
            physics_weight: 物理损失权重
            learning_rate: 学习率
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def _build_model(self):
        """构建神经网络模型"""
        # 输入层
        inputs = Input(shape=(self.input_dim,))
        
        # 隐藏层
        x = inputs
        for units in self.hidden_layers:
            x = Dense(units, activation=self.activation, 
                     kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
            x = Dropout(0.2)(x)
        
        # 输出层
        outputs = Dense(1)(x)
        
        # 创建模型
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._combined_loss,
            metrics=['mae']
        )
    
    def _physics_loss(self, y_true, y_pred):
        """
        物理约束损失函数
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            tf.Tensor: 物理损失
        """
        # 这里实现物理约束损失
        # 例如，可以基于物理定律添加约束
        # 简单示例：预测值应该是正的
        return tf.reduce_mean(tf.maximum(0.0, -y_pred))
    
    def _combined_loss(self, y_true, y_pred):
        """
        组合损失函数
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            tf.Tensor: 组合损失
        """
        # MSE损失
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # 物理损失
        phys_loss = self._physics_loss(y_true, y_pred)
        
        # 组合损失
        return (1 - self.physics_weight) * mse_loss + self.physics_weight * phys_loss
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=None, verbose=1):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批量大小
            callbacks: 回调函数列表
            verbose: 详细程度
        
        Returns:
            self: 训练后的模型
        """
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        self._build_model()
        
        # 设置回调函数
        if callbacks is None:
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10)
            ]
        
        # 训练模型
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted_ = True
        self.history_ = history.history
        
        # 计算训练指标
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"训练完成，训练集指标 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵
        
        Returns:
            numpy.ndarray: 预测值
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_scaled)
        
        # 如果预测结果是二维的，取第一列
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            dict: 评估指标
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred = self.predict(X)
        
        # 计算指标
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练，无法保存")
        
        # 保存Keras模型
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # 保存缩放器和其他属性
        model_data = {
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'physics_weight': self.physics_weight,
            'learning_rate': self.learning_rate,
            'feature_names_': self.feature_names_,
            'is_fitted_': self.is_fitted_,
            'model_path': model_path
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到 {filepath} 和 {model_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            PhysicsInformedNeuralNetwork: 加载的模型
        """
        # 加载模型数据
        model_data = joblib.load(filepath)
        
        # 加载Keras模型
        model_path = model_data['model_path']
        
        # 创建新实例
        instance = cls(
            input_dim=model_data['input_dim'],
            hidden_layers=model_data['hidden_layers'],
            activation=model_data['activation'],
            physics_weight=model_data['physics_weight'],
            learning_rate=model_data['learning_rate']
        )
        
        # 设置属性
        instance.model = load_model(model_path, custom_objects={
            '_combined_loss': instance._combined_loss,
            '_physics_loss': instance._physics_loss
        })
        instance.scaler = model_data['scaler']
        instance.feature_names_ = model_data['feature_names_']
        instance.is_fitted_ = model_data['is_fitted_']
        
        logger.info(f"从 {filepath} 和 {model_path} 加载模型")
        return instance

def optimize_hyperparameters(X, y, model_type, n_trials=100, timeout=3600):
    """
    使用Optuna优化超参数
    
    Args:
        X: 特征矩阵
        y: 目标变量
        model_type: 模型类型
        n_trials: 试验次数
        timeout: 超时时间（秒）
    
    Returns:
        dict: 最佳超参数
    """
    logger.info(f"开始优化 {model_type} 模型的超参数")
    
    # 创建目标函数
    def objective(trial):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': RANDOM_SEED
            }
            model = RandomForestRegressor(**params)
        
        elif model_type == 'gradient_boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': RANDOM_SEED
            }
            model = GradientBoostingRegressor(**params)
        
        elif model_type == 'gaussian_process':
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern', 'combined'])
            
            if kernel_type == 'rbf':
                kernel = ConstantKernel() * RBF(length_scale=trial.suggest_float('length_scale', 0.1, 10.0))
            elif kernel_type == 'matern':
                kernel = ConstantKernel() * Matern(
                    length_scale=trial.suggest_float('length_scale', 0.1, 10.0),
                    nu=trial.suggest_float('nu', 0.5, 2.5)
                )
            else:  # combined
                kernel = (
                    ConstantKernel() * RBF(length_scale=trial.suggest_float('rbf_length_scale', 0.1, 10.0)) +
                    ConstantKernel() * Matern(
                        length_scale=trial.suggest_float('matern_length_scale', 0.1, 10.0),
                        nu=trial.suggest_float('nu', 0.5, 2.5)
                    )
                )
            
            params = {
                'kernel': kernel,
                'alpha': trial.suggest_float('alpha', 1e-10, 1e-2, log=True),
                'normalize_y': trial.suggest_categorical('normalize_y', [True, False]),
                'random_state': RANDOM_SEED
            }
            model = GaussianProcessRegressor(**params)
        
        elif model_type == 'neural_network':
            params = {
                'hidden_layer_sizes': (
                    trial.suggest_int('n_units_1', 10, 200),
                    trial.suggest_int('n_units_2', 10, 100)
                ),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'random_state': RANDOM_SEED
            }
            model = MLPRegressor(**params)
        
        elif model_type == 'physics_informed_residual':
            ml_model_type = trial.suggest_categorical('ml_model_type', 
                                                    ['random_forest', 'gradient_boosting', 'neural_network'])
            
            if ml_model_type == 'random_forest':
                ml_model_params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20)
                }
            elif ml_model_type == 'gradient_boosting':
                ml_model_params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10)
                }
            else:  # neural_network
                ml_model_params = {
                    'hidden_layer_sizes': (
                        trial.suggest_int('n_units_1', 10, 200),
                        trial.suggest_int('n_units_2', 10, 100)
                    ),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
                }
            
            params = {
                'ml_model_type': ml_model_type,
                'ml_model_params': ml_model_params
            }
            model = PhysicsInformedResidualModel(**params)
        
        elif model_type == 'ensemble':
            ensemble_method = trial.suggest_categorical('ensemble_method', ['voting', 'stacking'])
            
            # 创建基础模型
            base_models = [
                ('rf', RandomForestRegressor(
                    n_estimators=trial.suggest_int('rf_n_estimators', 50, 200),
                    max_depth=trial.suggest_int('rf_max_depth', 3, 15),
                    random_state=RANDOM_SEED
                )),
                ('gb', GradientBoostingRegressor(
                    n_estimators=trial.suggest_int('gb_n_estimators', 50, 200),
                    learning_rate=trial.suggest_float('gb_learning_rate', 0.01, 0.2),
                    random_state=RANDOM_SEED
                )),
                ('ridge', Ridge(
                    alpha=trial.suggest_float('ridge_alpha', 0.1, 10.0),
                    random_state=RANDOM_SEED
                ))
            ]
            
            if ensemble_method == 'voting':
                weights = [
                    trial.suggest_float('weight_rf', 0.1, 1.0),
                    trial.suggest_float('weight_gb', 0.1, 1.0),
                    trial.suggest_float('weight_ridge', 0.1, 1.0)
                ]
                params = {
                    'base_models': base_models,
                    'ensemble_method': 'voting',
                    'weights': weights
                }
            else:  # stacking
                meta_model = Ridge(alpha=trial.suggest_float('meta_alpha', 0.1, 10.0))
                params = {
                    'base_models': base_models,
                    'ensemble_method': 'stacking',
                    'meta_model': meta_model
                }
            
            model = EnsembleModel(**params)
        
        elif model_type == 'physics_informed_neural_network':
            params = {
                'input_dim': X.shape[1],
                'hidden_layers': [
                    trial.suggest_int('n_units_1', 32, 256),
                    trial.suggest_int('n_units_2', 16, 128)
                ],
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'physics_weight': trial.suggest_float('physics_weight', 0.1, 0.9),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            }
            model = PhysicsInformedNeuralNetwork(**params)
            
            # 对于神经网络模型，使用交叉验证可能太耗时，这里简化为单次训练验证
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
            
            try:
                model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        TFKerasPruningCallback(trial, 'val_loss')
                    ],
                    verbose=0
                )
                
                # 评估
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                return score
            except Exception as e:
                logger.error(f"训练失败: {e}")
                return float('-inf')
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 使用交叉验证评估模型
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()
    
    # 创建学习器
    study = optuna.create_study(direction='maximize')
    
    # 运行优化
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # 获取最佳参数
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"优化完成，最佳R²: {best_value:.4f}")
    logger.info(f"最佳参数: {best_params}")
    
    return best_params

def clone_keras_model(model):
    """
    克隆Keras模型
    
    Args:
        model: Keras模型
    
    Returns:
        tf.keras.Model: 克隆的模型
    """
    # 获取模型配置
    config = model.get_config()
    
    # 克隆模型
    cloned_model = tf.keras.models.model_from_config(config)
    
    # 复制权重
    cloned_model.set_weights(model.get_weights())
    
    return cloned_model

def create_model_factory(model_type, **kwargs):
    """
    创建模型工厂函数
    
    Args:
        model_type: 模型类型
        **kwargs: 模型参数
    
    Returns:
        function: 模型工厂函数
    """
    def model_factory():
        if model_type == 'random_forest':
            return RandomForestRegressor(**kwargs)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**kwargs)
        elif model_type == 'gaussian_process':
            return GaussianProcessRegressor(**kwargs)
        elif model_type == 'neural_network':
            return MLPRegressor(**kwargs)
        elif model_type == 'physics_informed_residual':
            return PhysicsInformedResidualModel(**kwargs)
        elif model_type == 'ensemble':
            return EnsembleModel(**kwargs)
        elif model_type == 'physics_informed_neural_network':
            return PhysicsInformedNeuralNetwork(**kwargs)
        elif model_type == 'domain_decomposition':
            return DomainDecompositionModel(**kwargs)
        elif model_type == 'transfer_learning':
            return TransferLearningModel(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model_factory

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_factory, model_name):
    """
    训练和评估模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        model_factory: 模型工厂函数
        model_name: 模型名称
    
    Returns:
        tuple: (模型, 评估指标)
    """
    logger.info(f"开始训练 {model_name} 模型")
    start_time = time.time()
    
    # 创建模型
    model = model_factory()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 评估模型
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # 计算训练时间
    train_time = time.time() - start_time
    
    # 记录评估指标
    metrics = {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'train_time': train_time
    }
    
    logger.info(f"{model_name} 模型训练完成，耗时 {train_time:.2f} 秒")
    logger.info(f"训练集指标 - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    logger.info(f"测试集指标 - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    
    return model, metrics

def compare_models(X, y, model_factories, test_size=0.2, random_state=RANDOM_SEED):
    """
    比较多个模型的性能
    
    Args:
        X: 特征矩阵
        y: 目标变量
        model_factories: 模型工厂字典，键为模型名称，值为模型工厂函数
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        tuple: (最佳模型, 所有模型的评估指标)
    """
    logger.info(f"开始比较 {len(model_factories)} 个模型")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 训练和评估每个模型
    models = {}
    metrics = {}
    
    for model_name, model_factory in model_factories.items():
        model, model_metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, model_factory, model_name
        )
        
        models[model_name] = model
        metrics[model_name] = model_metrics
    
    # 找出最佳模型
    best_model_name = max(metrics, key=lambda k: metrics[k]['test_r2'])
    best_model = models[best_model_name]
    
    logger.info(f"模型比较完成，最佳模型: {best_model_name}，测试集R²: {metrics[best_model_name]['test_r2']:.4f}")
    
    return best_model, models, metrics

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试代码
    logger.info("测试高级建模模块")
    
    # 创建示例数据
    np.random.seed(RANDOM_SEED)
    n_samples = 100
    X = np.random.rand(n_samples, 5)
    y = 2 * X[:, 0] + 1 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, n_samples)
    
    # 创建数据框
    df = pd.DataFrame(X, columns=['permeability', 'oil_viscosity', 'well_spacing', 'effective_thickness', 'formation_pressure'])
    
    # 测试物理信息残差模型
    model = PhysicsInformedResidualModel()
    model.fit(df, y)
    metrics = model.evaluate(df, y)
    logger.info(f"物理信息残差模型评估指标: {metrics}")
    
    # 测试集成模型
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)),
        ('ridge', Ridge(alpha=1.0, random_state=RANDOM_SEED))
    ]
    
    ensemble = EnsembleModel(base_models=base_models, ensemble_method='voting')
    ensemble.fit(df, y)
    metrics = ensemble.evaluate(df, y)
    logger.info(f"集成模型评估指标: {metrics}")
    
    logger.info("高级建模模块测试完成")
