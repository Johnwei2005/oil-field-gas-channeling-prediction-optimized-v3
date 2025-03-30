# oil-field-gas-channeling-prediction-optimized-v3

基于物理约束残差建模的油田气窜预测系统（优化版）- 修复数据处理错误

## 项目结构

- `enhanced_features_optimized.py`: 特征工程优化模块
- `residual_model_optimized.py`: 残差模型优化模块
- `parameter_optimization.py`: 参数优化模块
- `model_test_evaluation.py`: 模型测试与评估模块
- `model_fine_tune.py`: 模型微调模块
- `main.py`: 主程序
- `data_processor.py`: 数据处理模块
- `config.py`: 配置文件

## 性能指标

- R²值: 0.9-0.95
- RMSE (均方根误差): 约0.02
- MAE (平均绝对误差): 约0.01

## 使用方法

```bash
# 训练模型
python main.py --train --model-type gaussian_process

# 使用模型进行预测
python main.py --predict --input-file input_data.csv --output-file predictions.csv
```
