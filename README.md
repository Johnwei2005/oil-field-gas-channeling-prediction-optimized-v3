# oil-field-gas-channeling-prediction-optimized-v3

基于物理约束残差建模的油田气窜预测系统（优化版v3）- 修复数据处理错误

## 项目结构

- `enhanced_features_optimized.py`: 特征工程优化模块
- `residual_model_optimized.py`: 残差模型优化模块
- `parameter_optimization.py`: 参数优化模块
- `model_test_evaluation.py`: 模型测试与评估模块
- `model_fine_tune.py`: 模型微调模块
- `main.py`: 主程序
- `data_processor.py`: 数据处理模块
- `config.py`: 配置文件
- `run_all.py`: 全流程自动化脚本

## 性能指标

- R²值: 0.9-0.95
- RMSE (均方根误差): 约0.02
- MAE (平均绝对误差): 约0.01

## 使用方法

### 全流程自动运行

使用全流程自动化脚本可以一键完成从数据处理到模型评估的全部流程：

```bash
# 使用默认设置运行全流程（高斯过程模型）
python run_all.py

# 指定模型类型
python run_all.py --model-type random_forest

# 跳过特定步骤
python run_all.py --skip-steps data,fine_tune
```

### 单独运行各模块

如果需要单独运行各个模块，可以使用以下命令：

```bash
# 数据处理
python data_processor.py

# 训练模型
python main.py --train --model-type gaussian_process

# 使用模型进行预测
python main.py --predict --input-file data/raw/CO2气窜原始表.csv --output-file results/predictions.csv

# 模型微调
python model_fine_tune.py

# 模型评估
python model_test_evaluation.py
```

## 目录结构

```
oil-field-gas-channeling-prediction-optimized-v3/
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   │   └── CO2气窜原始表.csv       # 原始数据文件
│   └── processed/                # 处理后的数据
├── docs/                         # 文档目录
├── logs/                         # 日志目录
├── models/                       # 模型目录
├── notebooks/                    # Jupyter笔记本目录
├── results/                      # 结果目录
├── .vscode/                      # VSCode配置
├── config.py                     # 配置文件
├── data_processor.py             # 数据处理模块
├── enhanced_features_optimized.py # 特征工程优化模块
├── main.py                       # 主程序
├── model_fine_tune.py            # 模型微调模块
├── model_test_evaluation.py      # 模型测试与评估模块
├── parameter_optimization.py     # 参数优化模块
├── residual_model_optimized.py   # 残差模型优化模块
├── run_all.py                    # 全流程自动化脚本
├── run_model_optimization_optimized.py # 模型优化运行脚本
└── README.md                     # 项目说明文件
```

## 环境要求

- Python 3.8+
- 依赖包：见 requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```
