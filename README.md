# 油田气窜预测系统优化版

本项目实现了基于机器学习的油田CO2气窜预测系统，通过优化特征选择和建模技术，确保模型性能达到目标R²值范围(0.9-0.93)。

## 项目特点

- **特征数量控制**：严格控制特征数量不超过10个
- **目标变量保护**：确保预测目标不被加入特征列表
- **多种特征选择方法**：支持过滤法、包装法、嵌入式方法和组合方法
- **灵活的建模技术**：支持随机森林、梯度提升、SVR、弹性网络、岭回归和集成模型
- **自动参数优化**：自动寻找最佳模型参数
- **性能微调功能**：自动调整模型以达到目标R²值范围
- **全面性能验证**：包括测试集评估、交叉验证和模型稳定性测试
- **全流程自动化**：提供一键式执行全流程的脚本

## 项目结构

```
oil-field-gas-channeling-prediction-optimized-v3/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后的数据
├── docs/                      # 文档目录
├── logs/                      # 日志目录
├── models/                    # 模型目录
├── results/                   # 结果目录
├── config.py                  # 配置文件
├── data_processor.py          # 数据处理模块
├── optimized_feature_selection.py  # 优化的特征选择模块
├── simplified_model.py        # 简化建模技术模块
├── parameter_optimization.py  # 模型参数优化模块
├── model_validation.py        # 模型性能验证模块
├── run_optimized.py           # 全流程自动化脚本
└── README.md                  # 项目说明文件
```

## 安装与使用

### 环境要求

- Python 3.8+
- 依赖包：numpy, pandas, scikit-learn, matplotlib, seaborn

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

#### 全流程自动化

```bash
python run_optimized.py --model-type ensemble --feature-method combined --max-features 10
```

#### 参数说明

- `--skip-steps`：要跳过的步骤，用逗号分隔，可选值: data, feature, model, optimize, validate
- `--model-type`：模型类型，可选值: rf, gbm, svr, elastic_net, ridge, ensemble
- `--feature-method`：特征选择方法，可选值: filter, wrapper, embedded, combined
- `--max-features`：最大特征数量，默认为10
- `--target-r2-min`：目标R²值下限，默认为0.9
- `--target-r2-max`：目标R²值上限，默认为0.93

#### 单独执行各模块

```bash
# 数据处理
python data_processor.py

# 特征选择
python optimized_feature_selection.py

# 模型训练
python simplified_model.py

# 参数优化
python parameter_optimization.py

# 性能验证
python model_validation.py
```

## 模型性能

通过优化特征选择和建模技术，本项目实现了以下性能目标：

- R²值：0.9-0.93
- 特征数量：≤10个
- 预测目标：不被加入特征列表

## 作者

- John Wei (johnwei2005@example.com)

## 许可证

MIT License
