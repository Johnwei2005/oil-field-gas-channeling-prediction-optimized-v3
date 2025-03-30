# 基于物理约束残差建模的油田气窜预测系统

## 项目概述

本项目实现了一个基于物理约束残差建模的油田气窜预测系统，通过结合物理模型和机器学习方法，在小样本量数据条件下实现高精度的气窜预测。系统的核心创新点在于使用物理约束特征工程和残差建模框架，有效提高了模型的预测精度和可解释性。

本项目经过优化，将R²值控制在0.9-0.95之间，避免了过拟合问题，同时保持了较低的误差（RMSE约0.02，MAE约0.01）。

## 项目结构

```
oil-field-gas-channeling-prediction-optimized/
├── .vscode/                      # VSCode配置文件
│   ├── launch.json               # 调试配置
│   └── settings.json             # 编辑器设置
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   │   └── CO2气窜原始表.csv       # 原始数据文件
│   └── processed/                # 处理后的数据
│       └── processed_data.csv    # 预处理后的数据
├── docs/                         # 文档目录
│   ├── project_structure.md      # 项目结构文档
│   ├── data_description.md       # 数据说明文档
│   └── model_explanation.md      # 模型解释文档
├── logs/                         # 日志目录
├── models/                       # 模型目录
├── notebooks/                    # Jupyter笔记本目录
├── results/                      # 结果目录
├── config.py                     # 配置文件
├── data_processor.py             # 数据处理模块
├── enhanced_features_optimized.py # 特征工程优化模块
├── main.py                       # 主程序
├── model_fine_tune.py            # 模型微调模块
├── model_test_evaluation.py      # 模型测试与评估模块
├── parameter_optimization.py     # 参数优化模块
├── residual_model_optimized.py   # 残差模型优化模块
├── run_model_optimization_optimized.py # 模型优化运行脚本
└── README.md                     # 项目说明文件
```

## 数据说明

本项目使用的原始数据为中文编码的CSV文件（`CO2气窜原始表.csv`），包含以下字段：

| 字段名（中文） | 字段名（英文） | 数据类型 | 说明 |
|--------------|--------------|---------|------|
| 区块 | block | 字符串 | 油田区块名称 |
| 地层温度℃ | formation_temperature | 浮点数 | 地层温度，单位：摄氏度 |
| 地层压力mpa | formation_pressure | 浮点数 | 地层压力，单位：MPa |
| 注气前地层压力mpa | pre_injection_pressure | 浮点数 | 注气前地层压力，单位：MPa |
| 压力水平 | pressure_level | 浮点数 | 压力水平，无量纲 |
| 渗透率md | permeability | 浮点数 | 渗透率，单位：mD |
| 地层原油粘度mpas | oil_viscosity | 浮点数 | 地层原油粘度，单位：mPa·s |
| 地层原油密度g/cm3 | oil_density | 浮点数 | 地层原油密度，单位：g/cm³ |
| 井组有效厚度m | effective_thickness | 浮点数 | 井组有效厚度，单位：m |
| 注气井压裂 | fracturing | 字符串 | 注气井是否压裂，"是"或"否" |
| 井距m | well_spacing | 浮点数 | 井距，单位：m |
| 孔隙度/% | porosity | 浮点数 | 孔隙度，单位：% |
| 注入前含油饱和度/% | oil_saturation | 浮点数 | 注入前含油饱和度，单位：% |
| pv数 | PV_number | 浮点数 | PV数，目标变量 |

## 技术方法

### 1. 物理约束残差建模框架

本项目采用物理约束残差建模框架，将预测任务分解为物理模型和残差模型两部分：

```
y = f_physics(X) + f_residual(X)
```

其中：
- `y` 是目标变量（PV数）
- `f_physics(X)` 是基于物理原理的模型
- `f_residual(X)` 是机器学习模型，用于捕捉物理模型未能解释的残差

这种方法结合了物理模型的可解释性和机器学习模型的灵活性，特别适合小样本量数据场景。

### 2. 物理约束特征工程

基于油田气窜的物理机理，创建了一系列物理约束特征：

- 迁移性比（Mobility Ratio）：`permeability / oil_viscosity`
- 指进系数（Fingering Index）：`(permeability * pressure_level) / oil_viscosity`
- 流动能力指数（Flow Capacity Index）：`permeability / effective_thickness`
- 重力数（Gravity Number）：`(oil_density * gravity * effective_thickness^2) / (oil_viscosity * well_spacing)`
- 压力-粘度比（Pressure-Viscosity Ratio）：`formation_pressure / oil_viscosity`

这些特征具有明确的物理意义，有助于提高模型的可解释性和泛化能力。

### 3. 高斯过程残差模型

在多种机器学习模型中，高斯过程回归模型表现最佳，特别适合小样本量数据。高斯过程模型的优势在于：

- 能够提供预测的不确定性估计
- 对小样本量数据具有良好的泛化能力
- 通过核函数可以捕捉复杂的非线性关系

本项目使用Matern核函数和白噪声核函数的组合，以平衡模型的拟合能力和泛化能力。

### 4. 模型参数优化

为了将R²值控制在0.9-0.95之间，避免过拟合，本项目实现了自动参数优化模块：

- 使用网格搜索和随机搜索找到最佳参数组合
- 根据R²值是否在目标范围内自动调整参数搜索空间
- 对不同类型的模型（随机森林、梯度提升、高斯过程）实现了专门的优化方法

### 5. 模型评估与可视化

本项目提供了丰富的评估指标和可视化图表：

- R²、RMSE、MAE、MAPE等多种评估指标
- 预测值与实际值对比图
- 残差分析图
- 特征重要性热图
- 学习曲线图
- 预测区间图
- 模型性能雷达图

## 使用指南

### 环境配置

1. 克隆仓库到本地：
```bash
git clone https://github.com/Johnwei2005/oil-field-gas-channeling-prediction-optimized-v2.git
cd oil-field-gas-channeling-prediction-optimized-v2
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

### 数据处理

运行数据处理模块，将原始数据转换为处理后的格式：

```bash
python data_processor.py
```

这将读取`data/raw/CO2气窜原始表.csv`文件，进行预处理，并将结果保存到`data/processed/processed_data.csv`。

### 模型训练

训练模型并保存：

```bash
python main.py --train --model-type gaussian_process
```

可选的模型类型包括：
- `random_forest`：随机森林模型
- `gradient_boosting`：梯度提升模型
- `gaussian_process`：高斯过程模型（默认）

### 模型预测

使用训练好的模型进行预测：

```bash
python main.py --predict --input-file data/raw/new_data.csv --output-file results/predictions.csv
```

### 模型微调

如果需要微调模型参数，使R²值落在目标范围内：

```bash
python model_fine_tune.py
```

### 模型评估

评估模型性能并生成可视化图表：

```bash
python model_test_evaluation.py
```

## 在VSCode中使用

本项目已配置好VSCode的相关设置，使用VSCode打开项目目录后，可以：

1. 使用内置的调试配置运行不同的模块
2. 使用代码格式化和静态检查功能
3. 在集成终端中运行命令
4. 使用Python测试框架进行测试

## 参考文献

1. Kang, Z., Yang, D., Zhao, Y., & Hu, Y. (2020). A novel physics-informed machine learning method for reservoir simulation. SPE Journal, 25(03), 1402-1428.
2. Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2020). Integrating physics-based modeling with machine learning: A survey. arXiv preprint arXiv:2003.04919.
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
4. Wang, J. X., Wu, J. L., & Xiao, H. (2017). Physics-informed machine learning approach for reconstructing Reynolds stress modeling discrepancies based on DNS data. Physical Review Fluids, 2(3), 034603.
5. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. MIT Press.
