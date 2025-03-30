# 油田气窜预测系统项目文档

## 目录
1. [项目概述](#项目概述)
2. [项目结构](#项目结构)
3. [安装与环境配置](#安装与环境配置)
4. [数据处理流程](#数据处理流程)
5. [特征工程](#特征工程)
6. [模型架构](#模型架构)
7. [可视化功能](#可视化功能)
8. [使用说明](#使用说明)
9. [API参考](#API参考)
10. [常见问题](#常见问题)

## 项目概述

本项目是一个基于机器学习和物理信息的油田CO2气窜预测系统，旨在通过结合地质物理特性和先进的机器学习技术，准确预测油田开发过程中的气窜现象，为油田开发决策提供科学依据。

气窜现象是油田开发中的常见问题，会导致注入气体无法有效驱油，降低采收率，增加生产成本。准确预测气窜发生的可能性和严重程度，对于优化注采参数、调整开发方案具有重要意义。

本系统的主要特点：

1. **物理信息与机器学习结合**：融合达西定律等物理模型与先进机器学习算法，提高预测精度
2. **多层次特征工程**：从原始数据中提取物理意义明确的特征，增强模型解释性
3. **高级可视化分析**：提供丰富的可视化工具，帮助理解数据特征和模型预测结果
4. **集成多种先进模型**：实现物理信息残差模型、迁移学习、域分解等多种先进建模技术
5. **不确定性量化**：提供预测结果的不确定性评估，增强决策可靠性

## 项目结构

```
oil-field-gas-channeling-prediction-optimized-v3/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后的数据
│   ├── reports/               # 数据质量报告
│   └── scalers/               # 特征缩放器
├── docs/                      # 文档
│   ├── project_overview.md    # 项目概述
│   └── api_reference.md       # API参考
├── logs/                      # 日志文件
├── models/                    # 模型文件
├── notebooks/                 # Jupyter笔记本
├── results/                   # 结果输出
│   ├── figures/               # 图表
│   └── reports/               # 报告
├── config.py                  # 配置文件
├── data_processor.py          # 数据处理模块
├── data_processor_improved.py # 改进的数据处理模块
├── enhanced_features_optimized.py # 特征工程模块
├── main.py                    # 主程序
├── model_test_evaluation.py   # 模型测试与评估
├── advanced_modeling.py       # 高级建模技术
├── visualization_enhanced.py  # 增强可视化功能
├── run_all.py                 # 全流程启动脚本
└── requirements.txt           # 依赖包列表
```

## 安装与环境配置

### 系统要求
- Python 3.8+
- 64位操作系统（Windows/Linux/MacOS）
- 至少4GB内存（推荐8GB以上）
- 至少2GB可用磁盘空间

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/Johnwei2005/oil-field-gas-channeling-prediction-optimized-v3.git
cd oil-field-gas-channeling-prediction-optimized-v3
```

2. 创建虚拟环境（可选但推荐）
```bash
python -m venv venv
# Windows激活虚拟环境
venv\Scripts\activate
# Linux/MacOS激活虚拟环境
source venv/bin/activate
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

4. 验证安装
```bash
python run_all.py --test
```

如果安装成功，将看到测试通过的消息。

## 数据处理流程

数据处理是本项目的关键环节，包括数据加载、数据验证、缺失值处理、异常值处理、特征标准化等步骤。

### 数据格式

系统支持以下格式的输入数据：
- CSV文件（支持中文编码）
- Excel文件（.xlsx或.xls）

原始数据应包含以下关键字段：
- 区块：油田区块名称
- 地层温度℃：油藏温度
- 地层压力mpa：油藏压力
- 注气前地层压力mpa：注气前油藏压力
- 压力水平：压力水平指标
- 渗透率md：岩石渗透率
- 地层原油粘度mpas：原油粘度
- 地层原油密度g/cm3：原油密度
- 井组有效厚度m：有效厚度
- 注气井压裂：是否进行压裂（是/否）
- 井距m：井距
- 孔隙度/%：岩石孔隙度
- 注入前含油饱和度/%：含油饱和度
- pv数：PV数（目标变量）

### 数据处理步骤

1. **数据加载**：支持多种编码格式，自动检测并处理中文编码问题
2. **数据验证**：生成数据质量报告，包括基本统计信息、缺失值分析、异常值检测等
3. **智能缺失值处理**：
   - 分类变量：使用众数填充
   - 数值变量：根据分布特性选择均值或中位数填充
   - 关键特征：使用KNN方法进行高级填充
4. **高级异常值处理**：
   - 区分极端异常值和中度异常值
   - 对极端异常值进行截断处理
   - 对中度异常值进行Winsorization处理
5. **特征标准化**：
   - 支持StandardScaler和RobustScaler两种标准化方法
   - 保存标准化器以便后续转换

### 数据质量报告

系统会自动生成数据质量报告，包括：
- 基本统计信息（均值、中位数、标准差等）
- 缺失值分析（缺失比例、缺失模式）
- 异常值检测结果
- 特征分布可视化
- 相关性分析

## 特征工程

特征工程模块负责从原始数据中提取和创建有物理意义的特征，增强模型的预测能力和解释性。

### 基础特征

基础特征直接来源于原始数据，包括：
- 渗透率（permeability）
- 油相粘度（oil_viscosity）
- 井距（well_spacing）
- 有效厚度（effective_thickness）
- 地层压力（formation_pressure）
- 孔隙度（porosity）
- 含油饱和度（oil_saturation）
- 压裂状态（fracturing）

### 物理信息特征

基于流体力学和油藏工程原理创建的特征：
- 流动能力指数（mobility_index）：渗透率与油相粘度的比值
- 几何形状因子（geometric_factor）：井距与有效厚度的比值
- 驱动力指数（driving_force_index）：地层压力与参考压力的比值
- 孔喉半径估计（pore_throat_radius）：基于Kozeny-Carman方程
- 毛细管数（capillary_number）：表征粘性力与毛细管力的比值
- 流动单元指数（flow_unit_index）：表征储层质量的综合指标

### 交互特征

捕捉特征间相互作用的特征：
- 渗透率与粘度的交互（perm_visc_interaction）
- 几何形状与压力的交互（geom_press_interaction）
- 孔隙度与饱和度的交互（poro_sat_interaction）

### 特征选择

系统使用多种方法进行特征选择：
- 基于相关性的筛选
- 基于特征重要性的筛选
- 基于物理意义的专家选择

## 模型架构

本系统实现了多种先进的建模技术，以提高预测精度和模型解释性。

### 物理信息残差模型

物理信息残差模型结合物理模型和机器学习模型，使用机器学习模型预测物理模型的残差。

工作流程：
1. 使用简化的物理模型（基于达西定律）进行初步预测
2. 计算物理模型预测值与实际值之间的残差
3. 训练机器学习模型预测这些残差
4. 最终预测 = 物理模型预测 + 残差预测

优势：
- 保留物理模型的解释性
- 利用机器学习模型捕捉复杂非线性关系
- 减少对数据的依赖

### 迁移学习模型

迁移学习模型利用在相似问题上预训练的模型知识，应用到目标问题上。

实现方法：
1. 微调（Fine-tuning）：使用预训练模型的权重，但允许更新
2. 特征迁移（Feature Transfer）：使用预训练模型作为特征提取器

优势：
- 减少对大量标记数据的需求
- 加速模型收敛
- 提高小样本学习效果

### 集成学习模型

集成学习模型结合多个基础模型的预测结果，提高整体预测性能。

实现方法：
1. 投票法（Voting）：加权平均多个模型的预测结果
2. 堆叠法（Stacking）：使用元模型组合基础模型的预测

优势：
- 减少过拟合风险
- 提高预测稳定性
- 量化预测不确定性

### 域分解模型

域分解模型将问题域分解为多个子域，每个子域使用专门的模型。

实现方法：
1. 基于物理特性（如渗透率、油相粘度）划分域
2. 为每个域训练专门的模型
3. 预测时根据样本特征选择合适的域模型

优势：
- 处理数据异质性
- 提高特定条件下的预测精度
- 增强模型适应性

### 物理信息神经网络

物理信息神经网络在神经网络训练过程中融入物理约束。

实现方法：
1. 设计包含物理约束的损失函数
2. 在网络结构中嵌入物理知识
3. 使用物理启发的正则化方法

优势：
- 确保预测结果符合物理规律
- 提高小样本下的泛化能力
- 增强模型解释性

## 可视化功能

本系统提供丰富的可视化功能，帮助理解数据特征和模型预测结果。

### 数据可视化

- **特征分布图**：展示各特征的分布情况
- **相关性热图**：展示特征间的相关关系
- **特征对关系图**：展示重要特征对之间的关系
- **3D特征重要性图**：以3D方式展示重要特征与目标变量的关系

### 模型性能可视化

- **预测值与实际值对比图**：直观展示预测精度
- **残差分析图**：分析预测误差的分布和模式
- **学习曲线**：展示模型训练过程中的性能变化
- **模型比较图**：比较不同模型在各项评估指标上的表现

### 交互式可视化

- **交互式数据探索**：支持缩放、筛选、悬停查看详情
- **交互式预测结果分析**：动态分析预测结果和误差
- **综合报告仪表板**：集成多种可视化于一体的报告页面

### 可视化输出格式

- 静态图像（PNG、JPG、PDF）
- 交互式HTML文件
- 综合报告（HTML格式）

## 使用说明

### 基本使用流程

1. **准备数据**：将数据文件放入`data/raw/`目录
2. **运行全流程**：执行`run_all.py`脚本
```bash
python run_all.py
```
3. **查看结果**：结果将保存在`results/`目录

### 命令行参数

`run_all.py`脚本支持以下命令行参数：

- `--data-file`：指定数据文件路径，默认使用`data/raw/CO2气窜原始表.csv`
- `--model-type`：指定模型类型，可选值包括：
  - `random_forest`：随机森林（默认）
  - `gradient_boosting`：梯度提升树
  - `physics_informed`：物理信息残差模型
  - `ensemble`：集成模型
  - `domain_decomposition`：域分解模型
  - `physics_nn`：物理信息神经网络
- `--skip-steps`：跳过特定步骤，多个步骤用逗号分隔，可选值：
  - `data`：数据处理
  - `feature`：特征工程
  - `train`：模型训练
  - `evaluate`：模型评估
  - `visualize`：可视化
- `--test`：运行测试模式，使用小数据集快速验证
- `--optimize`：执行超参数优化
- `--verbose`：显示详细输出

示例：
```bash
# 使用物理信息残差模型，跳过数据处理步骤
python run_all.py --model-type physics_informed --skip-steps data

# 执行超参数优化
python run_all.py --optimize

# 测试模式
python run_all.py --test
```

### 高级用法

#### 自定义物理模型

可以通过继承`PhysicsInformedResidualModel`类并重写`_default_physics_model`方法来自定义物理模型：

```python
from advanced_modeling import PhysicsInformedResidualModel

class CustomPhysicsModel(PhysicsInformedResidualModel):
    def _default_physics_model(self, X):
        # 实现自定义物理模型
        # ...
        return physics_pred
```

#### 集成多个模型

可以使用`EnsembleModel`类集成多个模型：

```python
from advanced_modeling import EnsembleModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# 创建基础模型
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gb', GradientBoostingRegressor(n_estimators=100)),
    ('ridge', Ridge(alpha=1.0))
]

# 创建集成模型
ensemble = EnsembleModel(
    base_models=base_models,
    ensemble_method='voting',
    weights=[0.5, 0.3, 0.2]
)

# 训练和预测
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

#### 自定义可视化

可以使用`visualization_enhanced.py`模块创建自定义可视化：

```python
from visualization_enhanced import plot_feature_importance, create_interactive_visualization

# 绘制特征重要性图
plot_feature_importance(
    feature_names=['feature1', 'feature2', 'feature3'],
    importances=[0.5, 0.3, 0.2],
    title="我的特征重要性图",
    output_path="results/my_feature_importance.png"
)

# 创建交互式可视化
create_interactive_visualization(
    df=my_dataframe,
    target_col='target',
    selected_features=['feature1', 'feature2', 'feature3'],
    output_path="results/my_interactive_viz.html"
)
```

## API参考

### 数据处理模块

```python
# 加载数据
from data_processor_improved import load_data
df = load_data(data_path="path/to/data.csv")

# 预处理数据
from data_processor_improved import preprocess_data
df_processed = preprocess_data(df, normalize=True, normalization_method='robust')

# 加载并预处理数据
from data_processor_improved import load_and_preprocess_data
df_processed = load_and_preprocess_data(
    data_path="path/to/data.csv",
    save_processed=True,
    normalize=True,
    normalization_method='standard'
)
```

### 特征工程模块

```python
# 创建物理信息特征
from enhanced_features_optimized import create_physics_informed_features
df_with_features = create_physics_informed_features(df)

# 特征选择
from enhanced_features_optimized import select_features
selected_features = select_features(df, target_col='PV_number', method='correlation')
```

### 高级建模模块

```python
# 物理信息残差模型
from advanced_modeling import PhysicsInformedResidualModel
model = PhysicsInformedResidualModel(ml_model_type='random_forest')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 集成模型
from advanced_modeling import EnsembleModel
ensemble = EnsembleModel(base_models=base_models, ensemble_method='stacking')
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

# 域分解模型
from advanced_modeling import DomainDecompositionModel
domain_model = DomainDecompositionModel()
domain_model.fit(X_train, y_train)
predictions = domain_model.predict(X_test)

# 物理信息神经网络
from advanced_modeling import PhysicsInformedNeuralNetwork
pinn = PhysicsInformedNeuralNetwork(input_dim=X_train.shape[1])
pinn.fit(X_train, y_train, epochs=100, batch_size=32)
predictions = pinn.predict(X_test)

# 超参数优化
from advanced_modeling import optimize_hyperparameters
best_params = optimize_hyperparameters(X, y, model_type='random_forest', n_trials=100)
```

### 可视化模块

```python
# 预测值与实际值对比图
from visualization_enhanced import plot_prediction_vs_actual
plot_prediction_vs_actual(y_true, y_pred, "模型预测结果")

# 残差分析图
from visualization_enhanced import plot_residual_analysis
plot_residual_analysis(y_true, y_pred, "残差分析")

# 特征重要性图
from visualization_enhanced import plot_feature_importance
plot_feature_importance(feature_names, importances, "特征重要性")

# 交互式可视化
from visualization_enhanced import create_interactive_visualization
create_interactive_visualization(df, target_col, selected_features)

# 生成综合报告
from visualization_enhanced import generate_comprehensive_report
report_files = generate_comprehensive_report(
    model_results, df, target_col, selected_features, y_true, y_pred
)
```

## 常见问题

### 数据相关问题

**Q: 系统支持哪些数据格式？**

A: 系统主要支持CSV和Excel格式的数据文件，推荐使用CSV格式。对于中文编码，系统会自动尝试多种编码方式（gbk, utf-8, gb18030, latin1）。

**Q: 如何处理缺失数据？**

A: 系统提供了智能缺失值处理功能，会根据数据分布特性选择合适的填充方法。对于分类变量使用众数填充，对于数值变量根据分布偏度选择均值或中位数填充，对于关键特征可以使用KNN方法进行高级填充。

**Q: 如何处理异常值？**

A: 系统使用IQR方法检测异常值，并区分极端异常值和中度异常值。对极端异常值进行截断处理，对中度异常值进行Winsorization处理。

### 模型相关问题

**Q: 如何选择最合适的模型？**

A: 系统提供了模型比较功能，可以通过`compare_models`函数比较多个模型的性能。一般来说，物理信息残差模型和集成模型在大多数情况下表现较好，但具体问题需要具体分析。

**Q: 如何解释模型预测结果？**

A: 系统提供了多种可视化工具帮助解释模型预测结果，包括特征重要性图、残差分析图等。物理信息残差模型和物理信息神经网络也具有较好的解释性，可以分析物理部分和机器学习部分的贡献。

**Q: 如何处理过拟合问题？**

A: 可以通过以下方法减轻过拟合：
1. 使用集成模型
2. 增加正则化强度
3. 减少模型复杂度
4. 使用物理信息约束
5. 增加训练数据

### 系统相关问题

**Q: 如何加速模型训练？**

A: 可以通过以下方法加速训练：
1. 使用`--skip-steps`参数跳过不必要的步骤
2. 减少模型复杂度
3. 使用更小的数据集进行初步测试
4. 使用更强大的硬件（如GPU）

**Q: 如何保存和加载模型？**

A: 所有模型类都提供了`save`和`load`方法：
```python
# 保存模型
model.save("models/my_model.pkl")

# 加载模型
from advanced_modeling import PhysicsInformedResidualModel
loaded_model = PhysicsInformedResidualModel.load("models/my_model.pkl")
```

**Q: 系统是否支持增量学习？**

A: 目前系统不直接支持增量学习，但可以通过保存预处理器和模型，然后在新数据上重新训练来实现类似功能。未来版本可能会添加增量学习支持。
