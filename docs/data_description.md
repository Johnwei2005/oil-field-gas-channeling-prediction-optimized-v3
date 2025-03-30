# 数据说明文档

## 原始数据概述

本项目使用的原始数据为`CO2气窜原始表.csv`，这是一个包含油田气窜相关特征和目标变量的数据集。数据采用GBK编码，包含70个样本和14个特征。

## 数据字段详解

| 字段名（中文） | 字段名（英文） | 数据类型 | 单位 | 说明 |
|--------------|--------------|---------|------|------|
| 区块 | block | 字符串 | - | 油田区块名称，如"庄13块"、"马3块"等 |
| 地层温度℃ | formation_temperature | 浮点数 | ℃ | 地层温度，范围约70-100℃ |
| 地层压力mpa | formation_pressure | 浮点数 | MPa | 地层压力，范围约14-22MPa |
| 注气前地层压力mpa | pre_injection_pressure | 浮点数 | MPa | 注气前地层压力，范围约4-9MPa |
| 压力水平 | pressure_level | 浮点数 | - | 压力水平，无量纲，范围约0.2-0.6 |
| 渗透率md | permeability | 浮点数 | mD | 渗透率，范围约15-200mD |
| 地层原油粘度mpas | oil_viscosity | 浮点数 | mPa·s | 地层原油粘度，范围约4-36mPa·s |
| 地层原油密度g/cm3 | oil_density | 浮点数 | g/cm³ | 地层原油密度，范围约0.75-0.88g/cm³ |
| 井组有效厚度m | effective_thickness | 浮点数 | m | 井组有效厚度，范围约4-9m |
| 注气井压裂 | fracturing | 字符串 | - | 注气井是否压裂，"是"或"否" |
| 井距m | well_spacing | 浮点数 | m | 井距，范围约160-380m |
| 孔隙度/% | porosity | 浮点数 | % | 孔隙度，范围约16-24% |
| 注入前含油饱和度/% | oil_saturation | 浮点数 | % | 注入前含油饱和度，范围约57-66% |
| pv数 | PV_number | 浮点数 | - | PV数，目标变量，范围约0.004-0.08 |

## 数据预处理流程

原始数据经过以下预处理步骤：

1. **编码处理**：自动检测并处理中文编码（GBK、UTF-8、GB18030等）
2. **缺失值处理**：检测缺失值并使用均值填充
3. **异常值处理**：使用IQR方法检测异常值，并将其限制在合理范围内
4. **分类变量转换**：将"注气井压裂"列从"是/否"转换为1/0
5. **列名转换**：将中文列名转换为英文，便于后续处理
6. **保存处理后的数据**：将处理后的数据保存为UTF-8编码的CSV文件

## 数据加载示例

```python
# 使用data_processor模块加载数据
from data_processor import load_data, preprocess_data

# 加载原始数据
raw_data = load_data()
print("原始数据形状:", raw_data.shape)

# 预处理数据
processed_data = preprocess_data(raw_data)
print("处理后数据形状:", processed_data.shape)
print("处理后列名:", processed_data.columns.tolist())
```

## 特征工程

基于原始数据，我们创建了以下物理约束特征：

1. **迁移性比（Mobility Ratio）**：
   - 计算公式：`permeability / oil_viscosity`
   - 物理意义：表示油相与气相的流动能力比值，影响气窜的发生和发展

2. **指进系数（Fingering Index）**：
   - 计算公式：`(permeability * pressure_level) / oil_viscosity`
   - 物理意义：表示气相在油相中形成指进的趋势，值越大指进越严重

3. **流动能力指数（Flow Capacity Index）**：
   - 计算公式：`permeability / effective_thickness`
   - 物理意义：表示单位厚度的流动能力，影响气体在垂向上的分布

4. **重力数（Gravity Number）**：
   - 计算公式：`(oil_density * 9.8 * effective_thickness^2) / (oil_viscosity * well_spacing)`
   - 物理意义：表示重力与粘性力的比值，影响气体在垂向上的分布

5. **压力-粘度比（Pressure-Viscosity Ratio）**：
   - 计算公式：`formation_pressure / oil_viscosity`
   - 物理意义：表示驱动力与阻力的比值，影响气体的流动速度

6. **井距-厚度比（Well Spacing-Thickness Ratio）**：
   - 计算公式：`well_spacing / effective_thickness`
   - 物理意义：表示水平与垂直方向的尺度比，影响气体的流动路径

7. **渗透率-粘度比（Permeability-Viscosity Ratio）**：
   - 计算公式：`permeability / oil_viscosity`
   - 物理意义：表示介质对流体的导流能力，是气窜发生的关键因素

8. **温度-粘度因子（Temperature-Viscosity Factor）**：
   - 计算公式：`formation_temperature / oil_viscosity`
   - 物理意义：表示温度对粘度的影响，温度越高粘度越低，气窜风险越大

9. **压力梯度（Pressure Gradient）**：
   - 计算公式：`(formation_pressure - pre_injection_pressure) / well_spacing`
   - 物理意义：表示单位距离的压力变化，是气体流动的驱动力

10. **饱和度-孔隙度比（Saturation-Porosity Ratio）**：
    - 计算公式：`oil_saturation / porosity`
    - 物理意义：表示孔隙中油相的占比，影响气体的流动空间

## 特征选择

为了避免维度灾难和过拟合，我们限制特征数量为10个，并使用以下方法选择最优特征：

1. **互信息法（Mutual Information）**：评估特征与目标变量之间的相互依赖性
2. **随机森林特征重要性**：使用随机森林模型评估特征重要性
3. **Lasso正则化**：使用L1正则化筛选特征
4. **混合方法**：综合以上三种方法的结果，选择最优特征集

最终选择的10个关键特征为：
- 渗透率(permeability)
- 油相粘度(oil_viscosity)
- 井距(well_spacing)
- 有效厚度(effective_thickness)
- 地层压力(formation_pressure)
- 迁移性比(mobility_ratio)
- 指进系数(fingering_index)
- 流动能力指数(flow_capacity_index)
- 重力数(gravity_number)
- 压力-粘度比(pressure_viscosity_ratio)

## 数据可视化

数据处理和特征工程后，我们生成了以下可视化图表：

1. **特征分布图**：展示各特征的分布情况
2. **特征相关性热图**：展示特征之间的相关性
3. **特征重要性图**：展示各特征对目标变量的重要性
4. **特征对比散点图**：展示关键特征与目标变量的关系

这些可视化图表有助于理解数据特征和模型预测的关系，提高模型的可解释性。
