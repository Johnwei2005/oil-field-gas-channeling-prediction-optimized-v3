# VSCode设置与使用指南

本文档提供了在VSCode中设置和使用"基于物理约束残差建模的油田气窜预测系统"的详细指南。

## 1. 安装VSCode

如果您尚未安装VSCode，请从[官方网站](https://code.visualstudio.com/)下载并安装。

## 2. 安装必要的VSCode扩展

为了获得最佳体验，建议安装以下VSCode扩展：

1. **Python**：提供Python语言支持
2. **Pylint**：Python代码静态检查工具
3. **autopep8**：Python代码格式化工具
4. **Jupyter**：支持Jupyter笔记本
5. **Git Graph**：可视化Git历史记录
6. **Chinese (Simplified) Language Pack**：中文语言包（如需要）

安装方法：
- 点击VSCode左侧的扩展图标
- 在搜索框中输入扩展名称
- 点击"安装"按钮

## 3. 克隆项目仓库

### 方法一：使用VSCode界面

1. 打开VSCode
2. 按下`Ctrl+Shift+P`（Windows/Linux）或`Cmd+Shift+P`（Mac）打开命令面板
3. 输入"Git: Clone"并选择
4. 输入仓库URL：`https://github.com/Johnwei2005/oil-field-gas-channeling-prediction-optimized-v3.git`
5. 选择保存位置
6. 等待克隆完成后，点击"打开"按钮

### 方法二：使用命令行

```bash
# 切换到您想保存项目的目录
cd /path/to/your/workspace

# 克隆仓库
git clone https://github.com/Johnwei2005/oil-field-gas-channeling-prediction-optimized-v3.git

# 打开VSCode并加载项目
code oil-field-gas-channeling-prediction-optimized-v3
```

## 4. 配置Python环境

### 创建虚拟环境

```bash
# 在项目根目录下创建虚拟环境
cd oil-field-gas-channeling-prediction-optimized-v3
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 安装依赖包

```bash
# 确保虚拟环境已激活
pip install -r requirements.txt
```

### 在VSCode中选择Python解释器

1. 按下`Ctrl+Shift+P`（Windows/Linux）或`Cmd+Shift+P`（Mac）打开命令面板
2. 输入"Python: Select Interpreter"并选择
3. 选择刚刚创建的虚拟环境（通常路径包含"venv"）

## 5. 使用项目

### 项目结构导航

VSCode左侧的资源管理器面板显示项目的文件结构。主要目录和文件包括：

- `.vscode/`：VSCode配置文件
- `data/`：数据目录
- `docs/`：文档目录
- `models/`：模型目录
- `notebooks/`：Jupyter笔记本目录
- `results/`：结果目录
- `*.py`：Python源代码文件
- `README.md`：项目说明文件

### 运行代码

项目已配置好VSCode的调试设置，您可以通过以下方式运行代码：

1. **使用调试配置**：
   - 点击左侧的"运行和调试"图标
   - 从下拉菜单中选择一个配置（如"Python: 主程序"）
   - 点击绿色的运行按钮或按下F5

2. **使用集成终端**：
   - 按下`` Ctrl+` ``打开集成终端
   - 确保虚拟环境已激活
   - 运行命令，如：`python main.py --train --model-type gaussian_process`

### 调试代码

VSCode提供了强大的调试功能：

1. **设置断点**：点击代码行号左侧设置断点
2. **启动调试**：选择调试配置并按F5
3. **调试控制**：使用调试工具栏控制执行（继续、单步执行、步入、步出等）
4. **查看变量**：在调试面板中查看变量值
5. **调试控制台**：在调试控制台中执行命令

### 使用Git版本控制

VSCode集成了Git功能，您可以：

1. **查看更改**：点击左侧的源代码管理图标查看更改
2. **提交更改**：输入提交信息并点击"提交"按钮
3. **推送更改**：点击状态栏中的同步按钮
4. **查看历史**：使用Git Graph扩展查看提交历史

## 6. 常见任务

### 数据处理

```bash
# 在集成终端中运行
python data_processor.py
```

或者使用VSCode调试配置"Python: 数据处理"。

### 模型训练

```bash
# 在集成终端中运行
python main.py --train --model-type gaussian_process
```

或者使用VSCode调试配置"Python: 主程序"。

### 模型预测

```bash
# 在集成终端中运行
python main.py --predict --input-file data/raw/new_data.csv --output-file results/predictions.csv
```

### 模型微调

```bash
# 在集成终端中运行
python model_fine_tune.py
```

### 模型评估

```bash
# 在集成终端中运行
python model_test_evaluation.py
```

## 7. 自定义VSCode设置

项目已包含基本的VSCode设置，但您可以根据个人偏好进行自定义：

1. 打开`.vscode/settings.json`文件
2. 修改设置，如字体大小、主题、自动保存等
3. 保存文件，设置将立即生效

## 8. 故障排除

### 问题：找不到模块

确保您已激活虚拟环境并安装了所有依赖包：

```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 问题：中文编码错误

如果遇到中文编码问题，请确保：

1. VSCode的文件编码设置为UTF-8
2. 数据文件的编码已正确处理（项目已包含自动编码检测功能）

### 问题：调试配置不起作用

检查`.vscode/launch.json`文件是否正确配置，确保Python解释器路径正确。

## 9. 获取帮助

如果您遇到任何问题，可以：

1. 查阅项目文档（`docs/`目录）
2. 查看VSCode官方文档：https://code.visualstudio.com/docs
3. 在GitHub仓库中提交Issue
