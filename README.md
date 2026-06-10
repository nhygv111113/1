# 心脏病分类预测项目

基于PyCharm的机器学习项目，使用多种算法预测心脏病风险分类。

## 项目概述

本项目使用UCI心脏病数据集，通过数据预处理、特征工程和多种机器学习算法来预测患者是否患有心脏病。

### 主要特性
- ✅ 完整的数据处理流程
- ✅ 多种算法对比（逻辑回归、随机森林、SVM、XGBoost）
- ✅ 详细的模型评估和可视化
- ✅ 交叉验证和超参数优化
- ✅ Jupyter Notebook探索式分析
- ✅ PyCharm项目配置

## 项目结构

```
heart-disease-classification/
├── data/
│   ├── raw/                 # 原始数据
│   │   └── heart.csv        # UCI心脏病数据集
│   └── processed/           # 处理后的数据
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载模块
│   ├── preprocessing.py     # 数据预处理模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── model.py             # 模型定义和训练
│   ├── evaluate.py          # 模型评估模块
│   ├── utils.py             # 工具函数
│   └── visualize.py         # 可视化模块
├── notebooks/
│   ├── 01_data_exploration.ipynb      # 数据探索
│   ├── 02_preprocessing.ipynb         # 数据预处理
│   └── 03_model_training.ipynb        # 模型训练和评估
├── models/                  # 保存的模型
├── plots/                   # 生成的图表
├── requirements.txt         # 依赖库列表
├── config.yaml             # 项目配置
├── main.py                 # 主程序入口
├── train.py                # 训练脚本
└── .gitignore              # Git忽略文件
```

## 环境配置

### 要求
- Python 3.8+
- PyCharm Community 或 Professional

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/nhygv111113/1.git
cd 1
```

2. **创建虚拟环境**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 使用方法

### 方式1：运行主程序
```bash
python main.py
```

### 方式2：运行训练脚本
```bash
python train.py
```

### 方式3：在Jupyter中探索
```bash
jupyter notebook
```
然后打开 `notebooks/` 目录下的文件

## 数据集说明

UCI心脏病数据集包含以下特征：

| 特征 | 说明 |
|------|------|
| age | 年龄 |
| sex | 性别 (1=男, 0=女) |
| cp | 胸痛类型 (0-3) |
| trestbps | 静息血压 |
| chol | 血清胆固醇 |
| fbs | 空腹血糖 > 120 mg/dl (1=是, 0=否) |
| restecg | 静息心电图 |
| thalach | 最大心率 |
| exang | 运动诱导心绞痛 |
| oldpeak | ST压低 |
| slope | ST段斜率 |
| ca | 主要血管数 |
| thal | 血红蛋白异常 |
| **target** | **心脏病 (1=有, 0=无)** |

## 模型对比

项目使用以下算法：

| 算法 | 优点 | 缺点 |
|------|------|------|
| 逻辑回归 | 快速、可解释性强 | 线性假设 |
| 随机森林 | 处理非线性、特征重要性 | 容易过拟合 |
| SVM | 高维数据表现好 | 参数调优复杂 |
| XGBoost | 性能最优、梯度提升 | 计算资源多 |

## 主要流程

### 1. 数据加载
```python
from src.data_loader import load_data
X, y = load_data('data/raw/heart.csv')
```

### 2. 数据预处理
```python
from src.preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data(X, y)
```

### 3. 模型训练
```python
from src.model import train_models
results = train_models(X_train, X_test, y_train, y_test)
```

### 4. 模型评估
```python
from src.evaluate import evaluate_models
report = evaluate_models(results)
```

## 性能指标

- **准确率 (Accuracy)**: 分类正确的比例
- **精确率 (Precision)**: 预测为正的样本中实际为正的比例
- **召回率 (Recall)**: 实际为正的样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均数
- **ROC-AUC**: 接收者操作特性曲线下的面积

## PyCharm配置

### 推荐设置

1. **Python解释器**
   - File → Settings → Project → Python Interpreter
   - 选择 `venv/bin/python` 作为解释器

2. **运行配置**
   - Run → Edit Configurations
   - 创建 Python 配置，脚本路径为 `main.py`

3. **Jupyter支持**
   - Settings → Plugins → 搜索 "Jupyter"
   - 确保已安装 Jupyter 插件

## 输出结果

运行完成后会生成：
- 模型准确率对比图表
- ROC曲线
- 混淆矩阵
- 特征重要性分析
- 详细的性能报告

## 常见问题

**Q: 如何下载数据集？**
A: 数据集会在首次运行时自动从UCI Machine Learning Repository下载。

**Q: 如何修改模型参数？**
A: 修改 `config.yaml` 文件中的参数配置。

**Q: 支持GPU加速吗？**
A: XGBoost和其他库支持GPU，需要额外配置。

## 许可证

MIT License

## 作者

Created on 2026-06-10

## 贡献

欢迎提交Issue和Pull Request！

