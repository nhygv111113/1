# 心脏病分类预测项目

基于PyCharm的机器学习项目，使用多种算法预测心脏病风险分类。

## 📋 项目概述

本项目使用UCI心脏病数据集，通过数据预处理、特征工程和多种机器学习算法来预测患者是否患有心脏病。

### 主要特性
- ✅ 完整的数据处理流程
- ✅ 多种算法对比（逻辑回归、随机森林、SVM、XGBoost）
- ✅ 详细的模型评估和可视化
- ✅ 交叉验证和超参数优化
- ✅ Jupyter Notebook探索式分析
- ✅ PyCharm项目配置
- ✅ Flask Web界面（可选）

## 📁 项目结构

```
heart-disease-classification/
├── data/
│   ├── raw/                    # 原始数据
│   │   └── heart.csv          # UCI心脏病数据集
│   └── processed/             # 处理后的数据
├── src/
│   ├── __init__.py
│   ├── utils.py               # 工具函数
│   ├── data_loader.py         # 数据加载模块
│   ├── preprocessing.py       # 数据预处理模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── model.py               # 模型定义和训练
│   ├── evaluate.py            # 模型评估模块
│   └── visualize.py           # 可视化模块
├── notebooks/
│   ├── 01_data_exploration.ipynb      # 数据探索
│   ├── 02_preprocessing.ipynb         # 数据预处理
│   └── 03_model_training.ipynb        # 模型训练
├── models/                    # 保存的模型
├── plots/                     # 生成的图表
├── reports/                   # 分析报告
├── requirements.txt           # 依赖库列表
├── config.yaml               # 项目配置
├── main.py                   # 主程序入口
├── train.py                  # 训练脚本
├── .gitignore               # Git忽略文件
└── README.md                # 项目说明
```

## 🚀 快速开始

### 步骤 1: 安装环境

```bash
# 克隆项目
git clone https://github.com/nhygv111113/1.git
cd 1

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 运行项目

#### 方式1：运行完整的训练管道
```bash
python main.py
```

#### 方式2：只运行训练
```bash
python train.py
```

#### 方式3：Jupyter Notebook 探索
```bash
jupyter notebook
```

### 步骤 3: PyCharm 配置

1. **打开项目**
   - File → Open → 选择项目文件夹

2. **配置 Python 解释器**
   - File → Settings → Project → Python Interpreter
   - 选择虚拟环境中的 Python：`venv/bin/python`

3. **标记源代码目录**
   - 右键 `src/` 文件夹 → Mark Directory as → Sources Root

4. **配置运行**
   - Run → Edit Configurations
   - 创建 Python 配置，脚本路径为 `main.py`

5. **启用 Jupyter**
   - Settings → Plugins → 搜索 "Jupyter"
   - 确保已安装

## 📊 数据集说明

UCI心脏病数据集包含以下13个特征和1个目标变量：

| 特征代码 | 特征名 | 说明 |
|---------|--------|------|
| age | 年龄 | 患者年龄（岁） |
| sex | 性别 | 1=男性, 0=女性 |
| cp | 胸痛类型 | 0-3: 不同类型的胸痛 |
| trestbps | 静息血压 | 毫米汞柱 |
| chol | 血清胆固醇 | mg/dl |
| fbs | 空腹血糖 | 1: >120 mg/dl, 0: ≤120 mg/dl |
| restecg | 静息心电图 | 0-2: 异常程度 |
| thalach | 最大心率 | 最大达到的心率 |
| exang | 运动诱导心绞痛 | 1=有, 0=无 |
| oldpeak | ST压低 | 相对于静息的ST段压低 |
| slope | ST段斜率 | 1=上升, 2=平坦, 3=下降 |
| ca | 主要血管数 | 0-3: 荧光透视显示 |
| thal | 血红蛋白异常 | 0=正常, 1=固定缺陷, 2=可逆缺陷 |
| **target** | **心脏病** | **0=无, 1=有** |

## 🤖 算法对比

### 逻辑回归 (Logistic Regression)
- **优点**: 速度快、可解释性强
- **缺点**: 线性假设可能不适合复杂关系
- **使用场景**: 二分类，需要快速结果

### 随机森林 (Random Forest)
- **优点**: 处理非线性、特征重要性强
- **缺点**: 容易过拟合、计算量大
- **使用场景**: 一般分类任务

### 支持向量机 (SVM)
- **优点**: 高维数据表现好、泛化能力强
- **缺点**: 参数调优复杂、训练时间长
- **使用场景**: 小数据集、高维特征

### XGBoost
- **优点**: 性能最优、梯度提升、快速
- **缺点**: 参数众多、调优复杂
- **使用场景**: 追求最高性能

## 📈 评估指标

项目使用以下指标评估模型性能：

| 指标 | 说明 | 公式 |
|------|------|------|
| 准确率 (Accuracy) | 分类正确的比例 | (TP+TN)/(TP+TN+FP+FN) |
| 精确率 (Precision) | 预测为正的准确度 | TP/(TP+FP) |
| 召回率 (Recall) | 正样本被找到的比例 | TP/(TP+FN) |
| F1分数 | 精确率和召回率的调和平均 | 2×(P×R)/(P+R) |
| ROC-AUC | ROC曲线下的面积 | 0-1 (越大越好) |

其中：
- TP (True Positive) = 真正例
- TN (True Negative) = 真负例
- FP (False Positive) = 假正例
- FN (False Negative) = 假负例

## 🔧 配置文件说明

编辑 `config.yaml` 自定义项目参数：

```yaml
# 修改测试集大小
data:
  test_size: 0.3  # 改为30%

# 修改模型超参数
models:
  random_forest:
    hyperparameters:
      n_estimators: 200  # 增加树的数量

# 修改交叉验证折数
cross_validation:
  n_splits: 10  # 改为10折
```

## 📂 使用示例

### 基础用法

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_models
from src.evaluate import evaluate_models

# 1. 加载数据
X, y = load_data('data/raw/heart.csv')

# 2. 数据预处理
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# 3. 训练模型
results = train_models(X_train, X_test, y_train, y_test)

# 4. 评估模型
metrics = evaluate_models(results, y_test)
print(metrics)
```

### 加载已训练的模型

```python
from src.utils import load_model

# 加载模型
model = load_model('models/random_forest.pkl')

# 进行预测
y_pred = model.predict(X_test)
```

## 📊 输出结果

运行完成后会生成以下文件：

```
outputs/
├── plots/
│   ├── accuracy_comparison.png       # 准确率对比
│   ├── metrics_comparison.png        # 指标对比
│   ├── roc_curves.png                # ROC曲线
│   ├── confusion_matrices.png        # 混淆矩阵
│   └── feature_importance.png        # 特征重要性
├── models/
│   ├── logistic_regression.pkl       # 逻辑回归模型
│   ├── random_forest.pkl             # 随机森林模型
│   ├── svm.pkl                       # SVM模型
│   └── xgboost.pkl                   # XGBoost模型
└── reports/
    └── evaluation_report.txt         # 评估报告
```

## 🐛 常见问题

### Q: 如何下载数据集？

**A**: 数据集会在首次运行时自动从 UCI Machine Learning Repository 下载。如果下载失败，可以手动下载：
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
mv processed.cleveland.data data/raw/heart.csv
```

### Q: 如何修改模型参数？

**A**: 编辑 `config.yaml` 文件，修改相应的超参数：
```yaml
models:
  random_forest:
    hyperparameters:
      n_estimators: 200  # 增加树的数量
      max_depth: 15      # 增加树的深度
```

### Q: 如何查看特征重要性？

**A**: 运行完成后，`plots/feature_importance.png` 会显示特征重要性排排名。

### Q: 如何保存模型进行部署？

**A**: 模型会自动保存到 `models/` 目录，可以通过 `joblib` 加载使用：
```python
import joblib
model = joblib.load('models/random_forest.pkl')
```

## 📚 学习资源

- [Scikit-learn 官方文档](https://scikit-learn.org/)
- [XGBoost 官方文档](https://xgboost.readthedocs.io/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [机器学习基础知识](https://ml-cheatsheet.readthedocs.io/)

## 📝 更新日志

### v1.0.0 (2026-06-10)
- ✅ 初始版本发布
- ✅ 支持4种算法
- ✅ 完整的评估和可视化
- ✅ Jupyter Notebook 示例

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 👨‍💻 作者

Created by nhygv111113  
Date: 2026-06-10

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献步骤
1. Fork 此项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📞 联系方式

如有问题，请提交 GitHub Issue 或 Email: 3319606590@qq.com