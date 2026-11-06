# 基于机器学习的文本情感分类

> **项目描述：** 基于微博文本数据，使用多种机器学习和深度学习模型进行中文情感分类任务  
> **技术栈：** Python, PyTorch, scikit-learn, Pandas, Matplotlib  
> **任务类型：** 三分类问题（积极/中立/消极）

## 📊 项目概述

本项目实现了一个完整的中文文本情感分类系统，从数据预处理到模型训练，再到性能评估，涵盖了机器学习项目的完整流程。

项目基于约 36 万条微博文本数据，实现了多种经典和现代的机器学习算法。

### 主要特点

- 🔍 **大规模数据集**：358,584 条微博文本，经过 LLM 标注的高质量情感标签
- 🤖 **多模型对比**：实现了 5 种不同的机器学习和深度学习模型
- 📈 **完整pipeline**：从数据清洗、特征工程到模型训练的完整流程
- 🎯 **性能优化**：针对数据不平衡问题采用了多种优化策略
- 📊 **详细分析**：包含丰富的数据分析和可视化结果

## 🗂️ 项目结构

```
├── 01Data Wrangling.ipynb              # 数据整理与清洗
├── 02Feature Engineering and Data Analysis.ipynb  # 特征工程与数据分析
├── 03Data Augmentation.ipynb           # 数据增强
├── 04Segmentation and Tokenization.ipynb  # 分词与标记化
├── 05Implement.ipynb                   # 传统机器学习模型实现
├── 06Pre-trained Word Embeddings.ipynb # 预训练词向量模型
├── 07Pre-trained Language Model.ipynb  # 预训练语言模型
├── 08Comparison and Analysis.ipynb     # 模型对比与分析
├── MyModule.py                         # 自定义模块和工具函数
├── README.md                           # 项目说明文档
├── requirements.txt                    # 项目依赖包
├── data/                               # 数据目录
│   ├── original/                       # 原始数据
│   ├── preparation/                    # 预处理数据
│   └── final/                          # 最终处理数据
│       └── vocab.txt                   # 词汇表
├── models/                             # 预训练模型
│   ├── chinese-macbert-base/           # MacBERT 模型
│   ├── opus-mt-en-zh/                  # 英中翻译模型
│   └── opus-mt-zh-en/                  # 中英翻译模型
├── pics/                               # 图片资源
└── results/                            # 结果输出
```

## 🚀 实现的模型

1. **多层感知机 (MLP)**
2. **卷积神经网络 (CNN)**
3. **长短期记忆网络 (LSTM)**
4. **Transformer**
5. **预训练语言模型 (MacBERT)**

## 📈 数据集概况

- **数据规模**：358,584 条微博文本
- **数据来源**：新浪微博用户原创内容
- **标注方式**：基于大语言模型(DeepSeek API)的自动标注
- **类别分布**：
  - 积极：167,778 条 (46.8%)
  - 消极：124,334 条 (34.7%)
  - 中立：66,472 条 (18.5%)
- **文本长度**：平均 18.8 字符，标准差 11.1 字符

## 🔧 环境配置

### 依赖包
```python
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
ltp>=4.1.0
```

### 安装方式
```bash
pip install -r requirements.txt
```

## 🎯 主要功能模块

### MyModule.py 核心功能
- **Vocab类**：词汇表管理
- **数据加载**：[`load_data`](MyModule.py)、[`load_vocab`](MyModule.py)
- **分词处理**：[`segment_data`](MyModule.py) 
- **模型训练**：[`train_model_with_validation`](MyModule.py)
- **模型评估**：[`test_model_comprehensive`](MyModule.py)、[`evaluate_model`](MyModule.py)
- **结果展示**：[`print_training_summary`](MyModule.py)

### 数据处理特色
- **数据增强**：原始数据分层划分后，仅对训练集中的中立情感样本进行回译增强，避免验证集和测试集泄漏
- **类别平衡**：使用类别权重缓解数据不平衡
- **分词优化**：基于 LTP 的中文分词处理
- **词汇管理**：构建 16,000+ 词汇的专用词典


## 🔬 技术亮点

1. **多维度数据分析**：从文本长度、情感分布等多个维度分析数据特征
2. **先进的数据增强**：使用回译技术平衡类别分布  
3. **模型架构优化**：自实现的 CNN、LSTM、Transformer 架构
4. **预训练模型微调**：基于 MacBERT 的中文情感分类优化
5. **综合性能评估**：包含准确率、精确率、召回率、F1分数等多项指标

## 📝 Notebook说明

- **[01Data Wrangling.ipynb](01Data%20Wrangling.ipynb)**：原始数据处理，从JSON格式提取文本和标签
- **[02Feature Engineering and Data Analysis.ipynb](02Feature%20Engineering%20and%20Data%20Analysis.ipynb)**：特征工程，数据统计分析和可视化
- **[03Data Augmentation.ipynb](03Data%20Augmentation.ipynb)**：先分层划分原始数据，再仅对训练集使用回译技术平衡标签分布
- **[04Segmentation and Tokenization.ipynb](04Segmentation_and%20Tokenization.ipynb)**：中文分词和词汇表构建
- **[05Implement.ipynb](05Implement.ipynb)**：传统机器学习模型实现和训练
- **[06Pre-trained Word Embeddings.ipynb](06Pre-trained%20Word%20Embeddings.ipynb)**：基于预训练词向量的深度学习模型
- **[07Pre-trained Language Model.ipynb](07Pre-trained%20Language%20Model.ipynb)**：预训练语言模型微调
- **[08Comparison and Analysis.ipynb](08Comparison%20and%20Analysis.ipynb)**：所有模型的对比分析和总结

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👤 作者

项目作者完成了一个完整的机器学习项目实践，从数据处理到模型实现，展现了扎实的技术功底和项目管理能力（存疑）。

---

⭐ 如果这个项目对您有帮助，请给个 Star 支持一下（实际对 $0$ 人有帮助）！
