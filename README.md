# Transformers Pipeline 实践项目

本项目是基于 Hugging Face Transformers 库的 pipeline 功能的实践集合，特别优化用于 MacBook Pro M2 Max 上使用 MPS 加速。

## 项目结构

```
transformers-pipeline-practice/
├── README.md                 # 项目说明文档
├── utils/                    # 工具函数
│   ├── device_utils.py       # 设备检测和配置工具
│   └── model_utils.py        # 模型下载和管理工具
├── tasks/                    # 按任务类型组织的子目录
│   ├── text_generation/      # 文本生成任务
│   │   └── text_gen.py       # 文本生成示例
│   ├── translation/          # 翻译任务
│   │   └── translator.py     # 翻译示例
│   ├── question_answering/   # 问答任务
│   │   └── qa.py             # 问答示例
│   └── conversation/         # 会话任务
│       └── chatbot.py        # 聊天机器人示例
├── examples/                 # 综合示例
│   └── pipeline_showcase.py  # 多种pipeline展示
└── data/                     # 样本数据
    └── text/                 # 文本样本
```

## 环境设置

建议创建名为 `transformers` 的虚拟环境，并安装必要的依赖：

```bash
# 创建虚拟环境
python -m venv transformers

# 激活虚拟环境
source transformers/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 问答系统

```bash
python tasks/question_answering/qa.py
```

### 聊天机器人

```bash
python tasks/conversation/chatbot.py
```

## MPS 加速支持

本项目所有脚本都支持在 MacBook M 系列芯片上自动使用 MPS 加速，提升处理速度。

## 模型管理

所有模型会自动下载并保存在本地，支持离线使用。

## 自定义模型大小

对于各种任务，可以选择不同大小的模型以平衡速度和精度。详细使用说明请参考各任务目录下的说明文档。
