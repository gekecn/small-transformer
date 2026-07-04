# 中文小说生成模型

基于 Transformer 的中文小说生成模型，使用 PyTorch 实现，支持 CPU 训练。

## 📁 项目结构

```
small-transformer/
├── data/                  # 训练数据
│   ├── large_scifi.txt    # 科幻小说语料
│   └── 穿进赛博游戏后干掉BOSS成功上位.txt
├── src/small_transformer/ # 源代码
│   ├── model.py           # Transformer 模型
│   ├── dataset.py         # 数据集处理
│   ├── trainer.py         # 训练器
│   └── __init__.py
├── models/                # 模型保存目录
├── main.py                # 训练入口
├── generate.py            # 生成脚本
├── requirements.txt       # 依赖列表
└── setup.py               # 包配置
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd small-transformer
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行训练

```bash
python main.py
```

### 4. 生成文本

训练完成后，使用 `generate.py` 生成文本：

```bash
python generate.py
```

## ⚙️ 配置参数

在 `main.py` 中可以调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 8 | 批大小 |
| `max_seq_len` | 256 | 最大序列长度 |
| `embed_dim` | 256 | 嵌入维度 |
| `num_heads` | 4 | 注意力头数 |
| `hidden_dim` | 512 | 隐藏层维度 |
| `num_layers` | 4 | Transformer 层数 |
| `num_epochs` | 50 | 训练轮数 |
| `lr` | 1e-3 | 学习率 |
| `device` | 'cpu' | 训练设备 |

## 📊 训练数据

项目包含约 279 万字的中文科幻小说训练数据：

- `large_scifi.txt`：精选科幻小说片段
- `穿进赛博游戏后干掉BOSS成功上位.txt`：赛博朋克风格小说

## 🎯 功能特性

- ✅ 字符级 Transformer 模型
- ✅ CPU 训练支持
- ✅ 自动构建词表
- ✅ 实时训练进度显示
- ✅ 定期生成样本
- ✅ 模型检查点保存

## 📝 技术栈

- Python 3.8+
- PyTorch 2.0+
- tqdm（进度条）

## 📄 许可证

MIT License
