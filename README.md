# 中文小说生成模型

基于 Transformer 的中文小说生成模型，使用 PyTorch 实现，支持 CPU 训练。

## 📁 项目结构

```
small-transformer/
├── data_scifi/            # 已准备好的科幻小说训练语料
├── src/small_transformer/ # 源代码
│   ├── model.py           # Transformer 模型
│   ├── dataset.py         # 数据集处理
│   ├── model_io.py        # 模型、词表与checkpoint统一加载
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

默认从随机初始化开始，一次连续训练16轮：训练样本按来源均衡分成A/B，
奇数轮使用A、偶数轮使用B，相当于完整覆盖全部语料8遍。前50个Batch
执行线性预热，学习率升到`3e-4`后，再按本次16轮的全部训练步数进行余弦衰减。

每次训练都会创建独立目录，例如`models/run-20260823-120000-000000/`，
其中包含模型、Tokenizer、运行状态和完整恢复信息，不会与旧模型混放。
每轮还会更新：

- `training_history.json`：结构化训练记录；
- `training_history.csv`：便于Excel查看训练Loss、验证Loss、学习率和每轮耗时。

checkpoint会保存语料文件SHA-256、Batch、序列长度、数据拆分、采样权重、
学习率和CPU配置。续训时任一关键项不一致都会被拒绝，避免在变化后的语料上误续训。

中断后从该运行目录的最新checkpoint继续：

```bash
python main.py --resume-run models/run-20260823-120000-000000
```

### 4. 生成文本

训练完成后，使用 `generate.py` 生成文本：

```bash
python generate.py
```

默认使用最近一次完成训练的`best_model.pt`。也可以明确选择：

```bash
python generate.py --run-dir models/run-20260823-120000-000000
python generate.py --checkpoint models/run-20260823-120000-000000/model_epoch_3.pt
python generate.py --prompt "火星基地的警报突然响起"
```

默认生成参数偏向稳定展示：`temperature=0.6`、`top_k=20`、轻度重复惩罚
`1.05`，并禁止完全重复的4字符片段；这些参数均可通过命令行覆盖。

新训练默认用当前小说语料现场学习一个6,000词的SentencePiece BPE Tokenizer，
常用词组可以合并成一个token，并用`<NL>`保留小说换行。旧训练运行中的字符级
Tokenizer仍可自动识别和加载。若要做字符级教学对照，可使用：

```bash
python main.py --tokenizer char --tokenizer-vocab-size 5000
```

## ⚙️ 配置参数

在 `main.py` 中可以调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 批大小；本机4线程短测吞吐量高于32 |
| `num_workers` | 0 | 保证checkpoint精确续训；语料已在内存，增加Worker收益可忽略 |
| `cpu_threads` | 4 | PyTorch矩阵计算线程数；按本机实测设置 |
| `tokenizer_type` | `sentencepiece_bpe` | 新训练默认使用本语料训练的BPE；旧字符模型保持兼容 |
| `tokenizer_vocab_size` | 6000 | BPE词表大小 |
| `max_seq_len` | 256 | 最大序列长度 |
| `embed_dim` | 256 | 嵌入维度 |
| `num_heads` | 4 | 注意力头数 |
| `hidden_dim` | 512 | 隐藏层维度 |
| `num_layers` | 4 | Transformer 层数 |
| `num_epochs` | 16 | A/B语料交替训练16轮，相当于完整覆盖全部语料8遍 |
| `lr` | 3e-4 | 线性预热结束时达到的目标学习率 |
| `warmup_steps` | 50 | 前50个Batch线性预热；设为0可关闭 |
| `device` | 'cpu' | 训练设备 |
| `val_ratio` | 0.02 | 留出2%验证集；前3轮跳过，此后每2轮验证 |

## 📊 训练数据

仓库内包含程序运行所需的全部训练语料：约1,206万字符，共2,239个训练文本（以章节为主）：

- `cc0_space_grimoire_zh_hans.txt`：约166万字符，276篇文档；
- `local_demo_scifi_mixed_zh_hans.txt`：约459万字符，529篇文档；
- `original_scifi_selected_zh_hans.txt`：3篇人工筛选并扩写的原创短篇；
- `scifi_fiction_zh_hans.txt`：《新石头记》的未来文明篇与《月界旅行》，约9.6万字符、28篇文档；
- `scifi_magazine_multi_author_zh_hans.txt`：约571万字符，1,403篇多作者科幻文本。

默认训练只读取`data_scifi/`，因此不会混入新闻、百科、问答或通用网页文本。
原始语料、历史通用语料及下载清洗代码已移至相邻的
`../small-transformer-tools/`，不再放在教学项目内。

BPE词表大小为6,000，模型参数约371万；罕见词可以继续拆分为更小的Token。
重复度高的模板语料及来源授权不明的旧小说已从项目中删除。
训练目录内全部是科幻叙事文本，不再使用针对百科/新闻的采样补偿权重；
训练集和验证集仍按独立章节/文档划分。
语料处理记录、来源文件和审计报告见
`../small-transformer-tools/README.md`及其`data_sources/`目录。

## 🎯 功能特性

- ✅ 默认SentencePiece BPE，并兼容旧字符级模型
- ✅ CPU 训练支持
- ✅ 自动构建词表
- ✅ 实时训练进度显示
- ✅ 定期生成样本
- ✅ 模型检查点保存
- ✅ 98%/2%训练验证拆分
- ✅ 延后验证、按验证损失保存最佳模型并提前停止

## 🧰 辅助工具

语料下载与清洗、质量审计、性能基准、固定提示词对比、模型评估和
checkpoint检查统一放在相邻目录`../small-transformer-tools/`。核心项目的
依赖中不再包含`mwparserfromhell`、OpenCC和`matplotlib`。

## 📝 技术栈

- Python 3.8+
- PyTorch 2.0+
- tqdm（进度条）

## 📄 许可证

MIT License
