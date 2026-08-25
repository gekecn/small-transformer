"""中文小说数据集处理，支持字符级和SentencePiece BPE tokenization。"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import random
import re
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:  # 旧字符模型仍可在未安装SentencePiece时加载
    spm = None


class CharTokenizer:
    """字符级别的tokenizer"""

    tokenizer_type = "char"
    
    def __init__(self, max_vocab_size=5000, min_frequency=2):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
    
    def build_vocab(self, texts):
        """构建词表"""
        # 按频率收集字符，避免百科中的罕见字和异常符号无限扩大模型。
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)
        
        # 添加特殊token
        special_tokens = ['<pad>', '<unk>']
        
        # 构建映射
        for i, token in enumerate(special_tokens):
            self.char2idx[token] = i
            self.idx2char[i] = token
        
        chars = [
            char for char, count in char_counts.items()
            if count >= self.min_frequency
        ]
        chars.sort(key=lambda char: (-char_counts[char], char))
        if self.max_vocab_size is not None:
            chars = chars[:max(0, self.max_vocab_size - len(special_tokens))]

        # 添加高频字符
        for i, char in enumerate(chars):
            idx = i + len(special_tokens)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        
        self.vocab_size = len(self.char2idx)
        
        print(f"词表大小: {self.vocab_size}")
        print(f"纳入高频字符: {len(chars)}")
        print(f"语料不同字符: {len(char_counts)}")
    
    def encode(self, text):
        """编码文本"""
        return [self.char2idx.get(char, self.char2idx['<unk>']) for char in text]
    
    def decode(self, indices):
        """解码索引"""
        chars = []
        for idx in indices:
            if idx in self.idx2char:
                char = self.idx2char[idx]
                if char not in ['<pad>', '<unk>']:
                    chars.append(char)
        return ''.join(chars)
    
    def save(self, path):
        """保存词表"""
        torch.save({
            'tokenizer_type': self.tokenizer_type,
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'min_frequency': self.min_frequency,
        }, path)
    
    def load(self, path):
        """加载词表"""
        data = torch.load(path, map_location='cpu', weights_only=False)
        self.char2idx = data['char2idx']
        self.idx2char = data['idx2char']
        self.vocab_size = data['vocab_size']
        self.max_vocab_size = data.get('max_vocab_size', self.vocab_size)
        self.min_frequency = data.get('min_frequency', 1)

    @property
    def special_token_ids(self):
        return [self.char2idx['<pad>'], self.char2idx['<unk>']]

    def fingerprint(self):
        payload = json.dumps(
            self.char2idx,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


class SentencePieceBPETokenizer:
    """由当前小说语料训练的小词表SentencePiece BPE Tokenizer。"""

    tokenizer_type = "sentencepiece_bpe"
    newline_token = "<NL>"

    def __init__(self, model_proto=None):
        self.processor = None
        self.model_proto = model_proto
        self.vocab_size = 0
        if model_proto is not None:
            self._load_proto(model_proto)

    @staticmethod
    def _require_sentencepiece():
        if spm is None:
            raise RuntimeError(
                "BPE Tokenizer需要sentencepiece，请先运行 pip install -r requirements.txt"
            )

    def _load_proto(self, model_proto):
        self._require_sentencepiece()
        self.model_proto = bytes(model_proto)
        self.processor = spm.SentencePieceProcessor()
        if not self.processor.LoadFromSerializedProto(self.model_proto):
            raise ValueError("无法加载SentencePiece模型")
        self.vocab_size = self.processor.get_piece_size()

    @classmethod
    def build(cls, texts, vocab_size=6000):
        """只用训练集正文学习BPE，验证集不会参与词表训练。"""
        cls._require_sentencepiece()
        if vocab_size < 1000:
            raise ValueError("BPE词表至少需要1000个token")
        with tempfile.TemporaryDirectory(prefix="small-transformer-bpe-") as temp_dir:
            temp_path = Path(temp_dir)
            corpus_path = temp_path / "corpus.txt"
            model_prefix = temp_path / "tokenizer"
            with corpus_path.open('w', encoding='utf-8', newline='\n') as stream:
                for text in texts:
                    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
                    # SentencePiece会把换行视为空白；用专用token保留小说段落和对话格式。
                    for line in normalized.split('\n'):
                        if line:
                            stream.write(line)
                            stream.write('\n')
                        stream.write(cls.newline_token)
                        stream.write('\n')
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                model_type='bpe',
                vocab_size=vocab_size,
                character_coverage=0.9995,
                hard_vocab_limit=False,
                byte_fallback=True,
                normalization_rule_name='identity',
                unk_id=0,
                pad_id=1,
                bos_id=-1,
                eos_id=-1,
                user_defined_symbols=[cls.newline_token],
                shuffle_input_sentence=False,
                max_sentence_length=16384,
                minloglevel=1,
            )
            model_proto = model_prefix.with_suffix('.model').read_bytes()
        tokenizer = cls(model_proto)
        print(f"BPE词表大小: {tokenizer.vocab_size}")
        return tokenizer

    def encode(self, text):
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        normalized = normalized.replace('\n', f' {self.newline_token} ')
        return self.processor.encode(normalized, out_type=int)

    def decode(self, indices):
        filtered = [int(index) for index in indices if int(index) != self.processor.pad_id()]
        text = self.processor.decode(filtered).replace(self.newline_token, '\n')
        return re.sub(r'[ \t]*\n[ \t]*', '\n', text)

    def save(self, path):
        torch.save({
            'tokenizer_type': self.tokenizer_type,
            'model_proto': self.model_proto,
            'vocab_size': self.vocab_size,
            'newline_token': self.newline_token,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data.get('tokenizer_type') != self.tokenizer_type:
            raise ValueError("Tokenizer文件不是SentencePiece BPE格式")
        self._load_proto(data['model_proto'])

    @property
    def special_token_ids(self):
        return [self.processor.unk_id(), self.processor.pad_id()]

    def fingerprint(self):
        # SentencePiece模型元数据会记录临时训练文件路径，不能直接对proto取哈希。
        # 对实际词片、分数和类型取哈希，同一语料重建后仍可安全恢复checkpoint。
        pieces = []
        for index in range(self.vocab_size):
            pieces.append({
                'piece': self.processor.id_to_piece(index),
                'score': round(float(self.processor.get_score(index)), 8),
                'unknown': self.processor.is_unknown(index),
                'control': self.processor.is_control(index),
                'unused': self.processor.is_unused(index),
                'byte': self.processor.is_byte(index),
            })
        payload = json.dumps(
            {'type': self.tokenizer_type, 'pieces': pieces},
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


def load_saved_tokenizer(path):
    """自动识别新BPE或旧字符Tokenizer。"""
    data = torch.load(path, map_location='cpu', weights_only=False)
    tokenizer_type = data.get('tokenizer_type', 'char')
    if tokenizer_type == SentencePieceBPETokenizer.tokenizer_type:
        tokenizer = SentencePieceBPETokenizer()
    elif tokenizer_type == CharTokenizer.tokenizer_type:
        tokenizer = CharTokenizer()
    else:
        raise ValueError(f"不支持的Tokenizer类型: {tokenizer_type}")
    tokenizer.load(path)
    return tokenizer


class ChineseNovelDataset(Dataset):
    """中文小说数据集"""
    
    def __init__(self, texts, tokenizer, max_seq_len=256):
        """
        texts: 文本列表
        tokenizer: CharTokenizer或SentencePieceBPETokenizer
        max_seq_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 处理文本
        self.encoded_texts = []
        self.chunk_locations = []
        
        for text_index, text in enumerate(texts):
            # 编码文本
            encoded = tokenizer.encode(text)
            encoded_tensor = torch.tensor(encoded, dtype=torch.int32)
            self.encoded_texts.append(encoded_tensor)
            
            # 只保存文本编号和起点，避免复制每个chunk的Python整数列表。
            for i in range(0, len(encoded) - max_seq_len, max_seq_len):
                self.chunk_locations.append((text_index, i))
        
        print(f"总chunk数: {len(self.chunk_locations)}")
    
    def __len__(self):
        return len(self.chunk_locations)
    
    def __getitem__(self, idx):
        text_index, start = self.chunk_locations[idx]
        chunk = self.encoded_texts[text_index][start:start + self.max_seq_len + 1]
        
        # 输入: chunk[:-1]
        # 目标: chunk[1:]（预测下一个token）
        x = chunk[:-1].long()
        y = chunk[1:].long()
        
        return x, y


class AlternatingCorpusDataset(Dataset):
    """把训练样本固定分组，并按Epoch轮换使用其中一组。"""

    def __init__(self, dataset, parts=2, seed=42, group_labels=None):
        if parts < 2:
            raise ValueError("交替语料至少需要分成2份")
        if len(dataset) < parts:
            raise ValueError("训练样本数少于语料分组数")
        self.dataset = dataset
        if group_labels is not None and len(group_labels) != len(dataset):
            raise ValueError("分层标签数量必须与训练样本数一致")
        labels = group_labels or [None] * len(dataset)
        grouped_indices = {}
        for index, label in enumerate(labels):
            grouped_indices.setdefault(label, []).append(index)
        self.parts = [[] for _ in range(parts)]
        rng = random.Random(seed)
        # 每个来源文件分别洗牌、轮流发到A/B，避免某一半被单一长篇占据。
        for label in sorted(grouped_indices, key=lambda value: str(value)):
            indices = grouped_indices[label]
            rng.shuffle(indices)
            start_part = min(range(parts), key=lambda index: len(self.parts[index]))
            for offset, index in enumerate(indices):
                self.parts[(start_part + offset) % parts].append(index)
        for part in self.parts:
            rng.shuffle(part)
        self.active_part = 0

    def set_epoch(self, epoch):
        self.active_part = (epoch - 1) % len(self.parts)

    @property
    def active_name(self):
        return chr(ord('A') + self.active_part)

    def __len__(self):
        return len(self.parts[self.active_part])

    def __getitem__(self, idx):
        source_index = self.parts[self.active_part][idx]
        return self.dataset[source_index]


def load_novel_texts(data_dir):
    """加载小说文本"""
    texts = []
    
    # 加载所有txt文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            print(f"加载文件: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                # utf-8-sig 文件开头可能包含 BOM；它不是正文字符。
                text = f.read().lstrip('\ufeff')
                
                # 清理文本（去除分隔线等）
                text = re.sub(r'---章节分隔线---', '', text)
                text = re.sub(r'【科幻片段 \d+】', '', text)
                
                texts.append(text)
    
    # 统计总字数
    total_chars = sum(len(t) for t in texts)
    print(f"\n总文件数: {len(texts)}")
    print(f"总字符数: {total_chars:,}")
    print(f"总字数: {total_chars/10000:.1f}万字")
    
    return texts


DOCUMENT_HEADER = re.compile(
    r"(?m)(?=^(?:【(?:新闻|百科|作品|章节)[^】]*】|序章[^\n]*|"
    r"第[0-9零〇一二三四五六七八九十百千万两]+[章节回][^\n]*))"
)


def split_training_documents(text, min_chars=257):
    """按新闻、百科或小说章节边界拆分，避免验证集泄漏。"""
    parts = [part.strip() for part in DOCUMENT_HEADER.split(text) if part.strip()]
    if not parts:
        return []

    documents = []
    pending = ""
    for part in parts:
        if pending:
            part = pending + "\n" + part
            pending = ""
        if len(part) < min_chars:
            pending = part
        else:
            documents.append(part)
    if pending:
        if documents:
            documents[-1] += "\n" + pending
        else:
            documents.append(pending)
    return documents


def load_training_document_records(data_dir, min_chars=257):
    """加载顶层txt文件，返回正文及其来源文件。"""
    records = []
    file_stats = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().lstrip('\ufeff')
        text = re.sub(r'---章节分隔线---', '', text)
        text = re.sub(r'【科幻片段 \d+】', '', text)
        file_documents = split_training_documents(text, min_chars=min_chars)
        records.extend((document, filename) for document in file_documents)
        file_stats.append((filename, len(text), len(file_documents)))

    for filename, chars, count in file_stats:
        print(f"加载文件: {filename} ({chars:,}字符, {count:,}篇文档)")
    print(f"\n总文件数: {len(file_stats)}")
    print(f"总文档数: {len(records):,}")
    print(f"总字符数: {sum(len(doc) for doc, _source in records):,}")
    return records


def load_training_documents(data_dir, min_chars=257):
    """兼容接口：只返回训练文档正文。"""
    return [document for document, _source in load_training_document_records(data_dir, min_chars)]


def create_dataloader(data_dir, batch_size=8, max_seq_len=256, shuffle=True):
    """创建数据加载器"""
    
    # 加载文本
    texts = load_novel_texts(data_dir)
    
    # 创建tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    
    # 创建数据集
    dataset = ChineseNovelDataset(texts, tokenizer, max_seq_len)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # CPU训练，不用多进程
        pin_memory=False
    )
    
    return dataloader, tokenizer


def create_train_val_dataloaders(
    data_dir,
    batch_size=8,
    max_seq_len=256,
    val_ratio=0.05,
    seed=42,
    source_weights=None,
    num_workers=0,
    tokenizer_type='sentencepiece_bpe',
    tokenizer_vocab_size=6000,
    alternating_train_parts=1,
):
    """创建可复现的数据加载器；val_ratio=0时不运行验证。"""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio 必须大于等于0且小于1")
    records = load_training_document_records(data_dir, min_chars=max_seq_len + 1)
    documents = [document for document, _source in records]
    sources = [source for _document, source in records]
    if not documents:
        raise ValueError("没有可用于训练的文档")

    if val_ratio == 0:
        train_documents = documents
        train_sources = sources
        val_documents = []
    else:
        if len(documents) < 2:
            raise ValueError("至少需要两篇独立文档才能划分训练集和验证集")
        indices = list(range(len(documents)))
        random.Random(seed).shuffle(indices)
        target_val_chars = sum(len(doc) for doc in documents) * val_ratio
        val_indices = []
        val_chars = 0
        # 按完整文档切分，同时尽量贴近目标字符数。
        for index in indices:
            document_chars = len(documents[index])
            if val_chars + document_chars <= target_val_chars:
                val_indices.append(index)
                val_chars += document_chars
        val_index_set = set(val_indices)
        remaining = [index for index in indices if index not in val_index_set]
        if not val_indices:
            selected = min(indices, key=lambda index: len(documents[index]))
            val_indices.append(selected)
            val_chars = len(documents[selected])
        elif remaining:
            selected = min(
                remaining,
                key=lambda index: abs(val_chars + len(documents[index]) - target_val_chars),
            )
            if abs(val_chars + len(documents[selected]) - target_val_chars) < abs(
                val_chars - target_val_chars
            ):
                val_indices.append(selected)
                val_chars += len(documents[selected])
        val_index_set = set(val_indices)
        train_documents = [doc for i, doc in enumerate(documents) if i not in val_index_set]
        train_sources = [source for i, source in enumerate(sources) if i not in val_index_set]
        val_documents = [documents[i] for i in val_indices]
    if not train_documents:
        raise ValueError("验证集划分占用了全部文档")

    # 启用验证时，词表只根据训练文档建立；关闭时使用全部正文。
    if tokenizer_type == 'sentencepiece_bpe':
        tokenizer = SentencePieceBPETokenizer.build(
            train_documents, vocab_size=tokenizer_vocab_size,
        )
    elif tokenizer_type == 'char':
        tokenizer = CharTokenizer(max_vocab_size=tokenizer_vocab_size)
        tokenizer.build_vocab(train_documents)
    else:
        raise ValueError(f"不支持的Tokenizer类型: {tokenizer_type}")
    train_dataset = ChineseNovelDataset(train_documents, tokenizer, max_seq_len)
    full_train_sample_count = len(train_dataset)
    if alternating_train_parts > 1:
        if source_weights:
            raise ValueError("交替语料模式暂不与source_weights同时使用")
        chunk_sources = [
            train_sources[text_index]
            for text_index, _start in train_dataset.chunk_locations
        ]
        train_dataset = AlternatingCorpusDataset(
            train_dataset,
            parts=alternating_train_parts,
            seed=seed,
            group_labels=chunk_sources,
        )
    val_dataset = (
        ChineseNovelDataset(val_documents, tokenizer, max_seq_len)
        if val_documents else None
    )
    train_generator = torch.Generator().manual_seed(seed)
    sampler = None
    if source_weights:
        chunk_weights = [
            float(source_weights.get(train_sources[text_index], 1.0))
            for text_index, _start in train_dataset.chunk_locations
        ]
        sampler = WeightedRandomSampler(
            chunk_weights,
            num_samples=len(chunk_weights),
            replacement=True,
            generator=train_generator,
        )
        print("训练采样权重:")
        for source in sorted(set(train_sources)):
            print(f"  {source}: {float(source_weights.get(source, 1.0)):.2f}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        generator=train_generator,
        persistent_workers=num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
    print(f"训练文档: {len(train_documents):,}，全部训练样本: {full_train_sample_count:,}")
    if alternating_train_parts > 1:
        part_sizes = ", ".join(
            f"{chr(ord('A') + index)}={len(part):,}"
            for index, part in enumerate(train_dataset.parts)
        )
        print(
            f"交替语料: 分成{alternating_train_parts}份（{part_sizes}），"
            "每个Epoch只训练一份"
        )
    if val_dataset is None:
        print("验证已关闭：全部文档用于训练")
    else:
        print(f"验证文档: {len(val_documents):,}，验证样本: {len(val_dataset):,}")
        print(f"验证字符比例: {sum(len(doc) for doc in val_documents) / sum(len(doc) for doc in documents):.2%}")
    return train_loader, val_loader, tokenizer


if __name__ == '__main__':
    # 测试数据加载
    data_dir = 'data'
    
    dataloader, tokenizer = create_dataloader(data_dir, batch_size=4, max_seq_len=128)
    
    print(f"\nDataLoader batch数: {len(dataloader)}")
    
    # 测试一个batch
    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"输入形状: {x.shape}")
        print(f"目标形状: {y.shape}")
        
        # 解码第一个样本
        decoded_x = tokenizer.decode(x[0].tolist())
        decoded_y = tokenizer.decode(y[0].tolist())
        
        print(f"\n输入示例: {decoded_x[:50]}...")
        print(f"目标示例: {decoded_y[:50]}...")
        
        if batch_idx >= 2:
            break
