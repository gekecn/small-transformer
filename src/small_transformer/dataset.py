"""
中文小说数据集处理
使用字符级别的tokenization
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import re


class CharTokenizer:
    """字符级别的tokenizer"""
    
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
    
    def build_vocab(self, texts):
        """构建词表"""
        # 收集所有字符
        chars = set()
        for text in texts:
            chars.update(text)
        
        # 添加特殊token
        special_tokens = ['<pad>', '<unk>']
        
        # 构建映射
        for i, token in enumerate(special_tokens):
            self.char2idx[token] = i
            self.idx2char[i] = token
        
        # 添加字符
        for i, char in enumerate(sorted(chars)):
            idx = i + len(special_tokens)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        
        self.vocab_size = len(self.char2idx)
        
        print(f"词表大小: {self.vocab_size}")
        print(f"包含字符: {len(chars)}")
    
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
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'vocab_size': self.vocab_size
        }, path)
    
    def load(self, path):
        """加载词表"""
        data = torch.load(path)
        self.char2idx = data['char2idx']
        self.idx2char = data['idx2char']
        self.vocab_size = data['vocab_size']


class ChineseNovelDataset(Dataset):
    """中文小说数据集"""
    
    def __init__(self, texts, tokenizer, max_seq_len=256):
        """
        texts: 文本列表
        tokenizer: CharTokenizer
        max_seq_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 处理文本
        self.chunks = []
        
        for text in texts:
            # 编码文本
            encoded = tokenizer.encode(text)
            
            # 分割成chunks
            # 每个chunk包含 max_seq_len+1 个token（用于预测下一个）
            for i in range(0, len(encoded) - max_seq_len, max_seq_len):
                chunk = encoded[i:i + max_seq_len + 1]
                self.chunks.append(chunk)
        
        print(f"总chunk数: {len(self.chunks)}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # 输入: chunk[:-1]
        # 目标: chunk[1:]（预测下一个token）
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


def load_novel_texts(data_dir):
    """加载小说文本"""
    texts = []
    
    # 加载所有txt文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            print(f"加载文件: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
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