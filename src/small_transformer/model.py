"""
GPT-style 中文小说生成模型
支持因果注意力（只看之前的词）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """因果自注意力 - 只能看到之前的token"""
    
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 因果掩码 - 只能看到之前的token
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 计算 Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到V
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, embed_dim, num_heads, hidden_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm结构（GPT-2风格）
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ChineseGPT(nn.Module):
    """中文小说生成模型"""
    
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, hidden_dim=512, 
                 num_layers=4, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终LayerNorm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # 输出层（预测下一个token）
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 权重绑定（减少参数）
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx):
        """
        前向传播
        idx: [batch_size, seq_len] token索引
        返回: [batch_size, seq_len, vocab_size] logits
        """
        batch_size, seq_len = idx.size()
        
        # Token嵌入
        token_emb = self.token_embedding(idx)
        
        # 位置嵌入
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # 合并
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 最终LayerNorm
        x = self.ln_f(x)
        
        # 预测下一个token
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        生成文本
        idx: [batch_size, seq_len] 输入token
        max_new_tokens: 要生成的token数量
        temperature: 温度参数（控制随机性）
        top_k: 只从top_k个最可能的token中选择
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # 前向传播
            logits = self(idx_cond)
            
            # 只取最后一个位置的logits
            logits = logits[:, -1, :]
            
            # 温度调整
            logits = logits / temperature
            
            # Top-k过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax得到概率
            probs = F.softmax(logits, dim=-1)
            
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 添加到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def count_parameters(self):
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # 测试模型
    model = ChineseGPT(vocab_size=5000, embed_dim=256, num_heads=4, hidden_dim=512, num_layers=4)
    
    print(f"模型参数: {model.count_parameters():,}")
    
    # 测试前向传播
    idx = torch.randint(0, 5000, (2, 128))
    logits = model(idx)
    print(f"输入形状: {idx.shape}")
    print(f"输出形状: {logits.shape}")
    
    # 测试生成
    generated = model.generate(idx[:1, :10], max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"生成序列长度: {generated.shape}")