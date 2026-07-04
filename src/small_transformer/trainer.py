"""
中文小说生成模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm


class NovelTrainer:
    """小说生成模型训练器"""
    
    def __init__(self, model, tokenizer, device='cpu', lr=1e-3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 损失函数（语言模型损失）
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        
        # 将模型移到设备
        self.model.to(device)
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_batches = 0
        
        start_time = time.time()
        
        # 使用tqdm显示进度
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            # 移到设备
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 前向传播
            logits = self.model(x)
            
            # 计算损失
            # logits: [batch_size, seq_len, vocab_size]
            # y: [batch_size, seq_len]
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_batches:.4f}'
            })
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        
        return avg_loss, epoch_time
    
    def generate_sample(self, prompt="", max_new_tokens=100, temperature=0.8, top_k=50):
        """生成样本文本"""
        self.model.eval()
        
        with torch.no_grad():
            # 编码prompt
            if prompt:
                idx = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
            else:
                # 随机开始
                idx = torch.zeros((1, 1), dtype=torch.long).to(self.device)
            
            # 生成
            generated = self.model.generate(
                idx, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            # 解码
            text = self.tokenizer.decode(generated[0].tolist())
            
            return text
    
    def train(self, dataloader, num_epochs=10, save_dir='models', sample_interval=5, resume=False):
        """
        训练模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        save_dir: 模型保存目录
        sample_interval: 每隔多少轮生成样本
        resume: 是否从checkpoint恢复训练
        """
        print("=" * 60)
        print("开始训练中文小说生成模型")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"模型参数: {self.model.count_parameters():,}")
        print(f"词表大小: {self.tokenizer.vocab_size}")
        print(f"训练轮数: {num_epochs}")
        print(f"Batch数: {len(dataloader)}")
        print("=" * 60)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_loss = float('inf')
        start_epoch = 1
        
        # 尝试从checkpoint恢复
        if resume:
            checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('model_epoch_')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.replace('model_epoch_', '').replace('.pt', '')))
                latest_checkpoint = checkpoint_files[-1]
                checkpoint_path = os.path.join(save_dir, latest_checkpoint)
                
                print(f"\n📂 找到checkpoint: {latest_checkpoint}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('loss', float('inf'))
                
                print(f"✅ 从Epoch {start_epoch} 继续训练")
                print(f"   上一轮损失: {checkpoint.get('loss', 'N/A')}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # 训练
            avg_loss, epoch_time = self.train_epoch(dataloader, epoch)
            
            print(f"\nEpoch {epoch} 完成:")
            print(f"平均损失: {avg_loss:.4f}")
            print(f"耗时: {epoch_time:.2f}秒")
            print(f"学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, save_path)
                print(f"✅ 保存最佳模型 (loss={avg_loss:.4f})")
            
            # 定期保存
            save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            
            # 定期生成样本
            if epoch % sample_interval == 0 or epoch == 1:
                print(f"\n生成样本 (Epoch {epoch}):")
                sample = self.generate_sample(prompt="在", max_new_tokens=100, temperature=0.8, top_k=50)
                print(f"样本: {sample}")
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"最佳损失: {best_loss:.4f}")
        print(f"模型保存在: {save_dir}")
        
        # 保存tokenizer
        tokenizer_path = os.path.join(save_dir, 'tokenizer.pt')
        self.tokenizer.save(tokenizer_path)
        print(f"词表保存在: {tokenizer_path}")
        
        return best_loss


if __name__ == '__main__':
    # 测试训练器
    print("训练器测试")