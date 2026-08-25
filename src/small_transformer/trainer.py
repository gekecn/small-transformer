"""
中文小说生成模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import random
import time
import csv
import math
from tqdm import tqdm


class NovelTrainer:
    """小说生成模型训练器"""
    
    def __init__(
        self, model, tokenizer, device='cpu', lr=1e-3, scheduler_t_max=10,
        training_config=None, warmup_steps=0, total_training_steps=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.training_config = training_config or {}
        
        # 损失函数（语言模型损失）
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        self.warmup_steps = warmup_steps
        self.scheduler_step_per_batch = warmup_steps > 0
        if self.scheduler_step_per_batch:
            if not total_training_steps or total_training_steps <= warmup_steps:
                raise ValueError("total_training_steps必须大于warmup_steps")
            min_factor = min(1.0, 1e-5 / lr)

            def lr_factor(step):
                if step < warmup_steps:
                    return max(1.0 / warmup_steps, (step + 1) / warmup_steps)
                progress = min(
                    1.0,
                    (step - warmup_steps) / (total_training_steps - warmup_steps),
                )
                return min_factor + 0.5 * (1.0 - min_factor) * (
                    1.0 + math.cos(math.pi * progress)
                )

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=scheduler_t_max, eta_min=1e-5
            )
        
        # 将模型移到设备
        self.model.to(device)

    def _tokenizer_fingerprint(self):
        return self.tokenizer.fingerprint()

    @staticmethod
    def _write_json(path, data):
        temp_path = f"{path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)

    def _checkpoint_state(
        self,
        epoch,
        train_loss,
        val_loss,
        best_loss,
        epochs_without_improvement,
        dataloader,
        history,
    ):
        generator_state = None
        if getattr(dataloader, 'generator', None) is not None:
            generator_state = dataloader.generator.get_state()
        return {
            'checkpoint_version': 3,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model.get_config(),
            'tokenizer_fingerprint': self._tokenizer_fingerprint(),
            'training_config': self.training_config,
            'history': history,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss': val_loss if val_loss is not None else train_loss,
            'best_loss': best_loss,
            'epochs_without_improvement': epochs_without_improvement,
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'dataloader_generator_state': generator_state,
        }

    @staticmethod
    def _save_checkpoint_atomic(state, path):
        temp_path = f"{path}.tmp"
        torch.save(state, temp_path)
        os.replace(temp_path, path)

    def _write_history(self, save_dir, history):
        self._write_json(os.path.join(save_dir, 'training_history.json'), history)
        csv_path = os.path.join(save_dir, 'training_history.csv')
        temp_csv = f"{csv_path}.tmp"
        fields = [
            'epoch', 'train_loss', 'val_loss', 'learning_rate',
            'epoch_seconds', 'overfitting_warning',
        ]
        with open(temp_csv, 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        os.replace(temp_csv, csv_path)
    
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
            if self.scheduler_step_per_batch:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            total_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_batches:.4f}'
            })
        
        # 更新学习率
        if not self.scheduler_step_per_batch:
            self.scheduler.step()
        
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        
        return avg_loss, epoch_time

    def evaluate(self, dataloader):
        """计算验证集平均损失，不更新模型参数。"""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                total_loss += loss.item()
                total_batches += 1
        return total_loss / max(1, total_batches)
    
    def generate_sample(
        self,
        prompt="",
        max_new_tokens=100,
        temperature=0.6,
        top_k=20,
        repetition_penalty=1.18,
        no_repeat_ngram_size=4,
    ):
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
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                forbidden_token_ids=self.tokenizer.special_token_ids,
            )
            
            # 解码
            text = self.tokenizer.decode(generated[0].tolist())
            
            return text
    
    def train(
        self,
        dataloader,
        val_dataloader=None,
        num_epochs=10,
        save_dir='models',
        sample_interval=2,
        resume_from=None,
        early_stopping_patience=3,
        validation_start_epoch=1,
        validation_interval=1,
    ):
        """
        训练模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        save_dir: 模型保存目录
        sample_interval: 每隔多少轮生成样本
        resume_from: 要恢复的checkpoint路径；None表示全新训练
        """
        if validation_start_epoch < 1:
            raise ValueError("validation_start_epoch必须大于等于1")
        if validation_interval < 1:
            raise ValueError("validation_interval必须大于等于1")
        print("=" * 60)
        print("开始训练中文小说生成模型")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"模型参数: {self.model.count_parameters():,}")
        print(f"词表大小: {self.tokenizer.vocab_size}")
        print(f"训练轮数: {num_epochs}")
        print(f"Batch数: {len(dataloader)}")
        print("=" * 60)

        # 恢复运行必须先完成只读兼容性校验，校验失败时不得改写原运行目录。
        resume_checkpoint = None
        if resume_from:
            resume_checkpoint = torch.load(
                resume_from, map_location=self.device, weights_only=False
            )
            checkpoint_config = resume_checkpoint.get('model_config')
            if checkpoint_config and checkpoint_config != self.model.get_config():
                raise ValueError("checkpoint模型配置与当前模型不一致")
            checkpoint_tokenizer = resume_checkpoint.get('tokenizer_fingerprint')
            if checkpoint_tokenizer and checkpoint_tokenizer != self._tokenizer_fingerprint():
                raise ValueError("checkpoint使用的Tokenizer与当前Tokenizer不一致")
            checkpoint_training_config = resume_checkpoint.get('training_config')
            if checkpoint_training_config is not None and checkpoint_training_config != self.training_config:
                raise ValueError(
                    "checkpoint训练配置或语料指纹与当前运行不一致，拒绝续训。"
                    "请使用原配置，或创建新的训练运行。"
                )
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        tokenizer_path = os.path.join(save_dir, 'tokenizer.pt')
        self.tokenizer.save(tokenizer_path)
        metadata_path = os.path.join(save_dir, 'run_metadata.json')
        self._write_json(metadata_path, {
            'status': 'running',
            'save_dir': os.path.abspath(save_dir),
            'model_config': self.model.get_config(),
            'tokenizer_fingerprint': self._tokenizer_fingerprint(),
            'training_config': self.training_config,
            'max_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'validation_start_epoch': validation_start_epoch,
            'validation_interval': validation_interval,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'resume_from': os.path.abspath(resume_from) if resume_from else None,
        })
        
        best_loss = float('inf')
        start_epoch = 1
        epochs_without_improvement = 0
        last_epoch = start_epoch - 1
        history = []
        
        # 尝试从checkpoint恢复
        if resume_from:
            print(f"\n恢复checkpoint: {resume_from}")
            checkpoint = resume_checkpoint
            checkpoint_config = checkpoint.get('model_config')
            if checkpoint_config and checkpoint_config != self.model.get_config():
                raise ValueError("checkpoint模型配置与当前模型不一致")
            checkpoint_tokenizer = checkpoint.get('tokenizer_fingerprint')
            if checkpoint_tokenizer and checkpoint_tokenizer != self._tokenizer_fingerprint():
                raise ValueError("checkpoint使用的Tokenizer与当前语料生成的Tokenizer不一致")
            checkpoint_training_config = checkpoint.get('training_config')
            if checkpoint_training_config is not None and checkpoint_training_config != self.training_config:
                raise ValueError(
                    "checkpoint训练配置或语料指纹与当前运行不一致，拒绝续训。"
                    "请使用原配置，或创建新的训练运行。"
                )
            if checkpoint_training_config is None:
                print("提示：旧checkpoint没有完整训练配置，只能执行兼容恢复检查。")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            history = list(checkpoint.get('history', []))
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if 'python_rng_state' in checkpoint:
                random.setstate(checkpoint['python_rng_state'])
            generator_state = checkpoint.get('dataloader_generator_state')
            if generator_state is not None and getattr(dataloader, 'generator', None) is not None:
                dataloader.generator.set_state(generator_state)

            print(f"已从Epoch {start_epoch}继续训练")
            print(f"   当前最佳损失: {best_loss:.4f}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            last_epoch = epoch
            if hasattr(dataloader.dataset, 'set_epoch'):
                dataloader.dataset.set_epoch(epoch)
                print(
                    f"本轮语料: {dataloader.dataset.active_name} "
                    f"({len(dataloader.dataset):,}个训练样本)"
                )
            epoch_learning_rate = self.optimizer.param_groups[0]['lr']
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # 训练
            avg_loss, epoch_time = self.train_epoch(dataloader, epoch)
            validation_due = (
                val_dataloader is not None
                and epoch >= validation_start_epoch
                and (
                    (epoch - validation_start_epoch) % validation_interval == 0
                    or epoch == num_epochs
                )
            )
            val_loss = self.evaluate(val_dataloader) if validation_due else None
            # 配置验证集后，最佳模型只按验证Loss选择，不能把训练Loss和验证Loss混用。
            selection_loss = (
                val_loss
                if val_dataloader is not None
                else avg_loss
            )
            
            print(f"\nEpoch {epoch} 完成:")
            print(f"平均损失: {avg_loss:.4f}")
            if val_loss is not None:
                print(f"验证损失: {val_loss:.4f}")
            elif val_dataloader is not None:
                print("验证损失: 本轮按配置跳过")
            print(f"耗时: {epoch_time:.2f}秒")
            print(f"本轮学习率: {epoch_learning_rate:.6f}")
            print(f"下一轮学习率: {self.scheduler.get_last_lr()[0]:.6f}")

            overfitting_warning = False
            previous_validation = next(
                (row for row in reversed(history) if row['val_loss'] is not None),
                None,
            )
            if previous_validation is not None and val_loss is not None:
                overfitting_warning = (
                    avg_loss < previous_validation['train_loss']
                    and val_loss > previous_validation['val_loss']
                )
                if overfitting_warning:
                    print("提示：训练Loss下降但验证Loss上升，出现可能的过拟合信号。")
            history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'learning_rate': epoch_learning_rate,
                'epoch_seconds': epoch_time,
                'overfitting_warning': overfitting_warning,
            })
            self._write_history(save_dir, history)

            improved = selection_loss is not None and selection_loss < best_loss
            if selection_loss is not None:
                if improved:
                    best_loss = selection_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            
            # 定期生成样本
            if epoch % sample_interval == 0 or epoch == 1:
                print(f"\n生成样本 (Epoch {epoch}):")
                sample = self.generate_sample(prompt="在", max_new_tokens=100)
                print(f"样本: {sample}")

            # 生成样本也会消耗随机状态，因此必须在它之后保存恢复点。
            checkpoint_state = self._checkpoint_state(
                epoch, avg_loss, val_loss, best_loss,
                epochs_without_improvement, dataloader, history,
            )
            if improved:
                self._save_checkpoint_atomic(
                    checkpoint_state,
                    os.path.join(save_dir, 'best_model.pt'),
                )
                print(f"已保存最佳模型 (selection_loss={selection_loss:.4f})")
            self._save_checkpoint_atomic(
                checkpoint_state,
                os.path.join(save_dir, f'model_epoch_{epoch}.pt'),
            )

            if (
                val_dataloader is not None
                and validation_due
                and early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"\n验证损失连续 {early_stopping_patience} 轮未改善，提前停止训练。")
                break
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"最佳损失: {best_loss:.4f}")
        print(f"模型保存在: {save_dir}")
        
        # 保存tokenizer
        tokenizer_path = os.path.join(save_dir, 'tokenizer.pt')
        self.tokenizer.save(tokenizer_path)
        print(f"词表保存在: {tokenizer_path}")

        self._write_json(metadata_path, {
            'status': 'complete',
            'save_dir': os.path.abspath(save_dir),
            'model_config': self.model.get_config(),
            'tokenizer_fingerprint': self._tokenizer_fingerprint(),
            'training_config': self.training_config,
            'history': history,
            'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'best_loss': best_loss,
            'last_epoch': last_epoch,
        })
        model_root = os.path.dirname(os.path.abspath(save_dir))
        self._write_json(os.path.join(model_root, 'latest_run.json'), {
            'run_dir': os.path.basename(os.path.abspath(save_dir)),
            'checkpoint': 'best_model.pt',
        })
        
        return best_loss


if __name__ == '__main__':
    # 测试训练器
    print("训练器测试")
