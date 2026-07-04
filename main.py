"""
中文小说生成模型训练主程序
"""

import torch
import os

from src.small_transformer.model import ChineseGPT
from src.small_transformer.dataset import create_dataloader
from src.small_transformer.trainer import NovelTrainer


def main():
    print("=" * 60)
    print("中文小说生成模型")
    print("=" * 60)
    
    # 配置参数
    config = {
        # 数据参数
        'data_dir': 'data',
        'batch_size': 8,
        'max_seq_len': 256,
        
        # 模型参数
        'embed_dim': 256,       # 嵌入维度
        'num_heads': 4,         # 注意力头数
        'hidden_dim': 512,      # 隐藏层维度
        'num_layers': 4,        # Transformer层数
        'dropout': 0.1,         # Dropout率
        
        # 训练参数
        'num_epochs': 50,       # 训练轮数
        'lr': 1e-3,             # 学习率
        'device': 'cpu',        # 设备（CPU训练）
        'save_dir': 'models',   # 保存目录
        
        # 生成参数
        'sample_interval': 5,   # 每隔多少轮生成样本
    }
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 加载数据
    print("\n" + "=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    dataloader, tokenizer = create_dataloader(
        config['data_dir'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len']
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"\n词表大小: {vocab_size}")
    print(f"Batch数: {len(dataloader)}")
    
    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型...")
    print("=" * 60)
    
    model = ChineseGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    num_params = model.count_parameters()
    print(f"\n模型参数: {num_params:,}")
    print(f"模型结构:")
    print(f"  - 嵌入维度: {config['embed_dim']}")
    print(f"  - 注意力头数: {config['num_heads']}")
    print(f"  - 隐藏层维度: {config['hidden_dim']}")
    print(f"  - Transformer层数: {config['num_layers']}")
    
    # 创建训练器
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    trainer = NovelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=config['device'],
        lr=config['lr']
    )
    
    best_loss = trainer.train(
        dataloader=dataloader,
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir'],
        sample_interval=config['sample_interval'],
        resume=False
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳损失: {best_loss:.4f}")
    
    # 测试生成
    print("\n" + "=" * 60)
    print("测试生成...")
    print("=" * 60)
    
    test_prompts = ["在", "银河", "未来", "科技"]
    
    for prompt in test_prompts:
        print(f"\n提示词: '{prompt}'")
        generated = trainer.generate_sample(
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.8,
            top_k=50
        )
        print(f"生成结果: {generated}")
    
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()