"""
中文小说生成模型训练主程序
"""

import torch
import argparse
from datetime import datetime
import hashlib
import os
import sys
from pathlib import Path

from src.small_transformer.model import ChineseGPT
from src.small_transformer.dataset import create_train_val_dataloaders
from src.small_transformer.trainer import NovelTrainer


def configure_console_output():
    """避免生成文本中的生僻字符使Windows控制台输出崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')


def find_latest_checkpoint(run_dir):
    checkpoints = []
    for filename in os.listdir(run_dir):
        if filename.startswith('model_epoch_') and filename.endswith('.pt'):
            try:
                epoch = int(filename.removeprefix('model_epoch_').removesuffix('.pt'))
                checkpoints.append((epoch, os.path.join(run_dir, filename)))
            except ValueError:
                continue
    if not checkpoints:
        raise FileNotFoundError(f"运行目录中没有checkpoint: {run_dir}")
    return max(checkpoints)[1]


def resolve_run_paths(model_root, resume_run=None):
    if resume_run:
        resume_path = os.path.abspath(resume_run)
        if os.path.isdir(resume_path):
            return resume_path, find_latest_checkpoint(resume_path)
        if os.path.isfile(resume_path):
            return os.path.dirname(resume_path), resume_path
        raise FileNotFoundError(f"找不到恢复路径: {resume_path}")

    os.makedirs(model_root, exist_ok=True)
    run_name = datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
    run_dir = os.path.join(model_root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, None


def corpus_manifest(data_dir):
    """记录实际参与训练的顶层txt文件，防止语料变化后误续训。"""
    records = []
    for path in sorted(Path(data_dir).glob('*.txt')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        records.append({
            'file': path.name,
            'bytes': path.stat().st_size,
            'sha256': digest.hexdigest(),
        })
    if not records:
        raise ValueError(f"训练目录中没有txt语料: {data_dir}")
    return records


def tokenizer_fingerprint(tokenizer):
    return tokenizer.fingerprint()


def load_initial_weights(model, tokenizer, checkpoint_path, current_data_manifest):
    """校验并加载模型权重，但不继承旧优化器、调度器、轮次和随机状态。"""
    source = os.path.abspath(checkpoint_path)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"找不到初始权重checkpoint: {source}")
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if checkpoint.get('model_config') != model.get_config():
        raise ValueError("初始权重的模型配置与当前模型不一致")
    if checkpoint.get('tokenizer_fingerprint') != tokenizer_fingerprint(tokenizer):
        raise ValueError("初始权重的Tokenizer与当前语料生成的Tokenizer不一致")
    source_data = (checkpoint.get('training_config') or {}).get('data')
    if source_data is not None and source_data != current_data_manifest:
        raise ValueError("初始权重对应的语料指纹与当前语料不一致")
    model.load_state_dict(checkpoint['model_state_dict'])
    return source


def main():
    configure_console_output()
    parser = argparse.ArgumentParser(description="训练中文微型Transformer")
    parser.add_argument(
        '--resume-run',
        help="已有run目录或具体checkpoint路径；不提供则创建全新运行目录",
    )
    parser.add_argument(
        '--epochs', type=int,
        help="本次运行截止到第几轮；默认使用配置的最大轮数，可用于先完整试跑1轮",
    )
    parser.add_argument(
        '--init-checkpoint',
        help="从已有checkpoint的模型权重创建独立精调运行；不继承旧优化器和轮次",
    )
    parser.add_argument('--learning-rate', type=float, help="覆盖默认学习率")
    parser.add_argument(
        '--warmup-steps', type=int, default=50,
        help="线性预热的优化步骤数；默认前50个Batch预热，设为0可关闭",
    )
    parser.add_argument(
        '--tokenizer', choices=('bpe', 'char'), default='bpe',
        help="新训练使用的Tokenizer；默认使用本语料训练的BPE",
    )
    parser.add_argument('--tokenizer-vocab-size', type=int, help="覆盖Tokenizer词表大小")
    args = parser.parse_args()
    if args.resume_run and args.init_checkpoint:
        parser.error("--resume-run 与 --init-checkpoint 不能同时使用")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate 必须大于0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps 不能小于0")
    if args.tokenizer_vocab_size is not None and args.tokenizer_vocab_size < 1000:
        parser.error("--tokenizer-vocab-size 不能小于1000")
    print("=" * 60)
    print("中文小说生成模型")
    print("=" * 60)
    
    # 配置参数
    config = {
        # 数据参数
        # 仅加载科幻小说目录；语料准备工具和历史通用语料位于相邻tools项目。
        'data_dir': 'data_scifi',
        'batch_size': 64,
        'max_seq_len': 256,
        'num_workers': 0,       # 保证加权采样位置可精确恢复；数据已在内存，取数并非瓶颈
        'cpu_threads': 4,
        'tokenizer_type': 'sentencepiece_bpe' if args.tokenizer == 'bpe' else 'char',
        'tokenizer_vocab_size': args.tokenizer_vocab_size or (6000 if args.tokenizer == 'bpe' else 5000),
        
        # 模型参数
        'embed_dim': 256,       # 嵌入维度
        'num_heads': 4,         # 注意力头数
        'hidden_dim': 512,      # 隐藏层维度
        'num_layers': 4,        # Transformer层数
        'dropout': 0.1,         # Dropout率
        
        # 训练参数
        'num_epochs': 16,       # A/B语料交替训练16轮，相当于完整覆盖全部语料8遍
        'lr': 3e-4,             # 目标学习率；前50个Batch由线性预热逐步升到此值
        'device': 'cpu',        # 设备（CPU训练）
        'model_root': 'models', # 每次训练在其下创建独立运行目录
        'val_ratio': 0.02,      # 留出2%整篇文档；验证开销很小，又能识别过拟合
        'validation_start_epoch': 4,  # 前3轮只训练，从第4轮开始检查泛化
        'validation_interval': 2,     # 此后每2轮验证一次；最后一轮始终验证
        'alternating_train_parts': 2, # 奇数轮训练A，偶数轮训练B；两轮看完全部语料
        # 目录内已经全部是科幻小说，不再用采样权重补偿百科/新闻。
        'source_weights': {},
        'early_stopping_patience': 2, # 连续2次验证未改善再早停
        
        # 生成参数
        'sample_interval': 2,   # 每隔多少轮生成样本
    }
    run_epochs = args.epochs if args.epochs is not None else config['num_epochs']
    if not 1 <= run_epochs <= config['num_epochs']:
        parser.error(f"--epochs 必须在1到{config['num_epochs']}之间")
    if args.learning_rate is not None:
        config['lr'] = args.learning_rate
    
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # i7-12650H实测开满混合核心反而变慢；在创建模型前限制计算线程。
    torch.set_num_threads(config['cpu_threads'])
    
    # 加载数据
    print("\n" + "=" * 60)
    print("加载数据...")
    print("=" * 60)

    run_dir, resume_from = resolve_run_paths(config['model_root'], args.resume_run)
    print(f"运行目录: {run_dir}")
    if resume_from:
        print(f"恢复来源: {resume_from}")
    
    dataloader, val_dataloader, tokenizer = create_train_val_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        val_ratio=config['val_ratio'],
        source_weights=config['source_weights'],
        num_workers=config['num_workers'],
        tokenizer_type=config['tokenizer_type'],
        tokenizer_vocab_size=config['tokenizer_vocab_size'],
        alternating_train_parts=config['alternating_train_parts'],
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
    data_manifest = corpus_manifest(config['data_dir'])
    initial_weights_from = None
    if args.init_checkpoint:
        initial_weights_from = load_initial_weights(
            model, tokenizer, args.init_checkpoint, data_manifest,
        )
        print(f"已加载初始模型权重: {initial_weights_from}")
        print("优化器、学习率调度器和训练轮次将从头开始")
    
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
    
    scheduler_t_max = run_epochs if args.warmup_steps else config['num_epochs']
    training_config = {
        'data': data_manifest,
        'batch_size': config['batch_size'],
        'max_seq_len': config['max_seq_len'],
        'val_ratio': config['val_ratio'],
        'validation_start_epoch': config['validation_start_epoch'],
        'validation_interval': config['validation_interval'],
        'alternating_train_parts': config['alternating_train_parts'],
        'split_seed': 42,
        'split_strategy_version': 2,
        'source_weights': config['source_weights'],
        'num_workers': config['num_workers'],
        'cpu_threads': config['cpu_threads'],
        'tokenizer_type': config['tokenizer_type'],
        'tokenizer_vocab_size': config['tokenizer_vocab_size'],
        'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer),
        'learning_rate': config['lr'],
        'warmup_steps': args.warmup_steps,
        'total_training_steps': len(dataloader) * run_epochs,
        'scheduler_t_max': scheduler_t_max,
        'device': config['device'],
    }
    if initial_weights_from:
        training_config.update({
            'run_mode': 'fine_tune',
            'initial_weights_from': initial_weights_from,
        })

    trainer = NovelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=config['device'],
        lr=config['lr'],
        scheduler_t_max=scheduler_t_max,
        warmup_steps=args.warmup_steps,
        total_training_steps=len(dataloader) * run_epochs,
        training_config=training_config,
    )
    
    best_loss = trainer.train(
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        num_epochs=run_epochs,
        save_dir=run_dir,
        sample_interval=config['sample_interval'],
        early_stopping_patience=config['early_stopping_patience'],
        validation_start_epoch=config['validation_start_epoch'],
        validation_interval=config['validation_interval'],
        resume_from=resume_from,
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
        )
        print(f"生成结果: {generated}")
    
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
