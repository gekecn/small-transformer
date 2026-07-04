"""快速评估模型效果"""
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.small_transformer.model import ChineseGPT
from src.small_transformer.dataset import CharTokenizer, ChineseNovelDataset

def main():
    print("=" * 60)
    print("📊 模型快速评估")
    print("=" * 60)

    # 查找最新的模型
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_epoch_")]
    model_files.sort(key=lambda x: int(x.replace("model_epoch_", "").replace(".pt", "")))

    latest_model = model_files[-1]
    epoch_num = int(latest_model.replace("model_epoch_", "").replace(".pt", ""))
    print(f"\n📂 最新模型: {latest_model} (Epoch {epoch_num})")

    # 加载数据和词表
    print("\n📚 加载数据...")
    data_files = [
        "data/穿进赛博游戏后干掉BOSS成功上位.txt",
        "data/large_scifi.txt"
    ]

    tokenizer = CharTokenizer()
    all_texts = []
    for f in data_files:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                text = fp.read()
                all_texts.append(text)
                print(f"  ✓ {f}: {len(text)} 字符")

    full_text = "\n".join(all_texts)
    tokenizer.build_vocab(full_text)
    print(f"\n📝 词表大小: {tokenizer.vocab_size}")

    # 加载模型
    print("\n🤖 加载模型...")
    checkpoint = torch.load(os.path.join(model_dir, latest_model), map_location='cpu')

    config = checkpoint.get('config', {
        'embed_dim': 256,
        'num_heads': 4,
        'hidden_dim': 512,
        'num_layers': 4,
        'max_seq_len': 256,
        'dropout': 0.1
    })

    model = ChineseGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    print(f"\n📈 训练历史:")
    if train_losses:
        print(f"  初始Loss: {train_losses[0]:.4f}")
        print(f"  最新Loss: {train_losses[-1]:.4f}")
        print(f"  下降幅度: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")

    if val_losses:
        print(f"  验证Loss: {val_losses[-1]:.4f}")

    # 计算困惑度
    if train_losses:
        ppl = 2 ** train_losses[-1]
        print(f"\n📊 训练困惑度: {ppl:.2f}")

    # 生成样本
    print("\n" + "=" * 60)
    print("✍️  生成样本测试")
    print("=" * 60)

    prompts = [
        "在星际的边缘",
        "银河帝国",
        "隗辛站在",
        "人工智能",
    ]

    for prompt in prompts:
        print(f"\n📝 提示词: {prompt}")
        print("-" * 40)

        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        generated = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40
        )
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"生成: {generated_text[len(prompt):]}")

    print("\n" + "=" * 60)
    print("✅ 评估完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
