"""验证模型权重是否正确恢复"""
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.small_transformer.model import ChineseGPT
from src.small_transformer.dataset import CharTokenizer

def main():
    print("=" * 60)
    print("🔍 验证模型权重恢复")
    print("=" * 60)

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

    full_text = "\n".join(all_texts)
    tokenizer.build_vocab(full_text)
    vocab_size = tokenizer.vocab_size

    # 创建两个模型：一个随机初始化，一个从checkpoint加载
    print("\n🤖 创建模型...")

    # 模型1：随机初始化
    model_random = ChineseGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        hidden_dim=512,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    )

    # 模型2：从checkpoint加载
    model_loaded = ChineseGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        hidden_dim=512,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    )

    checkpoint = torch.load("models/model_epoch_24.pt", map_location='cpu')
    model_loaded.load_state_dict(checkpoint['model_state_dict'])

    # 获取权重
    random_state = model_random.state_dict()
    loaded_state = model_loaded.state_dict()

    # 比较权重
    print("\n📊 比较权重:")
    total_layers = 0
    matched_layers = 0
    mismatched_layers = 0

    for key in random_state.keys():
        total_layers += 1
        random_val = random_state[key]
        loaded_val = loaded_state[key]
        
        if torch.equal(random_val, loaded_val):
            matched_layers += 1
        else:
            mismatched_layers += 1
            # 计算差异
            diff = torch.abs(random_val - loaded_val).sum().item()
            print(f"  ❌ {key}: 差异={diff:.6f}")

    print(f"\n📈 结果统计:")
    print(f"  总层数: {total_layers}")
    print(f"  随机一致: {matched_layers}")
    print(f"  随机不一致: {mismatched_layers}")

    # 验证：从checkpoint加载两次，应该完全一致
    print("\n🔄 验证：两次加载同一checkpoint是否一致")
    model_loaded2 = ChineseGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        hidden_dim=512,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    )
    model_loaded2.load_state_dict(checkpoint['model_state_dict'])

    loaded_state2 = model_loaded2.state_dict()
    all_match = True
    for key in loaded_state.keys():
        if not torch.equal(loaded_state[key], loaded_state2[key]):
            all_match = False
            break

    if all_match:
        print("  ✅ 两次加载完全一致！")
    else:
        print("  ❌ 两次加载不一致！")

    # 验证：生成相同输入的输出是否一致
    print("\n🔍 验证：相同输入的输出是否一致")
    model_loaded.eval()
    model_loaded2.eval()

    with torch.no_grad():
        test_input = torch.tensor([tokenizer.encode("在星际")], dtype=torch.long)
        output1 = model_loaded(test_input)
        output2 = model_loaded2(test_input)

    if torch.equal(output1, output2):
        print("  ✅ 相同输入输出一致！")
    else:
        print("  ❌ 相同输入输出不一致！")

    # 验证：和训练时的损失是否匹配
    print("\n📉 验证：检查保存的损失值")
    print(f"  保存的损失值: {checkpoint.get('loss', 'N/A')}")
    print(f"  当前学习率: {checkpoint.get('lr', 'N/A')}")

    print("\n" + "=" * 60)
    print("✅ 验证完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
