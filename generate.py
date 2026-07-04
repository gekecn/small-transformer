"""
测试模型生成效果
"""

import torch
import os

from src.small_transformer.model import ChineseGPT
from src.small_transformer.dataset import create_dataloader

def main():
    print("=" * 60)
    print("测试模型生成效果")
    print("=" * 60)
    
    device = 'cpu'
    save_dir = 'models'
    
    print("加载数据和Tokenizer...")
    dataloader, tokenizer = create_dataloader(
        'data',
        batch_size=4,
        max_seq_len=256
    )
    
    print(f"词表大小: {tokenizer.vocab_size}")
    
    print("\n创建模型...")
    model = ChineseGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_heads=4,
        hidden_dim=512,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    ).to(device)
    
    checkpoint_path = os.path.join(save_dir, 'model_epoch_50.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"\n加载模型: model_epoch_50.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', 'N/A')
        print(f"训练到Epoch: {epoch}, Loss: {loss}")
    else:
        print("\n⚠️ 未找到模型checkpoint")
        return
    
    model.eval()
    
    test_prompts = ["在", "银河", "未来", "科技", "星际", "人类", "宇宙", "城市"]
    
    output_lines = []
    
    for prompt in test_prompts:
        header = f"\n{'='*60}\n提示词: '{prompt}'\n{'='*60}"
        print(header)
        output_lines.append(header)
        
        for temp in [0.5, 0.7, 1.0]:
            for top_k_val in [20, 50]:
                idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    generated = model.generate(
                        idx,
                        max_new_tokens=200,
                        temperature=temp,
                        top_k=top_k_val
                    )
                
                text = tokenizer.decode(generated[0].tolist())
                result = f"  temp={temp}, top_k={top_k_val}:\n    {text}"
                print(result)
                output_lines.append(result)
    
    output_file = 'generated_output.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n生成结果已保存到: {output_file}")
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
