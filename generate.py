"""
测试模型生成效果
"""

import torch
import argparse
from src.small_transformer.model_io import load_trained_model


DEFAULT_PROMPTS = [
    "火星",
    "机器人",
    "飞船",
    "星空",
]

def main():
    parser = argparse.ArgumentParser(description="使用训练好的微型Transformer生成文本")
    parser.add_argument('--run-dir', help="指定某次训练的run目录")
    parser.add_argument('--checkpoint', help="直接指定checkpoint文件")
    parser.add_argument(
        '--prompt', action='append',
        help="生成提示词，一个词即可；可重复提供",
    )
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--repetition-penalty', type=float, default=1.05)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='generated_output.txt')
    args = parser.parse_args()

    print("=" * 60)
    print("测试模型生成效果")
    print("=" * 60)

    device = 'cpu'
    model_root = 'models'
    try:
        model, tokenizer, checkpoint, checkpoint_path, run_dir, inferred = load_trained_model(
            model_root,
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            device=device,
        )
    except FileNotFoundError as error:
        print(f"无法生成：{error}")
        return
    print(f"运行目录: {run_dir}")
    print(f"模型文件: {checkpoint_path}")

    print("已加载与模型配套的Tokenizer。")
    print(f"词表大小: {tokenizer.vocab_size}")
    if inferred:
        print("提示：旧checkpoint没有模型配置，已按本项目历史配置推断（注意力头数=4）。")
    epoch = checkpoint['epoch']
    loss = checkpoint.get('val_loss', checkpoint.get('loss', 'N/A'))
    print(f"训练到Epoch: {epoch}, 验证Loss: {loss}")
    
    model.eval()
    
    test_prompts = args.prompt or DEFAULT_PROMPTS
    
    output_lines = []
    
    for prompt in test_prompts:
        header = f"\n{'='*60}\n提示词: '{prompt}'\n{'='*60}"
        print(header)
        output_lines.append(header)
        
        torch.manual_seed(args.seed)
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

        with torch.no_grad():
            generated = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                forbidden_token_ids=tokenizer.special_token_ids,
            )

        text = tokenizer.decode(generated[0].tolist())
        result = (
            f"  temp={args.temperature}, top_k={args.top_k}, "
            f"repetition_penalty={args.repetition_penalty}, "
            f"no_repeat_ngram={args.no_repeat_ngram_size}:\n    {text}"
        )
        print(result)
        output_lines.append(result)
    
    output_file = args.output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n生成结果已保存到: {output_file}")
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
