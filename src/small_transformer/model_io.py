"""训练运行、checkpoint 和配套 tokenizer 的统一加载工具。"""

import os
from pathlib import Path

import torch

from .dataset import load_saved_tokenizer
from .model import ChineseGPT


def latest_epoch_checkpoint(run_dir):
    checkpoints = []
    for path in Path(run_dir).glob("model_epoch_*.pt"):
        try:
            epoch = int(path.stem.removeprefix("model_epoch_"))
        except ValueError:
            continue
        checkpoints.append((epoch, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def resolve_checkpoint(model_root="models", checkpoint=None, run_dir=None):
    model_root = os.path.abspath(model_root)
    if checkpoint:
        selected_checkpoint = os.path.abspath(checkpoint)
        if not os.path.isfile(selected_checkpoint):
            raise FileNotFoundError(f"找不到checkpoint: {selected_checkpoint}")
        return selected_checkpoint, os.path.dirname(selected_checkpoint)

    if run_dir:
        selected_run = os.path.abspath(run_dir)
    else:
        run_dirs = [
            path for path in Path(model_root).glob("run-*")
            if path.is_dir()
        ]
        if not run_dirs:
            raise FileNotFoundError("没有找到当前训练运行，请先运行 python main.py")
        # 运行目录名包含创建时间；直接选择最新一次启动的训练，不回退到旧模型。
        selected_run = str(max(run_dirs, key=lambda path: path.name))

    if not os.path.isdir(selected_run):
        raise FileNotFoundError(f"找不到运行目录: {selected_run}")
    # “当前模型”指最新训练轮次，而不是历史最佳轮次。
    selected_checkpoint = latest_epoch_checkpoint(selected_run)
    if not selected_checkpoint:
        raise FileNotFoundError(
            "当前模型还没产生，请等待至少完成第1个Epoch后再运行 generate.py"
        )
    return selected_checkpoint, selected_run


def infer_model_config(checkpoint):
    model_config = checkpoint.get("model_config") or checkpoint.get("config")
    if model_config and "vocab_size" in model_config:
        return model_config, False

    state = checkpoint["model_state_dict"]
    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    inferred = {
        "vocab_size": state["token_embedding.weight"].shape[0],
        "embed_dim": state["token_embedding.weight"].shape[1],
        "num_heads": int((model_config or {}).get("num_heads", 4)),
        "hidden_dim": state["blocks.0.ffn.0.weight"].shape[0],
        "num_layers": max(block_indices) + 1,
        "max_seq_len": state["position_embedding.weight"].shape[0],
        "dropout": float((model_config or {}).get("dropout", 0.1)),
    }
    return inferred, True


def load_trained_model(model_root="models", checkpoint=None, run_dir=None, device="cpu"):
    checkpoint_path, selected_run = resolve_checkpoint(model_root, checkpoint, run_dir)
    tokenizer_path = os.path.join(selected_run, "tokenizer.pt")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"找不到与模型配套的Tokenizer: {tokenizer_path}")

    tokenizer = load_saved_tokenizer(tokenizer_path)
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config, inferred = infer_model_config(checkpoint_data)
    if model_config["vocab_size"] != tokenizer.vocab_size:
        raise ValueError(
            f"模型词表大小({model_config['vocab_size']})与Tokenizer({tokenizer.vocab_size})不一致"
        )
    model = ChineseGPT(**model_config).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    return model, tokenizer, checkpoint_data, checkpoint_path, selected_run, inferred
