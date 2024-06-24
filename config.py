from dataclasses import dataclass


@dataclass
class GPT2Config:
    d_vocab: int = 50257
    # d_pos: int = 50257
    d_seq: int = 512
    d_model: int = 768
    d_hidden: int = 3072
    batch_size: int = 512
    epochs: int = 100
    d_v: int = 512  # 64
    d_k: int = 512  # 64
    num_heads: int = 12
    num_layers: int = 12
    dropout_rate: float = 0.1
    output_dir: str = "gpt2_output/"
