from dataclasses import dataclass


@dataclass
class GPT2Config:
    d_vocab: int = 50257
    d_seq: int = 1024
    d_model: int = 768
    d_hidden: int = 3072
    batch_size: int = 512
    epochs: int = 100
    d_head: int = 64  # d_model / num_heads
    num_heads: int = 12
    num_layers: int = 12
    dropout_rate: float = 0.01
    output_dir: str = "gpt2_output/"
