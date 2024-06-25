import argparse

import datasets
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from bytephase import Tokenizer

from adam import Adam
from config import GPT2Config
from model import GPT2

def loss_function(logits, targets):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    return parser.parse_args()


def train(model, dataloader, optimizer, device):
    total_loss = 0
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input = batch["input_ids"].to(device)
        output = model(input)
        loss = loss_function(output, input)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def load_and_encode_dataset(dataset, tokenizer, context_size=GPT2Config.d_seq):
    data = datasets.load_dataset(dataset)

    def encode_function(text):
        print(text["text"])
        return tokenizer.encode(text["text"])

    encoded_data = data.map(encode_function, batched=True)
    encoded_data.set_format("torch")

    return encoded_data


def main():
    args = parse_args()
    tokenizer = Tokenizer()
    tokenizer.load("pile-10k-10kmerges.bpe")
    model = GPT2(GPT2Config())
    optimizer = Adam(model.parameters())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)

    encoded_dataset = load_and_encode_dataset(args.dataset, tokenizer)
    # train_dataloader = DataLoader(encoded_dataset["train"], GPT2Config.batch_size, shuffle=True)

    for epoch in range(GPT2Config.epochs):
        avg_loss = train(model, encoded_dataset, optimizer, device)
        # log to wandb and text file
        # log in train? log examples?


if __name__ == "__main__":
    main()
