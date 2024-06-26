import argparse

import datasets
import torch
import torch.nn as nn
import wandb
from bytephase import Tokenizer
from torch.utils.data import DataLoader

from adam import Adam
from config import GPT2Config
from model import GPT2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    return parser.parse_args()


def train(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # TODO gradient clipping
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)

        logits = logits.view(-1, logits.size(-1))  # Flatten to (batch_size * sequence_length, vocab_size)
        targets = targets.view(-1)  # Flatten to (batch_size * sequence_length)

        loss = loss_function(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = loss_function(logits, targets)
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
    wandb.init(project="gpt2-training")

    tokenizer = Tokenizer()
    tokenizer.load("TK.bpe")
    model = GPT2(GPT2Config())
    optimizer = Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model.to(device)

    train_data, val_data = load_and_encode_dataset(args.dataset, tokenizer)
    # train_dataloader = DataLoader(encoded_dataset["train"], GPT2Config.batch_size, shuffle=True)

    for epoch in range(GPT2Config.epochs):
        avg_train_loss = train(model, train_data, optimizer, loss_function, device)
        avg_val_loss = validate(model, val_data, loss_function, device)

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        #TODO state dict?


if __name__ == "__main__":
    main()
