import argparse

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from bytephase import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from adam import Adam
from config import GPT2Config
from model import GPT2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    return parser.parse_args()


def train(model, dataloader, optimizer, loss_function, d_vocab, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        targets = F.one_hot(targets, num_classes=d_vocab)

        targets = targets[:, 1:, :]
        logits = logits[:, :-1, :]

        loss = loss_function(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, loss_function, d_vocab, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for batch in progress_bar:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            targets = F.one_hot(targets, num_classes=d_vocab)

            targets = targets[:, 1:, :]
            logits = logits[:, :-1, :]

            loss = loss_function(logits, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def load_and_encode_dataset(dataset_name, tokenizer, batch_size=64, seq_len=1024):
    def tokenize_function(example):
        tokens = tokenizer.encode(example["text"], train_mode=True)
        all_input_ids = []

        for i in range(0, len(tokens), seq_len):
            chunk = tokens[i : i + seq_len]

            if len(chunk) < seq_len:
                chunk = chunk + [256] * (seq_len - len(chunk))
            all_input_ids.append(chunk)

        return {"input_ids": all_input_ids}

    def create_dataloader(data):
        encoded_dataset = data.map(tokenize_function, remove_columns=["text"])
        list_dataset = [{"input_ids": torch.tensor(item["input_ids"], dtype=torch.long)} for item in encoded_dataset]

        return DataLoader(
            list_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: {
                "input_ids": torch.cat([item["input_ids"] for item in batch]),
            },
        )

    data = datasets.load_dataset(dataset_name)

    if "validation" in data:
        train_dataloader = create_dataloader(data["train"])
        val_dataloader = create_dataloader(data["validation"])
    else:
        # If there's no validation split, create one from the train set
        train_val = data["train"].train_test_split(test_size=0.1)
        train_dataloader = create_dataloader(train_val["train"])
        val_dataloader = create_dataloader(train_val["test"])

    return train_dataloader, val_dataloader


def main():
    args = parse_args()
    wandb.init(project="gpt2-training")
    config = GPT2Config()
    tokenizer = Tokenizer()
    tokenizer.load("TK.bpe")
    model = GPT2(config)
    optimizer = Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model.to(device)

    train_data, val_data = load_and_encode_dataset(args.dataset, tokenizer)
    best_val_loss = float("inf")

    for epoch in range(GPT2Config.epochs):
        avg_train_loss = train(model, train_data, optimizer, loss_function, config.d_vocab, device)
        avg_val_loss = validate(model, val_data, loss_function, config.d_vocab, device)

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved!")


if __name__ == "__main__":
    main()
