"""
This script trains a GPT-2 model on a specified dataset using the 
Adam optimizer and logs the training process with Weights & Biases (wandb).

Functions:
    parse_args():
        Parses command line arguments to get the dataset name.
    
    train(model, dataloader, optimizer, loss_function, d_vocab, device):
        Trains the model for one epoch.

    validate(model, dataloader, loss_function, d_vocab, device):
        Validates the model on the validation dataset.

    load_and_encode_dataset(dataset_name, tokenizer, batch_size=64, seq_len=1024):
        Loads and tokenizes the dataset, then creates dataloaders for training and validation.

    main():
        Main function to initialize the training process.

Usage:
    Run the script from the command line with the required arguments. For example:
        python train.py --dataset <dataset_name>
"""

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
    """
    Parses command line arguments to get the dataset name.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    return parser.parse_args()


def train(model, dataloader, optimizer, loss_function, d_vocab, device) -> float:
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader): Dataloader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_function (torch.nn.Module): Loss function.
        d_vocab (int): Size of the vocabulary.
        device (torch.device): Device to train the model on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # print(batch)
        inputs = batch["input_ids"].to(device)
        targets = inputs[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)

        # targets = targets[:, 1:, :]
        targets = F.one_hot(targets, num_classes=d_vocab)
        logits = logits[:, :-1, :]

        loss = loss_function(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, loss_function, d_vocab, device) -> float:
    """
    Validates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (DataLoader): Dataloader for the validation data.
        loss_function (torch.nn.Module): Loss function.
        d_vocab (int): Size of the vocabulary.
        device (torch.device): Device to validate the model on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch["input_ids"].to(device)
            targets = inputs[:, 1:]

            logits = model(inputs)
            targets = F.one_hot(targets, num_classes=d_vocab)

            logits = logits[:, :-1, :]

            loss = loss_function(logits, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def load_and_encode_dataset(dataset_name, tokenizer, batch_size=64, seq_len=1024) -> tuple:
    """
    Loads and tokenizes the dataset, then creates dataloaders for training and validation.

    Args:
        dataset_name (str): Name of the dataset to load.
        tokenizer (Tokenizer): Tokenizer to encode the dataset.
        batch_size (int, optional): Batch size for the dataloaders. Default is 64.
        seq_len (int, optional): Sequence length for the tokenized inputs. Default is 1024.

    Returns:
        tuple: Dataloaders for training and validation datasets.
    """

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
    """
    Main function to initialize the training process.

    This function:
    - Parses command line arguments.
    - Initializes wandb for logging.
    - Loads the tokenizer and model.
    - Loads the dataset and creates dataloaders.
    - Trains and validates the model for a specified number of epochs.
    - Logs the training and validation losses to wandb.
    - Saves the best model based on validation loss.
    """
    args = parse_args()
    # wandb.init(project="gpt2-training")
    config = GPT2Config()
    tokenizer = Tokenizer()
    tokenizer.load("pile10k-20kmerges.bpe")
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

        # wandb.log(
        #     {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        # )
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved!")


if __name__ == "__main__":
    main()
