import regex
from typing import Dict, Set, Union, List
from collections import Counter

from functools import lru_cache

from _bpe import train, build_trie, manual_free_trie, encode

__version__ = "1.0"

GPT2_REGEX_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    """
    bytephase: A byte pair encoding (BPE) tokenizer with customizable regex pattern and C extension acceleration.

    This tokenizer implements a BPE algorithm for text tokenization, with support
    for training on input data, encoding and decoding text, and saving/loading
    the trained model. It utilizes C extensions for improved performance in critical operations.

    Attributes:
        pattern (str): Regex pattern used for tokenization.
        compiled_pattern (regex.Pattern): Compiled regex pattern.
        decode_dict (dict): Mapping of token IDs to byte sequences.
        _trie: Internal trie structure for efficient encoding (C extension).

    Note:
        The tokenizer uses a trie data structure implemented in C for fast encoding.
        The trie is automatically freed when the Tokenizer instance is deleted.
    """

    def __init__(self, pattern: Union[str, None] = None) -> None:
        """
        Initialize the Tokenizer with an optional regex pattern.

        Args:
            pattern (str, optional): Custom regex pattern for tokenization.
            If None, uses the default GPT-2 pattern. Defaults to None.
        """

        self.pattern = GPT2_REGEX_PATTERN if pattern is None else pattern
        self.compiled_pattern = regex.compile(self.pattern)
        self.decode_dict: Dict[int, bytes] = {}
        self._trie = None

    def __del__(self):
        if self._trie is not None:
            manual_free_trie(self._trie)
            self._trie = None

    def train(self, data: str, vocab_size: int) -> None:
        """
        Train the tokenizer on the given data using the BPE algorithm.

        Args:
            data (str): The text data to train on.
            vocab_size (int): The final size of the vocabulary.

        Raises:
            ValueError: If input data is not a string or vocab_size is not a positive integer.

        Note:
            This method updates the decode_dict attribute and builds the C-based trie structure.
        """
        if not isinstance(data, str):
            raise ValueError("Input data must be a string")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")

        words = {}
        num_merges = vocab_size - 256

        text_chunks = self.compiled_pattern.findall(data)
        words = dict(Counter(text_chunks))
        merges = train(words, len(words), num_merges)

        self.decode_dict = {idx: bytes([idx]) for idx in range(256)}

        idx = 256
        for merge in merges:
            byte_array = bytes(merge)
            self.decode_dict[idx] = byte_array
            idx += 1

        self._trie = build_trie(self.decode_dict)

    # @lru_cache
    def encode(self, input_text: str) -> List[int]:
        """
        Encode the input text into a list of token IDs using the C-based trie structure.

        Args:
            input_text (str): The input text to encode.

        Returns:
            List[int]: A list of token IDs representing the encoded text.

        Raises:
            ValueError: If input_text is not a string.
        """
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string")

        text_chunks = self.compiled_pattern.findall(input_text)
        encoded_text = encode(text_chunks, self._trie)
        return encoded_text

    def decode(self, input_tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            input_tokens (List[int]): A list of token IDs to decode.

        Returns:
            str: The decoded text.

        Raises:
            ValueError: If input is not a list of integers or if an invalid token ID is encountered.
        """
        if not isinstance(input_tokens, List):
            raise ValueError("Input must be a list of integers")

        bytes_array = bytearray()
        for token in input_tokens:
            if token in self.decode_dict:
                bytes_array.extend(self.decode_dict[token])
            else:
                raise ValueError(f"Invalid token id: {token}")

        decoded_text = bytes_array.decode("utf-8", errors="replace")
        return decoded_text

    def save(self, file_name: str, debug=False) -> None:
        """
        Save the trained tokenizer to a file.

        Args:
            file_name (str): The base name for the output file(s).
            debug (bool, optional): If True, also saves a human-readable
                version of the tokenizer. Defaults to False.

        Note:
            Saves the tokenizer to '{file_name}.bpe'.
            If debug is True, also saves to '{file_name}_debug.bpe'.
        """

        output_file = file_name + ".bpe"

        with open(output_file, "w") as f:
            f.write(
                f"bpe tokenizer by benjamin arnav v1\nregex patten: {self.pattern}\n"
            )
            for idx, token in self.decode_dict.items():
                token_ints = " ".join(str(b) for b in token)
                f.write(f"{idx} {token_ints}\n")

        if debug:
            # Outputs a human-readable version
            debug_filename = file_name + "_debug.bpe"
            with open(debug_filename, "w") as f:
                f.write(
                    f"bpe tokenizer by benjamin arnav v1\nregex patten: {self.pattern}\n"
                )
                for idx, token in self.decode_dict.items():
                    token_chars = self.decode(list(token))
                    f.write(f"{idx} {token_chars}\n")

    def load(self, file: str) -> None:
        """
        Load a previously saved tokenizer from a file.

        Args:
            file (str): The path to the .bpe file to load.

        Raises:
            AssertionError: If the file format is invalid or incompatible.

        Note:
            This method updates the decode_dict attribute and rebuilds the C-based trie structure.
        """

        assert file.endswith(".bpe")

        with open(file, "r") as f:
            version_line = f.readline().strip()
            assert version_line == "bpe tokenizer by benjamin arnav v1"
            regex_line = f.readline().split()
            assert regex_line[0] == "regex"

            for line in f:
                current_line = line.strip().split()
                idx = int(current_line[0])
                token = bytes(int(b) for b in current_line[1:])
                self.decode_dict[idx] = token

        self._trie = build_trie(self.decode_dict)

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.decode_dict.keys())
