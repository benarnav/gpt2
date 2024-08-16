# GPT-2 From Scratch

## tl;dr

- *Pure Python/PyTorch implementation*: Built from the ground up for deeper understanding.
- *C extensions for optimized tokenization*: Enhanced training and encoding performance.
- *Faithful recreation based on research papers*: No external implementations or resources.
- *Clear and well-documented code*: Emphasis on readability and comprehension.

## Overview

This project is a pure Python/PyTorch implementation of GPT-2, with custom C extensions for the [tokenizer](https://github.com/benarnav/bytephase) to improve performance. The goal is to recreate GPT-2 solely based on core research papers, enhancing my understanding of the transformer architecture and my ability to produce functional code from academic literature.

## Approach

The implementation is built from the ground up using only the following research papers:

1. Three papers on transformer architectures:
   - [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
   - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
   - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
2. One paper on [byte pair encoding](https://arxiv.org/pdf/1508.07909)
3. Two papers on Adam optimizer: 
   - [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)
   - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/pdf/1711.05101v1)
4. One paper on the [GeLU](https://arxiv.org/pdf/1606.08415v1) activation function
5. One paper on [Layer Normalization](https://arxiv.org/pdf/1607.06450)

By relying solely on these papers, this project ensures a deep understanding of GPT-2's architecture and principles.

## Challenges and Solutions

1. *Tokenization Speed* 

   - **Challenge**: Initial bottleneck due to slow tokenization when implemented in Python.
   - **Solution**: Implemented [C extensions](https://github.com/benarnav/bytephase) for the tokenizer, significantly improving performance in training and encoding.

2. *Missing Information*
   - **Gradient clipping details**:  Addressed by experimenting with common practices in transformer training.
   - **Exact composition of the training dataset**: Used available datasets of comparable composition and ensured robust preprocessing and tokenization.
   - **Distributed training architecture**: No training details were provided

## WIP

- Add training statistics
- Add generation examples
- Distributed Training: Implement multi-GPU and distributed training strategies.