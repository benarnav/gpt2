# GPT-2 From Scratch

## Overview
The project aims to recreate the GPT-2 model without relying on external implementations or resources beyond the core research papers. It is a pure Python/PyTorch implementation of GPT-2, focused on deepening my understanding of the transformer architecture and improving my ability to produce functional code based on academic papers. 

## tl;dr
- Pure Python/PyTorch implementation
- C extensions for optimized tokenization
- Faithful recreation of GPT-2 based solely on research papers

## Approach
The implementation is built from the ground up using only the following research papers:
1. Three papers on transformer architectures
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
2. One paper on [byte pair encoding](https://arxiv.org/pdf/1508.07909)
3. Two papers on [Adam](https://arxiv.org/pdf/1412.6980) [optimizer](https://arxiv.org/pdf/1711.05101v1) implementation
4. One paper on the [GeLU](https://arxiv.org/pdf/1606.08415v1) activation function
5. One paper on [Layer Normalization](https://arxiv.org/pdf/1607.06450)

This approach ensures a deep understanding of the model's architecture and underlying principles.

## Challenges and Solutions
1. **Tokenization Speed**: The speed of the tokenizer was initially a major bottleneck. This was overcome by [implementing C extensions](https://github.com/benarnav/bytephase), significantly improving performance.

2. **Missing Information**: The original paper did not provide complete information on certain aspects:
   - Gradient clipping details
   - Exact composition of the training dataset
   - Distributed training architecture

   These gaps make training from scratch challenging and will require innovative solutions.