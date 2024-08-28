# transformer_from_scratch

## Overview
This project implements a Transformer model from scratch using PyTorch for English to French translation. The model is trained on a large parallel corpus and utilizes AWS EC2 instances with powerful GPUs for efficient training. The project also includes an application built on top of the trained model to provide a user-friendly interface for translating text between English and French.

## Features
- Implementation of the Transformer model architecture from scratch.
- Training using PyTorch on AWS EC2 GPU instances.
- Preprocessing of text data using SentencePiece for subword tokenization.
- Evaluation using BLEU score and other relevant metrics.
- Application for translating text, with an API for easy integration.

## Getting Started

### Prerequisites
- Python 3.7 or later
- PyTorch
- Access to an AWS account with EC2 instance and GPU support

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/eng-fra-transformer.git
   cd transformer_from_scratch

