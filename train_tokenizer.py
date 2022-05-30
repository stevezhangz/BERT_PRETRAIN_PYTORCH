import torch
import transformers
from tokenizers import Tokenizer,ByteLevelBPETokenizer
import os


def train_tokenizer(special_tokens,
                    file_dir='dataset',
                    tokenizer_model_path='bpy_tokenizer_model',
                    vocab_size=30000,
                    min_frequency=2):
    file_paths=[os.path.join(file_dir,i) for i in os.listdir(file_dir)]
    if not os.path.exists(tokenizer_model_path):
        os.mkdir(tokenizer_model_path)
    tokenizer=ByteLevelBPETokenizer()
    tokenizer.train(
        file_paths[:5],
        min_frequency=min_frequency,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    tokenizer.save_model(tokenizer_model_path)
    return tokenizer


if __name__=='__main__':
    special_tokens = ['<cls>', '<sep>', '<pad>', '<unk>', '<mask>', '<seg1>', '<seg2>']
    train_tokenizer(special_tokens)