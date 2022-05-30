import torch
from tokenizers import ByteLevelBPETokenizer
from build_dataset import build_dataset
from bert import *
import argparse
from tensorboardX import SummaryWriter
import numpy as np
import random
def setup_seed(random_number):
    torch.manual_seed(random_number)
    torch.cuda.manual_seed(random_number)
    np.random.seed(random_number)
    random.seed(random_number)
setup_seed(9527)
parse=argparse.ArgumentParser()
parse.add_argument('--device',default='cuda')
parse.add_argument('--vocab_size',default=30000,type=int)
parse.add_argument('--max_len',default=512,type=int)
parse.add_argument('--batch_size',default=10,type=int)
parse.add_argument('--data_ratio',default=0.8,type=float)
parse.add_argument('--embedding_dim',default=768,type=int)
parse.add_argument('--d_model',default=64,type=int)
parse.add_argument('--n_layers',default=12,type=int)
parse.add_argument('--n_heads',default=12,type=int)
parse.add_argument('--train_epoch',default=1,type=int)
parse.add_argument('--drop_prob',default=0.1,type=float)
parse.add_argument('--warmup_steps',default=10000,type=int)
parse.add_argument('--lr',default=1e-4,type=float)
parse.add_argument('--beta1',default=0.9,type=float)
parse.add_argument('--beta2',default=0.99,type=float)
parse.add_argument('--weight_decay',default=0.01,type=float)
parse.add_argument('--save_path',default='pretrained_weight')
parse.add_argument('--logging_path',default='tf_logging')
parser=parse.parse_args()


tokenizer = ByteLevelBPETokenizer(
    'bpy_tokenizer_model/vocab.json',
    'bpy_tokenizer_model/merges.txt'
)

eos_token, sep_token, unk_token, cls_token, pad_token, mask_token,seg_ids, vocab_size, max_len, batch_size, percent = \
tokenizer.encode('<cls>').ids[1], tokenizer.encode('<sep>').ids[1], tokenizer.encode('<unk>').ids[1], \
tokenizer.encode('<cls>').ids[1], tokenizer.encode('<pad>').ids[1], tokenizer.encode('<mask>').ids[
    1], [tokenizer.encode('<seg1>').ids[1],tokenizer.encode('<seg2>').ids[1]],parser.vocab_size, parser.max_len, parser.batch_size,parser.data_ratio
train_loader = build_dataset(tokenizer, eos_token, sep_token, unk_token, cls_token, pad_token, mask_token, seg_ids,vocab_size,
                            max_len, batch_size, percent=percent)

writer=SummaryWriter(parser.logging_path)
bert_model=Bert(embedding_dim=parser.embedding_dim,d_model=parser.d_model,n_heads=parser.n_heads,n_layers=parser.n_layers,vocab_size=vocab_size,seq_length=max_len,drop_prob=0.1)
bert_lm=BertLM(bert_model,embedding_dim=parser.embedding_dim,vocab_size=vocab_size)
bert_lm.model.to(parser.device)
bert_lm.mlm.to(parser.device)
bert_lm.nps.to(parser.device)
train_model([train_loader,train_loader],bert_lm,train_epoch=parser.train_epoch,
            hidden=parser.embedding_dim,warmup_steps=parser.warmup_steps,lr=parser.lr,
            beta=(parser.beta1,parser.beta2),weight_decay=parser.weight_decay,device=parser.device,
            save_path=parser.save_path,writer=writer)
