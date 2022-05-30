import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import random
from transformers import BertTokenizer
from tokenizers import ByteLevelBPETokenizer
import copy


def collate_fn(data):
    return data

class bertlm_dataset(Dataset):

    def __init__(self,tokenizer,eos_token,sep_token,unk_token,cls_token,pad_token,mask_token,seg_ids,vocab_size,max_len,percent=None):
        super().__init__()
        assert os.path.exists('seq_pairs'),print('havent created bert-type dataset yet, please put your data at dataset folder, and run crate_sentence_pairs.py then')
        self.all_files=[]
        for i in os.listdir('seq_pairs'):
            fp=open(os.path.join('seq_pairs',i),'r').read().split('\n')
            if percent!=None and percent<1:
                fp=fp[:int(len(fp)*percent)]
            for i in fp:
                self.all_files.append(i.split('\t'))
        self.eos_token, self.sep_token, self.unk_token, self.cls_token, self.pad_token, self.mask_token=eos_token,sep_token,unk_token,cls_token,pad_token,mask_token
        self.vocab_size=vocab_size
        self.max_len=max_len
        self.tokenizer=tokenizer
        self.seg_ids=seg_ids

    def generated_mask(self, max_len, padding_length):
        return np.tril(np.ones(shape=(max_len, max_len)), padding_length - 1)
    def __getitem__(self, item):
        seq1_id,seq2_id=self.seg_ids
        prob_seq=random.random()
        if prob_seq>0.5:
            t1,t2,seq_label=self.all_files[item][0],self.all_files[item][1],1
        else:
            t1,t2,seq_label=self.all_files[item][0],self.random_sample_sentence(),0
            t2==self.all_files[item][1]
            seq_label=1
        t1_tokens,t1_mask,label_tokens_t1=self.random_word(t1)
        t2_tokens,t2_mask,label_tokens_t2=self.random_word(t2)
        tokens=[self.cls_token]+t1_tokens+[self.sep_token]+t2_tokens+[self.eos_token]
        seg_tokens=[seq1_id for i in range(len(t1_tokens)+2)]+[seq2_id for i in range(len(t2_tokens)+1)]
        masks = [1] + t1_mask + [1] + t2_mask + [1]
        groud_truth=[self.pad_token]+label_tokens_t1+[self.pad_token]+label_tokens_t2+[self.pad_token]
        if len(tokens)<self.max_len:
            attention_mask=self.generated_mask(self.max_len,len(tokens))
            padding=[self.pad_token for i in range(self.max_len-len(tokens))]
            padding2=[1 for i in range(self.max_len-len(tokens))]
            tokens+=padding
            seg_tokens+=copy.deepcopy(padding)
            masks+=padding2
            groud_truth+=copy.deepcopy(padding)
            return {'input':tokens,'mlm_label':groud_truth,'nps_label':seq_label,'masks':masks,'seg_tokens':seg_tokens,'attn_mask':attention_mask}
        else:
            attention_mask=self.generated_mask(self.max_len,self.max_len)
            tokens=tokens[:self.max_len]
            masks=masks[:self.max_len]
            groud_truth=groud_truth[:self.mask_len]
            return {'input':tokens,'mlm_label':groud_truth,'nps_label':seq_label,'masks':masks,'seg_tokens':seg_tokens,'attn_mask':attention_mask}

    def __len__(self):
        return len(self.all_files)

    def random_word(self, sequence):
        tokens=self.tokenizer.encode(sequence).ids[1:-1]
        ground_truth=tokens
        output_label=[]
        for index,token in enumerate(tokens):
            prob = random.random()
            if prob<0.15:
                prob/=0.15
                if prob<0.8:
                    tokens[index]=self.mask_token
                elif prob<0.9:
                    tokens[index]=random.randrange(0,self.vocab_size)
                else:
                    tokens[index]=self.unk_token
                output_label.append(0)
            else:
                output_label.append(1)
        return tokens,output_label,ground_truth

    def random_sample_sentence(self):
        return random.choice(random.choice(self.all_files))

def build_dataset(tokenizer,eos_token,sep_token,unk_token,cls_token,pad_token,mask_token,seg_ids,vocab_size,max_len,batch_size,percent=None):

    return DataLoader(
        dataset=bertlm_dataset(tokenizer,eos_token,sep_token,unk_token,cls_token,pad_token,mask_token,seg_ids,vocab_size,max_len,percent=percent),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

if __name__=='__main__':
    tokenizer = ByteLevelBPETokenizer(
        'bpy_tokenizer_model/vocab.json',
        'bpy_tokenizer_model/merges.txt'
    )
    eos_token, sep_token, unk_token, cls_token, pad_token, mask_token, vocab_size, max_len, batch_size, percent = tokenizer.encode('<s>').ids[1],tokenizer.encode('<s>').ids[1],tokenizer.encode('<unk>').ids[1],tokenizer.encode('<s>').ids[1],tokenizer.encode('<pad>').ids[1],tokenizer.encode('<mask>').ids[1],30000,512,12,0.15
    data_loader=build_dataset(tokenizer,eos_token,sep_token,unk_token,cls_token,pad_token,mask_token,vocab_size,max_len,batch_size,percent=percent)
