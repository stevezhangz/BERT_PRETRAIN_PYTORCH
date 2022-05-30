import os

from torch import nn
import torch
from torch.nn import functional as K
import numpy as np
from utils import topk_acc,ScheduledOptim

class ScaledDotProductAttention(nn.Module):
    def __init__(self,scale_factor=None,drop_prob=0.1):
        super().__init__()
        self.scale=scale_factor
        self.drop=nn.Dropout(drop_prob)

    def forward(self,q,k,v,mask=None):
        attn=torch.matmul(q,k.transpose(-2,-1))

        if mask!=None:
            mask.unsqueeze(1).repeat(1,attn.size(1),1,1)
            try:
                attn.masked_fill(mask==0,-1e9)
            except:
                mask=mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
                attn.masked_fill(mask == 0, -1e9)
        attn=self.drop(K.softmax(attn/self.scale,dim=-1))
        return torch.matmul(attn,v),attn


class MultiheadAttention(nn.Module):
    def __init__(self,embedding_dim,d_model,n_heads,drop_prob=0.1):
        super().__init__()
        self.linear_proj_q = nn.Linear(embedding_dim, d_model * n_heads)
        self.linear_proj_k = nn.Linear(embedding_dim, d_model * n_heads)
        self.linear_proj_v = nn.Linear(embedding_dim, d_model * n_heads)
        self.attn_layer=ScaledDotProductAttention(scale_factor=d_model**0.5,drop_prob=drop_prob)
        self.dim_container=[d_model,n_heads]
        self.norm=nn.LayerNorm(embedding_dim,1e-6)

    def forward(self,q,k,v,mask=None):
        d_model,n_heads=self.dim_container
        batch_size,seq_length=q.size()[:2]
        projed_q = self.linear_proj_q(q).view(batch_size,seq_length,n_heads,d_model).transpose(1,2)
        projed_k = self.linear_proj_k(k).view(batch_size,seq_length,n_heads,d_model).transpose(1,2)
        projed_v = self.linear_proj_v(v).view(batch_size,seq_length,n_heads,d_model).transpose(1,2)
        output, attn=self.attn_layer(projed_q,projed_k,projed_v,mask=mask)
        output=output.transpose(1,2).flatten(2)
        return attn,self.norm(output+q)

class PosistionWiseFeedForward(nn.Module):
    def __init__(self,embedding_dim,drop_prob=0.1):
        super().__init__()
        self.act=nn.GELU()
        self.Linear1=nn.Linear(embedding_dim,embedding_dim*4)
        self.Linear2=nn.Linear(embedding_dim*4,embedding_dim)
        self.drop_layer=nn.Dropout(drop_prob)
        self.norm=nn.LayerNorm(embedding_dim,1e-6)

    def forward(self,x):
        res=x
        pre=self.Linear2(self.drop_layer(self.act(self.Linear1(x))))
        return self.norm(res+pre)

class PosistionEmbedding(nn.Module):
    def __init__(self,seq_length,embedding_dim):
        super().__init__()
        raw_pos_embedding=[]
        for pos in range(seq_length):
            emb=[]
            for i in range(embedding_dim):
                emb.append(np.array(pos/(10000**(2*(i//2)/embedding_dim))))
            raw_pos_embedding.append(emb)
        raw_pos_embedding=np.array(raw_pos_embedding)
        raw_pos_embedding[:,0::2]=np.sin(raw_pos_embedding[:,0::2])
        raw_pos_embedding[:,1::2]=np.cos(raw_pos_embedding[:,1::2])
        self.register_buffer('raw_pos_embedding',torch.from_numpy(raw_pos_embedding))

    def forward(self,x):
        return self.raw_pos_embedding[:x.size(1)].unsqueeze(0).to(x.device).to(x.dtype)+x

class EncoderBlock(nn.Module):
    def __init__(self,embedding_dim,d_model,n_heads,drop_prob=0.1):
        super().__init__()
        self.feed_forward=PosistionWiseFeedForward(embedding_dim,drop_prob)
        self.multihead_attention=MultiheadAttention(embedding_dim,d_model,n_heads,drop_prob)
    def forward(self,q,k,v,mask=None):
        attn,output=self.multihead_attention(q,k,v,mask)
        output=self.feed_forward(output)
        return attn,output

class Bert(nn.Module):
    def __init__(self,embedding_dim,d_model,n_heads,n_layers,vocab_size,seq_length,drop_prob=0.1):
        super().__init__()
        self.word_emb_layer=nn.Embedding(vocab_size,embedding_dim)
        self.pos_emb_layer=PosistionEmbedding(seq_length,embedding_dim)
        model_list=[]
        for i in range(n_layers):
            model_list.append(EncoderBlock(embedding_dim, d_model, n_heads, drop_prob))
        self.transformer=nn.ModuleList(model_list)
        self.norm=nn.LayerNorm(embedding_dim,1e-6)

    def forward(self,x,seg_tokens,mask=None,return_attn=False):
        word_embd=self.word_emb_layer(x)
        x=self.norm(self.pos_emb_layer(seg_tokens.unsqueeze(-1).expand_as(word_embd)+word_embd))
        if return_attn:
            attn_record=[]
        for sub_layer in self.transformer:
            attn,output=sub_layer(x,x,x,mask)
            if return_attn:
                attn_record.append(attn)
        if return_attn:
            return output,attn_record
        else:
            return output

class BertMLM(nn.Module):
    def __init__(self,embedding_dim,vocab_size):
        super().__init__()
        self.token_pre=nn.Linear(embedding_dim,vocab_size)
    def forward(self,hidden_states,pre_mask):
        return self.token_pre(hidden_states),self.token_pre(hidden_states.masked_fill(pre_mask.unsqueeze(-1).expand_as(hidden_states)==1,0))

class BertNPS(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.bilevel_pre=nn.Linear(embedding_dim,2)
    def forward(self,hidden_states):
        return self.bilevel_pre(hidden_states)

class BertLM(nn.Module):
    def __init__(self,bert_model,embedding_dim,vocab_size):
        super().__init__()
        self.model=bert_model
        self.mlm=BertMLM(embedding_dim,vocab_size)
        self.nps=BertNPS(embedding_dim)

    def forward(self,x,seg_tokens,pre_mask=None,mask=None):
        output=self.model(x,seg_tokens,mask)
        mlm_pre,mlm_pre_for_loss=self.mlm(output,pre_mask)
        nps_pre=self.nps(output)
        return mlm_pre,mlm_pre_for_loss,nps_pre

def train_model(data_loader,bert_lm,train_epoch,hidden=768,
                warmup_steps=10000,lr=1e-4,beta=(0.9,0.99),
                weight_decay=0.01,device='cpu',save_path=None,
                writer=None,log_freq=100):
    """
    this method is used to train bert,
    :param data_loader:
    :param bert_lm:
    :return:
    """
    if len(data_loader)==2:
        train_loader,test_loader=data_loader
    else:
        train_loader=data_loader
        test_loader=None
    optimizer=torch.optim.Adam(bert_lm.parameters(),lr=lr,betas=beta,weight_decay=weight_decay)
    loss_fn=nn.CrossEntropyLoss()
    optim_schedule = ScheduledOptim(optimizer, hidden, n_warmup_steps=warmup_steps)
    for epc in range(train_epoch):
        bert_lm.train()
        mlm_count1 = 0
        nps_count1 = 0
        mlm_count5 = 0
        total_num = 0
        mlm_count=0
        for step,data in enumerate(train_loader):
            total_num+=len(data)
            new_data={k:[] for k in data[0]}
            for slice in data:
                for k in slice:
                    new_data[k].append(slice[k])
            for k in new_data:
                new_data[k]=torch.from_numpy(np.array(new_data[k])).to(device)
            data=new_data
            optim_schedule.zero_grad()
            mlm_pre,mlm_pre_for_loss,nps_pre=bert_lm(data['input'],data['seg_tokens'],data['masks'],data['attn_mask'])
            batch_size,seq_length,vocab_size=mlm_pre.size()
            loss_mlm=loss_fn(mlm_pre.view(batch_size*seq_length,vocab_size),data['mlm_label'].view(batch_size*seq_length))
            loss_nps =loss_fn(nps_pre[:,0],data['nps_label'])
            loss=loss_nps+loss_mlm
            print(f'epoch {epc}---training process: nps_loss: ',loss_nps.detach().cpu().numpy(),' mlm_loss: ',loss_mlm.detach().cpu().numpy(),' added_loss: ',loss.detach().cpu().numpy())
            loss.backward()
            optim_schedule.step_and_update_lr()
            mlm_count1 += topk_acc(mlm_pre.detach().view(batch_size*seq_length,vocab_size), 1,data['mlm_label'].view(batch_size*seq_length))
            nps_count1 += topk_acc(nps_pre[:, 0], 1, data['nps_label'])
            mlm_count5 += topk_acc(mlm_pre.detach().view(batch_size*seq_length,vocab_size), 5,data['mlm_label'].view(batch_size*seq_length))
            mlm_count+=batch_size*seq_length
            print(f'epoch {epc}---{step}---training process: mlm_top1_acc: {mlm_count1 / mlm_count} mlm_top5_acc:  {mlm_count5 / mlm_count} nps_top1_acc: {nps_count1 / total_num} ')
            if writer!=None and step%log_freq==0:
                writer.add_scalar("train/train_mlm_loss",loss_mlm.detach().cpu().numpy(),step)
                writer.add_scalar("train/train_nps_loss",loss_nps.detach().cpu().numpy(),step)
                writer.add_scalar("train/train_total_loss",loss.detach().cpu().numpy(),step)
        if writer!=None:
            writer.add_scalar('train/train_mlm_top1',mlm_count1,epc)
            writer.add_scalar('train/train_mlm_top5',mlm_count5,epc)
            writer.add_scalar('train/train_nps_top1',nps_count1,epc)
        if test_loader:
            mlm_count1=0
            nps_count1=0
            mlm_count5=0
            mlm_count=0
            bert_lm.eval()
            total_num=0
            for data in test_loader:
                total_num+=len(data)
                new_data = {k: [] for k in data[0]}
                for slice in data:
                    for k in slice:
                        new_data[k].append(slice[k])
                for k in new_data:
                    try:
                        new_data[k] = torch.from_numpy(new_data[k]).to(device)
                    except:
                        if isinstance(new_data[k],list):
                            new_data[k] = torch.from_numpy(np.array(new_data[k])).to(device)
                data = new_data
                mlm_pre,mlm_pre_for_loss,nps_pre = bert_lm(data['input'],data['seg_tokens'],data['masks'],data['attn_mask'])
                batch_size, seq_length, vocab_size = mlm_pre.size()
                mlm_count1 += topk_acc(mlm_pre.detach().view(batch_size * seq_length, vocab_size), 1,
                                       data['mlm_label'].view(batch_size * seq_length))
                nps_count1 += topk_acc(nps_pre[:, 0], 1, data['nps_label'])
                mlm_count5 += topk_acc(mlm_pre.detach().view(batch_size * seq_length, vocab_size), 5,
                                       data['mlm_label'].view(batch_size * seq_length))
                mlm_count+=batch_size*seq_length
                print(f'epoch {epc}---eval process: mlm_top1_acc: {mlm_count1 / mlm_count} mlm_top5_acc:  {mlm_count5/mlm_count} nps_top1_acc: {nps_count1 / total_num} ')
            if writer!=None :
                writer.add_scalar('test/train_mlm_top1',mlm_count1,epc)
                writer.add_scalar('test/train_mlm_top5',mlm_count5,epc)
                writer.add_scalar('test/train_nps_top1',nps_count1,epc)

            if save_path:
                torch.save({'backbone':bert_lm.model.parameters(),'mlm':bert_lm.mlm.parameters(),'nps':bert_lm.nps.parameters()},
                           os.path.join(save_path,f"model_{epc}.pt"))

