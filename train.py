import argparse
import pandas as pd
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AdamW

from utils import parse_model, parse_bool, load_model

parser=argparse.ArgumentParser(description="Train Korean GPT..")
parser.add_argument('--model', type=parse_model, required=True, help="pre-trained Korean GPT: kogpt2")
parser.add_argument('--gpu', type=parse_bool, default=True, help="train with GPU? True or False")
parser.add_argument('--ddp', type=parse_bool, default=False, help="use multiple GPUs? True or False")
parser.add_argument('--len_context', type=int, default=512, help="length of context: default 512")
parser.add_argument('--batch_size', type=int, default=2, help="size of batch: default 2")
parser.add_argument('--accumulation_steps', type=int, default=4, help="accumulation steps: default 4")
parser.add_argument('--lr', type=float, default=2e-5, help="learning rate: default 2e-5")
parser.add_argument('--epochs', type=int, default=30, help="epochs: default 30")
args=parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, len_context):
        self.data=[]
        
        data=[]
        
        df=pd.read_excel(path)
        for text in df['text']:
            _data=tokenizer.encode("</s>"+text)
            
            if len(data)+len(_data)<len_context-1:
                data.extend(_data)
                continue
                
            # append EOS token
            data+=[1]
            # padding
            data.extend([3]*(len_context-len(data)))

            self.data.append(data)
            data=_data
            
        print(len(self.data), "data")
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
    
    def __len__(self):
        return len(self.data)

def train(device):
    # load pre-trained tokenizer, model
    tokenizer, model=load_model(args.model)

    # load dataset
    dataset=CustomDataset(path='./data/train_data.xlsx', tokenizer=tokenizer, len_context=args.len_context)
    dataloader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # optimizer
    optimizer=AdamW(model.parameters(), lr=args.lr)

    model.to(device)
    model.train()

    # fine-tuning
    for epoch in range(args.epochs):
        loss_accum=0
        optimizer.zero_grad()
        for step, data in enumerate(dataloader):
            data=data.to(device)
            
            outputs=model(data, labels=data)
            
            loss=outputs[0]/args.accumulation_steps
            loss.backward()
            
            loss_accum+=loss.item()
            
            if (step+1)%args.accumulation_steps==0:
                print(f'epoch {epoch+1} step {(step+1)/args.accumulation_steps} loss {loss_accum:.4f}')
                loss_accum=0
                
                optimizer.step()
                optimizer.zero_grad()

    model.eval()
    model.to(torch.device('cpu'))

    # save fine-tuned model
    torch.save(model, './models/'+args.model+'_finetuned_'+str(len(os.listdir('./models/')))+'.pt')

def train_ddp(rank, world_size):
    # create default process group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8973', rank=rank, world_size=world_size)
    
    # load pre-trained tokenizer, model
    tokenizer, model=load_model(args.model)
    model=model.to(rank)
    model_ddp=DDP(model, device_ids=[rank])

    # load dataset
    dataset=CustomDataset(path='./data/train_data.xlsx', tokenizer=tokenizer, len_context=args.len_context)
    sampler=DistributedSampler(dataset)
    dataloader=DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)

    # optimizer
    optimizer=AdamW(model_ddp.parameters(), lr=args.lr)

    model_ddp.train()

    # fine-tuning
    for epoch in range(args.epochs):
        loss_accum=0
        optimizer.zero_grad()
        for step, data in enumerate(dataloader):
            data=data.to(rank)
            
            outputs=model_ddp(data, labels=data)
            
            loss=outputs[0]/args.accumulation_steps
            loss.backward()
            
            loss_accum+=loss.item()
            
            if (step+1)%args.accumulation_steps==0:
                print(f'rank {rank} epoch {epoch+1} step {(step+1)/args.accumulation_steps} loss {loss_accum:.4f}')
                loss_accum=0
                
                optimizer.step()
                optimizer.zero_grad()

    model_ddp.eval()

    # save fine-tuned model
    if rank==0:
        model_ddp.to(torch.device('cpu'))
        torch.save(model_ddp.module, './models/'+args.model+'_finetuned_'+str(len(os.listdir('./models/')))+'.pt')

def main():
    if args.gpu and torch.cuda.is_available():
        if args.ddp:
            # train with multiple GPUs
            world_size=torch.cuda.device_count()
            mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
        else:
            # train with single GPU
            device=torch.device('cuda:0')
            train(device)
    else:
        # train with CPU
        device=torch.device('cpu')
        train(device)
            

if __name__=="__main__":
    main()
