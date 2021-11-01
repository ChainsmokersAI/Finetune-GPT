import argparse
import pandas as pd

import torch

from utils import parse_model, parse_strategy, load_tokenizer

parser=argparse.ArgumentParser(description="Generation..")
parser.add_argument('--model', type=parse_model, required=True, help="fine-tuned Korean GPT: kogpt2")
parser.add_argument('--path', type=str, required=True, help="path of fine-tuned GPT")
parser.add_argument('--max_len', type=int, default=50, help="max length of generated text: default 50")
parser.add_argument('--strategy', type=parse_strategy, default='greedy', help="decoding strategy: greedy or sampling or top-k or top-k")
parser.add_argument('--temperature', type=float, default=0.7, help="temperature: default 0.7")
parser.add_argument('--k', type=int, default=30, help="K of top-k: default 30")
parser.add_argument('--p', type=float, default=0.9, help="P of top-p: default 0.9")
parser.add_argument('--n_sample', type=int, default=10, help="number of samples: default 10")
args=parser.parse_args()

def generate(model, tokenizer, prompt):
    input_ids=tokenizer.encode(prompt)

    if args.strategy=='greedy':
        output=model.generate(torch.tensor([input_ids]), max_length=args.max_len)
    elif args.strategy=='sampling':
        output=model.generate(torch.tensor([input_ids]), max_length=args.max_len, do_sample=True, top_k=0, temperature=args.temperature)
    elif args.strategy=='top-k':
        output=model.generate(torch.tensor([input_ids]), max_length=args.max_len, do_sample=True, top_k=args.k)
    elif args.strategy=='top-p':
        output=model.generate(torch.tensor([input_ids]), max_length=args.max_len, do_sample=True, top_k=0, top_p=args.p)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]

def main():
    # pre-trained tokenizer
    tokenizer=load_tokenizer(args.model)
    # fine-tuned GPT
    model=torch.load(args.path)
    
    # load prompts (for generation)
    df=pd.read_excel('./data/prompts.xlsx')

    prompts=[]
    outputs=[]
    
    for text in df['text']:
        if args.strategy=='greedy':
            prompts.append(text)
            outputs.append(generate(model, tokenizer, prompt=text))
            continue

        for i in range(args.n_sample):
            prompts.append(text)
            outputs.append(generate(model, tokenizer, prompt=text))
    
    pd.DataFrame({
        'id': list(range(len(outputs))),
        'prompt': prompts,
        'generated': outputs
    }).to_excel('./data/generated.xlsx', index=False)


if __name__=="__main__":
    main()
