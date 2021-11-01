import argparse

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

def parse_model(val):
    if val not in ['kogpt2']:
        raise argparse.ArgumentTypeError("kogpt2 expected.")
    return val

def parse_bool(val):
    if val=="True": return True
    elif val=="False": return False
    else: raise argparse.ArgumentTypeError("True or False expected.")

def parse_strategy(val):
    if val not in ['greedy', 'sampling', 'top-k', 'top-p']:
        raise argparse.ArgumentTypeError("greedy or sampling or top-k or top-k expected.")
    return val

def load_model(model):
    if model=="kogpt2":
        tokenizer=PreTrainedTokenizerFast.from_pretrained(
            'skt/kogpt2-base-v2',
            bos_token='</s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )
        model=GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    return tokenizer, model

def load_tokenizer(model):
    if model=="kogpt2":
        tokenizer=PreTrainedTokenizerFast.from_pretrained(
            'skt/kogpt2-base-v2',
            bos_token='</s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )

    return tokenizer
