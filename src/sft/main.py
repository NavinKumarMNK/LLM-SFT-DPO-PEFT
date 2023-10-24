import torch
import lightning.pytorch as pl
import bitsandbytes as bnb
import transformers

import os
import logging
import sys
import warnings
    
from typing import Dict, List, Tuple, Union, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from sft.dataloader import SFTDataLoader

class SupervisedFineTuning():
    def __init__(self, config: Dict) -> None:
        transformers.set_seed(config['seed'])
        self._load_model(
            path=config['model']['path'],
            config=config['model']['config'],
        )
        self._load_dataset(
            path=config['dataset']['path'],
            config=config['dataset']['config'],
        )
        
    def _load_model(self, path:Dict, config:Dict) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(self.model, self.tokenizer)
    
    def _load_dataset(self, path:str, config:Dict) -> None:
        sft_dataloader = SFTDataLoader(
            max_len=config['max_len'],
            val_size=config['val_size'],
            tokenizer=self.tokenizer,
        )
        self.train_dataset, self.val_dataset = sft_dataloader.get_datasets(path=path)
        print(self.train_dataset, self.val_dataset)
    
'''
path = '/workspace/LLM/models/mistral-7b/'
tokenizer = AutoTokenizer.from_pretrained(path)
'''
