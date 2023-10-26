# @author: NavinKumarMNK

import torch
import os
import sys
import pandas as pd
import logging

from torch.utils.data import Dataset

from typing import Dict, Tuple


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__),
        '../../')
    )
)

from src.utils import PROMPT_TEMPLATE, extract_anthropic_prompt
from datasets import load_dataset, concatenate_datasets

class DPODataLoader():
    def __init__(self, max_len:int, val_size: float,
                 tokenizer, num_proc:int, logger) -> None:
        super().__init__()
        self.max_len = max_len
        self.val_size = val_size
        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.logger: logging.Logger  = logger
    
    def get_datasets(self, path:str) -> Tuple[Dataset, Dataset]: 
        dataset = self._load_datasets(path)
        dataset =  dataset.train_test_split(
            test_size=self.val_size
        )
        return (dataset['train'], dataset['test'])
        
    def process_tokenize(self, data_point):
        model_inputs = {
            "input_ids": [],
            "label_ids": [],
        }
        columns = list(data_point.keys())
        
        for index in range(len(data_point[columns[0]])):
            if 'instruction' in columns and 'input' in columns and 'output' in columns:
                (instruction, input, output) = (
                    data_point['instruction'][index],
                    data_point['input'][index],
                    data_point['output'][index],
                )
                
                if input is not None and input != '':
                    instruction = instruction + '\n' + input
                
                assert len(output) != 2
                prompt, chosen, rejected = instruction, output[0], output[1]
            
            elif 'prompt' in columns and 'chosen' in columns and 'rejected' in columns:
                (prompt, chosen, rejected) = (
                    data_point['prompt'][index],
                    data_point['chosen'][index],
                    data_point['rejected'][index],
                )
                        
            elif 'chosen' in columns and 'rejected' in columns:
                
                prompt, chosen = extract_anthropic_prompt(data_point['chosen'][index])
                prompt, rejected = extract_anthropic_prompt(data_point['rejected'][index])

            else:
                raise ValueError("Columns must contain prompt, chosen and rejected")
    
            source = PROMPT_TEMPLATE.format_map({'instruction': prompt})
            source_ids = self.tokenizer.encode(text=source, add_special_tokens=False)
            accepts_ids = self.tokenizer.encode(text=chosen, add_special_tokens=False)
            rejects_ids = self.tokenizer.encode(text=rejected, add_special_tokens=False)
            
            if len(source_ids) > self.max_len - 1:
                source_ids = source_ids[:self.max_len - 1]
            if len(accepts_ids) > self.max_len - 1:
                accepts_ids = accepts_ids[:self.max_len - 1]
            if len(rejects_ids) > self.max_len - 1:
                rejects_ids = rejects_ids[:self.max_len - 1]
                
            
            source_accepts_ids = source_ids + [self.tokenizer.bos_token_id] + accepts_ids + [self.tokenizer.eos_token_id]
            source_accepts_labels = [-100] * len(source_ids) + [self.tokenizer.bos_token_id] + accepts_ids + [self.tokenizer.eos_token_id]
            source_rejects_ids = source_ids + [self.tokenizer.bos_token_id] + rejects_ids + [self.tokenizer.eos_token_id]
            source_rejects_labels = [-100] * len(source_ids) + [self.tokenizer.bos_token_id] + rejects_ids + [self.tokenizer.eos_token_id]


            source_accepts_length, source_rejects_length = len(source_accepts_ids), len(source_rejects_ids)
            max_length = max(source_accepts_length, source_rejects_length)
            
            source_accepts_ids = source_accepts_ids + [self.tokenizer.pad_token_id] * (max_length - source_accepts_length)
            source_accepts_labels = source_accepts_labels + [-100] * (max_length - source_accepts_length)
            source_rejects_ids = source_rejects_ids + [self.tokenizer.pad_token_id] * (max_length - source_rejects_length)
            source_rejects_labels = source_rejects_labels + [-100] * (max_length - source_rejects_length)
            
            inputs_ids = source_accepts_ids + source_rejects_ids
            labels = source_accepts_labels + source_rejects_labels 
            
            model_inputs["input_ids"].append(inputs_ids)
            model_inputs["label_ids"].append(labels)

        return model_inputs

    def _load_datasets(self, path):
        datasets = []
        self.logger.info(path)
        files = "Loaded file {}"
        for file in os.listdir(path):
            if file.endswith('.json'):
                datasets.append(
                    load_dataset(
                        'json',
                        data_files=os.path.join(path, file),    
                    )['train']
                )
                self.logger.info(files.format(file))
            elif file.endswith('.csv'):
                datasets.append(
                    load_dataset(
                        'csv',
                        data_files=os.path.join(path, file),
                    )['train']
                )
                self.logger.info(files.format(file))
            elif file.endswith('.parquet'):
                datasets.append(
                    load_dataset(
                        'parquet',
                        data_files=os.path.join(path, file),    
                    )['train']
                )
                self.logger.info(files.format(file))
            else:
                continue

            columns = list(datasets[-1].features)

            datasets[-1] = datasets[-1].map(
                self.process_tokenize,  
                batched=True,
                num_proc=self.num_proc,
                remove_columns=columns,
                load_from_cache_file=True,
            )
        
        if len(datasets) > 1:
            datasets = concatenate_datasets(datasets)
        else:
            datasets = datasets[0]
        
        self.logger.info(datasets)
        return datasets
    
if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    sft_dataloader = DPODataLoader(
        max_len=512,
        val_size=0.2,
        tokenizer=AutoTokenizer.from_pretrained('/workspace/LLM/models/mistral-7b/'),
        num_proc=4,
        logger=logger
    )
    
    train_dataset, val_dataset = sft_dataloader.get_datasets(path='/workspace/LLM/data/hh-rlhf/')
    print(train_dataset, val_dataset)
    print(train_dataset[0])
    
