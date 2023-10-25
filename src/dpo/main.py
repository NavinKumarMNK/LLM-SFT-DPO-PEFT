# @author: NavinKumarMNK

import torch
import lightning.pytorch as pl
import bitsandbytes as bnb
import transformers

from typing import Dict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from trl import DPOTrainer

try:
    from dataloader import DPODataLoader
except Exception as e:
    from .dataloader import DPODataLoader


class DirectPreferenceOptimization():
    def __init__(self, config:Dict, logger) -> None:
        pass

    def _load_model(self, path:Dict, config:Dict) -> None:
        pass

    def _load_dataset(self, path:Dict, config:Dict) -> None:
        pass

    def train(self):
        pass

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    # Load the yaml file
    import yaml
    with open('/workspace/LLM/config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
        config_dict = config_dict['dpo']
    
    dpo = DirectPreferenceOptimization(
        logger=logger,
        config=config_dict
    )
    dpo.train()