import torch
import lightning.pytorch as pl
import bitsandbytes as bnb
import transformers

from typing import Dict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

from dataloader import SFTDataLoader

class SupervisedFineTuning():
    def __init__(self, config: Dict, logger) -> None:
        transformers.set_seed(config['seed'])
        self.logger = logger
             
        self.tokenizer, self.model = None, None
        self._load_model(
            path=config['model']['path'],
            config=config['model']['params'],
        )
        
        self.train_dataset, self.val_dataset = None, None
        self._load_dataset(
            path=config['data']['path'],
            config=config['data']['params'],
        )    
            
    def _load_model(self, path:Dict, config:Dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        special_tokens_dict = {
            'sep_token': '<SEP>',
            'pad_token': '<PAD>',
            'cls_token': '<CLS>',
            'mask_token': '<MASK>'
        }

        tokenizer.add_special_tokens(special_tokens_dict)
        
        
        conf_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": (config['torch_dtype'] if config['torch_dtype'] in ["auto", None]
                else getattr(torch, config['torch_dtype']))[0],
            "low_cpu_mem_usage": True
        }
        
        if config['quantization'] == '16-bit':
            self.logger.info("Loading in 16-bit")
            pass
        elif config['quantization'] == '8-bit':
            self.logger.info("Loading in 8-bit")
            conf_kwargs["load_in_8bit"] = True
            conf_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                **config['quantization_config']
            )            
        elif config['quantization'] == '4-bit':
            self.logger.info("Loading in 4-bit")
            conf_kwargs["load_in_4bit"] = True
            conf_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                **config['quantization_config']
            )
        else:
            raise ValueError("Quantization must be 16-bit, 8-bit or 4-bit")
                
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            safe_load_=True,
            from_tf=bool(".ckpt" in path),
            **conf_kwargs    
        )
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.use_cache = True
        
        if 'peft_config' in config:
            if config['peft_config']['model_path'] == False:
                self.logger.info("New PEFT model")    
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=config['peft_config']['target_modules'],
                    r=config['peft_config']['r'],
                    lora_alpha=config['peft_config']['alpha'],
                    lora_dropout=config['peft_config']['dropout'],
                    modules_to_save=config['peft_config']['modules_to_save'],
                )
                model = get_peft_model(self.model, lora_config)
            else:
                self.logger.info(f"Loading PEFT model {config['peft_config']['model']}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    path=config['peft_config']['model_path'],
                    is_trainable=True,
                )
            
        model.print_trainable_parameters()
        
    def _load_dataset(self, path:str, config:Dict) -> None:
        sft_dataloader = SFTDataLoader(
            max_len=config['max_len'],
            val_size=config['val_size'],
            num_proc=config['num_proc'],
            tokenizer=self.tokenizer,
        )
        self.train_dataset, self.val_dataset = sft_dataloader.get_datasets(path=path)
        print(self.train_dataset, self.val_dataset)
    
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
        config_dict = config_dict['sft']
    
    sft = SupervisedFineTuning(
        logger=logger,
        config=config_dict
    )
    
    print(sft.model. sft.tokenizer)
    sft.train()