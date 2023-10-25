import argparse
import yaml
import logging

from src.sft import SupervisedFineTuning
from src.dpo import DirectPreferenceOptimization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    sft_parsers = subparsers.add_parser('sft')
    sft_parsers.add_argument("--params", type=str, required=True, help="Path to yaml file")
    
    rlhf_parsers = subparsers.add_parser('rlhf')
    rlhf_parsers.add_argument("--params", type=str, required=True, help="Path to yaml file")
    
    dpo_parsers = subparsers.add_parser('dpo')
    dpo_parsers.add_argument("--params", type=str, required=True, help="Path to yaml file")
    
    chat_parsers = subparsers.add_parser('chat')
    chat_parsers.add_argument("--params", type=str, required=True, help="Path to yaml file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    # SFT Parser
    if args.command == 'sft':
        if args.params:
            with open(args.params, 'r') as f:
                config = yaml.safe_load(f)
            sft = SupervisedFineTuning(config['sft'], logger=logger)
            sft.train()
        else:
            print("Please provide a yaml file")
            exit(1)
    
    # DPO Parser
    if args.command == 'dpo':
        if args.params:
            with open(args.params, 'r') as f:
                config = yaml.safe_load(f)
            dpo = DirectPreferenceOptimization(config['dpo'], logger=logger)
        else:
            print("Please provide a yaml file")
            exit(1)
    '''
    # RLHF Parser
    if args.command == 'rlhf':
        if args.params:
            with open(args.params, 'r') as f:
                config = yaml.safe_load(f)
            rlhf = ReinforcementLearning(config)
        else:
            print("Please provide a yaml file")
            exit(0)
    
    
    # Chat Parser
    if args.command == 'chat':
        if args.params:
            with open(args.params, 'r') as f:
                config = yaml.safe_load(f)
            print(config)
        else:
            print("Please provide a yaml file")
            exit(0)
    '''