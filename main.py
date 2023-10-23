import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, default="world")
    
    config = args.parse_args()
    print(config)
    
    
    