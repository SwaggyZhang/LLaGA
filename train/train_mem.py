# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
sys.path.append(".")
sys.path.append("./utils")
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


# replace_llama_attn_with_flash_attn()  # noted by Xin: 仅适用于bf16或者fp16


from train import _train

if __name__ == "__main__":
    _train()
