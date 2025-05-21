# Qwen3

# Introduction
Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and MoE models.  
The advancements of Qwen3 series are as follows:  
1. support seamless switching between thinking mode and non-thinking mode
2. significantly enhancement in its reasoning ability compared with previous QwQ and Qwen 2.5 Model
3. superior human prefrence alignment, excelling in creative-writing, role-playing, multi-turn dialogues and etc
3. expertise in agent capabilities
4. Support multiple languages

Model structure is also evolving. RMSNorm for Q and K have been added in attention layer to reduce variance.
Besides that, Qwen3 apply normalization in each head. Finally, shared postion embeddings have been applied.

# Get Started

## Requirements:
|mindspore | 	ascend driver | firmware       | cann tookit/kernel|
|--- |----------------|----------------| --- |
|2.5.0 | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.0.RC3.beta1|

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/qwen3
```

## Running

For convienience, you can use the following command:

```bash
python generate.py \
    --model_name "Qwen/Qwen3-30B-A3B" \
    --ms_mode 0
```

## Inference Speed
|model name	| precision* | cards | page attn |	tokens/s	|
| :---: | :---:  |:---:  | :---:  |:---:  |
| qwen3-0.6B-base |  bf16 | 1 | ✅  | 20.33 |
| qwen3-0.6B-0424 |  bf16 | 1 | ✅  | 21.13 |
| qwen3-1.7B-base |  bf16 | 1 | ✅  | 21.23 |
| qwen3-0.6B-0424 |  bf16 | 1 | ✅  | 22.85 |
| qwen3-4B-base |  bf16 | 1 | ✅  | 22.24 |
| qwen3-4B-0426 |  bf16 | 1 | ✅  | 19.92 |
| qwen3-8B-base |  bf16 | 1 | ✅  | 19.48 |
| qwen3-8B-0424 |  bf16 | 1 | ✅  | 19.12 |
