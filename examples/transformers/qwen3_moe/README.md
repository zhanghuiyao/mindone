# Qwen3-MoE

## Requirements:
|mindspore | 	ascend driver | firmware       | cann tookit/kernel|
|--- |----------------|----------------| --- |
|2.5.0 | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.0.RC3.beta1|

## Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

## Get Started

For convienience, you can use the following command:

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1
msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9001 --log_dir=outputs/parallel_logs \
python examples/transformers/qwen3_moe/generate.py \
  --model_name /PATH TO/Qwen3-30B-A3B \
  --ms_mode 0 \
  --zero3 True
```

## Inference Speed
|model name	| precision* | cards | parallelism | falsh attn | page attn |	tokens/s	|
| :---: | :---:  |:---:  | :---:  |:---:  | :---:  | :---:  |
| qwen3-30B-A3B |  bf16 | 2 | zero stage3 | ✖️  | ✖️ | testing |
