import ast
import time
import argparse
from functools import partial

from transformers import AutoTokenizer

import mindspore
from mindspore import mint, JitConfig
from mindspore.communication import GlobalComm

from mindone.transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
from mindone.trainers.zero import prepare_network


def generate(args):
    # load model
    s_time = time.time()
    model = Qwen2MoeForCausalLM.from_pretrained(
        args.model_name,
        mindspore_dtype=mindspore.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    if args.zero3:
        shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        model = shard_fn(model)

    if args.ms_mode == mindspore.GRAPH_MODE:
        jitconfig = JitConfig(jit_level="O0", infer_boost="on")
        model.set_jit_config(jitconfig)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # info
    print("*" * 100)
    print(
        f"Using {config._attn_implementation}, use_cache {config.use_cache}, "
        f"dtype {config.mindspore_dtype}, layer {config.num_hidden_layers}, "
        f"Run with {'ZeRO3' if args.zero3 else 'Native'}"
    )
    print(f"Successfully loaded Qwen3ForCausalLMtime, cost: {(time.time()-s_time)/60:.2f} min")

    # prepare inputs
    input_ids = mindspore.Tensor(tokenizer([args.prompt], return_tensors="np").input_ids, mindspore.int32)
    model_inputs = {}
    model_inputs["input_ids"] = input_ids

    # generate
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=50,
        do_sample=False,
        use_cache=False,
    )

    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qwen3 demo.")

    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--prompt", type=str, default="the secret to baking a really good cake is")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Path to the pre-trained model.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["paged_attention", "flash_attention_2", "eager"],
    )
    parser.add_argument("--zero3", type=ast.literal_eval, default=True)

    # Parse the arguments
    args = parser.parse_args()

    if args.zero3:
        mint.distributed.init_process_group(backend="hccl")
        mindspore.set_auto_parallel_context(parallel_mode="data_parallel")

    if args.ms_mode == mindspore.GRAPH_MODE:
        mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.STRICT)
    elif args.ms_mode == mindspore.PYNATIVE_MODE:
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    else:
        raise ValueError
    
    generate(args)
