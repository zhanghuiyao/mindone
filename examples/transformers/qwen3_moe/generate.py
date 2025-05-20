import argparse

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import JitConfig

from mindone.transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def generate(args):
    # load model
    model = Qwen3MoeForCausalLM.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    if args.ms_mode == ms.GRAPH_MODE:
        jitconfig = JitConfig(jit_level="O0", infer_boost="on")
        model.set_jit_config(jitconfig)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # info
    print("*" * 100)
    print(
        f"Using {config._attn_implementation}, use_cache {config.use_cache},"
        f"dtype {config.mindspore_dtype}, layer {config.num_hidden_layers}"
    )
    print("Successfully loaded Qwen3ForCausalLM")

    # prepare inputs
    input_ids = ms.Tensor(tokenizer([args.prompt], return_tensors="np").input_ids, ms.int32)
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

    parser.add_argument("--ms_mode", type=int, default=1, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--prompt", type=str, default="the secret to baking a really good cake is")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Path to the pre-trained model.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["paged_attention", "flash_attention_2", "eager"],
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.ms_mode == ms.GRAPH_MODE:
        ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    elif args.ms_mode == ms.PYNATIVE_MODE:
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        raise ValueError
    
    generate(args)
