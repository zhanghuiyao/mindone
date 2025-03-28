import argparse
import ast
import os
import time

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import JitConfig

from mindone.transformers.mindspore_adapter import auto_mixed_precision
from mindone.transformers.models.qwen2 import Qwen2ForCausalLM


def run_qwen2_generate(args):
    print("=====> test_qwen2_generate:")
    print("=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, mindspore_dtype=ms.bfloat16, use_flash_attention_2=args.use_fa)

    if args.ms_mode == 0:
        if args.infer_boost:
            # Bug when enable dynamic shape on MindSpore 2.5.0
            jit_config = JitConfig(jit_level="O0", infer_boost='on')
        else:
            jit_config = JitConfig(jit_level="O0")
        model.set_jit_config(jit_config)

    print("=====> Building model done.")

    is_first = True
    while True:
        if args.prompt is not None and is_first:
            prompt = args.prompt
        else:
            prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")
        is_first = False

        if prompt == "q":
            print("Generate task done, see you next time!")
            break

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = ms.Tensor(tokenizer([text], return_tensors="np").input_ids, ms.int32)

        model_inputs = {}
        model_inputs["input_ids"] = input_ids

        output_ids = model.generate(
            **model_inputs,
            use_cache=args.use_cache,
            max_new_tokens=10,
            do_sample=False,
            enable_dynamic_shape=args.enable_dynamic_shape
        )
        output_ids = output_ids.asnumpy()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=True)
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--enable_dynamic_shape", type=ast.literal_eval, default=True)
    parser.add_argument("--infer_boost", type=ast.literal_eval, default=False)
    parser.add_argument("--prompt", type=str, default=None)
    args, _ = parser.parse_known_args()

    ms.set_context(mode=args.ms_mode)

    run_qwen2_generate(args)
