import argparse
import ast
import os
import time

from transformers import AutoTokenizer

import mindspore as ms

from mindone.transformers.mindspore_adapter import auto_mixed_precision
from mindone.transformers.models.llama import LlamaForCausalLM


def run_llama3_generate(args):
    print("=====> test_llama3_generate:")
    print("=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path, use_flash_attention_2=args.use_fa)

    model = model.to(ms.float16)
    model = auto_mixed_precision(model, amp_level="O2", dtype=ms.float16)

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

        prompt = [
            prompt,
        ]
        input_ids = ms.Tensor(tokenizer(prompt).input_ids, ms.int32)

        input_kwargs = {}
        if args.use_embed_input:
            input_kwargs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        else:
            input_kwargs["input_ids"] = input_ids

        output_ids = model.generate(**input_kwargs, use_cache=args.use_cache, max_new_tokens=30, do_sample=False, enable_dynamic_shape=args.enable_dynamic_shape)
        output_ids = output_ids.asnumpy()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=True)
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--use_embed_input", type=ast.literal_eval, default=False)
    parser.add_argument("--enable_dynamic_shape", type=ast.literal_eval, default=True)
    parser.add_argument("--prompt", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.ms_mode == ms.GRAPH_MODE:
        # if os.environ.get("MS_DEV_RUNTIME_CONF") is None:
        #     os.environ["MS_DEV_RUNTIME_CONF"] = "synchronize:True"
        #     print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")
        # else:
        #     if "synchronize:True" not in os.environ.get("MS_DEV_RUNTIME_CONF"):
        #         _old = os.environ.get("MS_DEV_RUNTIME_CONF")
        #         _old.replace("synchronize:False,", "")
        #         _old.replace(",synchronize:False", "")
        #         _old.replace("synchronize:False", "")
        #         _new = "synchronize:True," + _old if len(_old) > 0 else "synchronize:True"
        #         os.environ["MS_DEV_RUNTIME_CONF"] = _new
        #         print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")

        # ms.set_context(
        #     mode=ms.GRAPH_MODE,
        #     device_target="Ascend",
        #     jit_config={"jit_level": args.jit_level},
        #     max_device_memory="59GB",
        #     deterministic="ON",
        # )

        ms.set_context(
            mode=ms.GRAPH_MODE,
            device_target="Ascend",
            jit_config={"jit_level": args.jit_level},
            max_device_memory="59GB",
        )

    elif args.ms_mode == ms.PYNATIVE_MODE:
        ms.set_context(
            mode=ms.PYNATIVE_MODE,
            device_target="Ascend",
            pynative_synchronize=True,
            max_device_memory="59GB",
            deterministic="ON",
        )

    else:
        raise ValueError

    run_llama3_generate(args)
