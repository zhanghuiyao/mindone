import os, sys
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

from opensora.test.init_env import init_env
from opensora.models.diffusion.latte.modules import MultiHeadAttention

from opensora.acceleration.parallel_states import get_sequence_parallel_state


if __name__ == '__main__':
    # 0. init env
    init_env(sp_size=2)

    # 1. init block
    attn1 = MultiHeadAttention(
        query_dim=1152,
        heads=16,
        dim_head=72,
        dropout=0.,
        bias=True,
        cross_attention_dim=None,
        upcast_attention=False,
        enable_flash_attention=True,
        use_rope=False,
        rope_scaling=None,
        compress_kv_factor=None,
        FA_dtype=ms.bfloat16,
        layout="SBH" if get_sequence_parallel_state() else "BSH",
    )

    # 2. load input
    norm_hidden_states = Tensor(np.load("dump_data/step00/2_tem_b_MHA1_0_norm_hidden_states.npy"))
    attention_mask = None
    position_q = None
    position_k = None

    cross_attention_kwargs = {}
    frame = int(9)

    # 3. run
    out = attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=attention_mask,
        position_q=position_q,
        position_k=position_k,
        last_shape=frame,
        **cross_attention_kwargs,
    )

    print(f"out.shape: {out.shape}")
    print(f"out.mean: {out.mean()}")
    print(f"out.min: {out.min()}")
    print(f"out.max: {out.max()}")

