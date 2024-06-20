import os, sys
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

from opensora.test.init_env import init_env
from opensora.models.diffusion.latte.modules import MultiHeadAttention

from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info


def run_mha_sp(norm_hidden_states, init_ckpt=None):

    # 0. split input
    # (f, bs * h*w, N) -> (f // sp, bs * h*w, N)
    norm_hidden_states = ops.chunk(norm_hidden_states, 2, axis=0)[hccl_info.rank%hccl_info.world_size]

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
    param_dict = ms.load_checkpoint(init_ckpt)
    ms.load_param_into_net(attn1, param_dict)
    print(f"attn1 sp layer load checkpoint from `{init_ckpt}` success")

    # 2. load input
    norm_hidden_states = norm_hidden_states
    attention_mask = None
    position_q = None
    position_k = None

    cross_attention_kwargs = {}
    frame = int(9)

    # 3. run
    atten_out = attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=attention_mask,
        position_q=position_q,
        position_k=position_k,
        last_shape=(frame,),
        **cross_attention_kwargs,
    )

    print(f"input_sp norm_hidden_states.shape: {norm_hidden_states.shape}")
    print(f"atten_out_sp.shape: {atten_out.shape}")
    print(f"atten_out_sp.mean: {atten_out.mean()}")
    print(f"atten_out_sp.min: {atten_out.min()}")
    print(f"atten_out_sp.max: {atten_out.max()}")

    return atten_out


def run_mha_nosp(norm_hidden_states, init_ckpt=None):

    # 0. permute input
    # (f, bs * h*w, N) -> (bs * h*w, f, N)
    norm_hidden_states = ops.permute(norm_hidden_states, (1, 0, 2))

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
        layout="BSH",
    )
    param_dict = ms.load_checkpoint(init_ckpt)
    ms.load_param_into_net(attn1, param_dict)
    print(f"attn1 layer load checkpoint from `{init_ckpt}` success")

    # 2. load input
    norm_hidden_states = norm_hidden_states
    attention_mask = None
    position_q = None
    position_k = None

    cross_attention_kwargs = {}
    frame = int(18)

    # 3. run
    atten_out = attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=attention_mask,
        position_q=position_q,
        position_k=position_k,
        last_shape=(frame,),
        **cross_attention_kwargs,
    )
    atten_out = ops.permute(atten_out, (1, 0, 2))

    print(f"input norm_hidden_states.shape: {norm_hidden_states.shape}")
    print(f"atten_out.shape: {atten_out.shape}")
    print(f"atten_out.mean: {atten_out.mean()}")
    print(f"atten_out.min: {atten_out.min()}")
    print(f"atten_out.max: {atten_out.max()}")

    return atten_out


if __name__ == '__main__':
    # 0. init env
    init_env(sp_size=2)

    # (f // sp, bs * h*w, N) -> (f, bs * h*w, N)
    norm_hidden_states = np.load("dump_data/step00/2_tem_b_MHA1_0_norm_hidden_states.npy")
    norm_hidden_states = np.concatenate(
        (norm_hidden_states[:, ...], norm_hidden_states[:, ...] * 0.3), axis=0
    )
    norm_hidden_states = Tensor(norm_hidden_states)

    init_ckpt = "./mha_random_init.ckpt"

    print("\n============== run sp ==============")
    atten_out_sp = run_mha_sp(norm_hidden_states, init_ckpt)
    print("====================================")

    print("\n============== run no sp ==============")
    atten_out = run_mha_nosp(norm_hidden_states, init_ckpt).chunk(2, axis=0)[hccl_info.rank%hccl_info.world_size]
    print("=======================================")

    atten_out_sp, atten_out = atten_out_sp.asnumpy(), atten_out.asnumpy()
    diff_abs = np.abs(atten_out_sp - atten_out).mean()
    diff_rel = (np.abs(atten_out_sp - atten_out) / np.abs(atten_out)).mean()
    diff_rel_eps = (np.abs(atten_out_sp - atten_out) / (np.abs(atten_out) + np.abs(atten_out.mean()))).mean()

    print("\n============== diff ==============")
    print(f"diff_abs: {diff_abs}")
    print(f"diff_rel: {diff_rel * 100:.2f}%")
    print(f"diff_rel_eps: {diff_rel_eps * 100:.2f}%")
    print("==================================")
