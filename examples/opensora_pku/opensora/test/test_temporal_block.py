
import os, sys

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

from opensora.test.init_env import init_env
from opensora.models.diffusion.latte.modules import BasicTransformerBlock_
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info


def run_tmp_block_sp(*args, init_ckpt=""):

    # 1. init block
    temporal_block = BasicTransformerBlock_(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        dropout=0.,
        cross_attention_dim=None,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        attention_bias=True,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
        norm_elementwise_affine=False,
        norm_type="ada_norm_single",
        norm_eps=1e-6,
        final_dropout=False,
        attention_type="default",
        positional_embeddings=None,
        num_positional_embeddings=None,
        enable_flash_attention=True,
        use_rope=False,
        rope_scaling=None,
        compress_kv_factor=None,
        FA_dtype=ms.bfloat16,
        layout="SBH"
    )

    param_dict = ms.load_checkpoint(init_ckpt)
    ms.load_param_into_net(temporal_block, param_dict)
    print(f"temporal block load checkpoint from `{init_ckpt}` success")

    out = temporal_block(*args)

    print(f"input hidden_states.shape: {args[0].shape}")
    print(f"input timestep.shape: {args[4].shape}")
    print(f"temporal_out.shape: {out.shape}")
    print(f"temporal_out.mean: {out.mean()}")
    print(f"temporal_out.min: {out.min()}")
    print(f"temporal_out.max: {out.max()}")

    return out


def run_tmp_block_no_sp(*args, init_ckpt=""):
    # 1. init block
    temporal_block = BasicTransformerBlock_(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        dropout=0.,
        cross_attention_dim=None,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        attention_bias=True,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
        norm_elementwise_affine=False,
        norm_type="ada_norm_single",
        norm_eps=1e-6,
        final_dropout=False,
        attention_type="default",
        positional_embeddings=None,
        num_positional_embeddings=None,
        enable_flash_attention=True,
        use_rope=False,
        rope_scaling=None,
        compress_kv_factor=None,
        FA_dtype=ms.bfloat16,
        layout="BSH"
    )

    param_dict = ms.load_checkpoint(init_ckpt)
    ms.load_param_into_net(temporal_block, param_dict)
    print(f"temporal block load checkpoint from `{init_ckpt}` success")

    out = temporal_block(*args)

    print(f"input hidden_states.shape: {args[0].shape}")
    print(f"input timestep.shape: {args[4].shape}")
    print(f"temporal_out.shape: {out.shape}")
    print(f"temporal_out.mean: {out.mean()}")
    print(f"temporal_out.min: {out.min()}")
    print(f"temporal_out.max: {out.max()}")

    return out


if __name__ == '__main__':
    # 0. init env
    init_env(sp_size=2)

    # 2. load input
    _hidden_states = np.load("dump_data/step00/0_tem_b_0_hidden_states.npy")  # (f // sp, b, N)
    full_hidden_states = np.concatenate((_hidden_states[:], _hidden_states[:] * 0.3), axis=0)  # (f, b, N)
    timestep_b6N = np.load("dump_data/step00/1_tem_b_4_timestep.npy")
    attention_mask = None
    encoder_hidden_states = None
    encoder_attention_mask = None
    class_labels = None
    position_q = None
    position_k = None
    cross_attention_kwargs = None
    frame = 9
    init_ckpt = "temporal_block_random_init.ckpt"

    print("\n============== run sp ==============")
    hidden_states = Tensor(_hidden_states)
    timestep = Tensor(timestep_b6N)
    out_sp = run_tmp_block_sp(
        hidden_states,
        None,  # attention_mask
        None,  # encoder_hidden_states
        None,  # encoder_attention_mask
        timestep,
        cross_attention_kwargs,
        class_labels,
        position_q,
        position_k,
        (frame,),
        init_ckpt=init_ckpt
    )
    print("====================================")

    print("\n============== run no sp ==============")

    hidden_states = Tensor(full_hidden_states.transpose((1, 0, 2)))
    timestep = Tensor(timestep_b6N.transpose((1, 0, 2)))
    out_no_sp = run_tmp_block_no_sp(
        hidden_states,
        None,  # attention_mask
        None,  # encoder_hidden_states
        None,  # encoder_attention_mask
        timestep,
        cross_attention_kwargs,
        class_labels,
        position_q,
        position_k,
        (frame,),
        init_ckpt=init_ckpt
    )
    # (b, f, N)
    out_no_sp = out_no_sp.transpose(1, 0, 2).chunk(2, axis=0)[hccl_info.rank%hccl_info.world_size]
    print("=======================================")

    out_no_sp, out_sp = out_no_sp.asnumpy(), out_sp.asnumpy()
    diff_abs = np.abs(out_no_sp - out_sp).mean()
    diff_rel = (np.abs(out_no_sp - out_sp) / np.abs(out_sp)).mean()
    diff_rel_eps = (np.abs(out_no_sp - out_sp) / (np.abs(out_sp) + np.abs(out_sp.mean()))).mean()

    print("\n============== diff ==============")
    print(f"diff_abs: {diff_abs}")
    print(f"diff_rel: {diff_rel * 100:.2f}%")
    print(f"diff_rel_eps: {diff_rel_eps * 100:.2f}%")
    print("==================================")
