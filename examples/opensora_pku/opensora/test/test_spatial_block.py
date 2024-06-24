
import os, sys

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

from opensora.test.init_env import init_env
from opensora.models.diffusion.latte.modules import BasicTransformerBlock
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info


def run_spatial_block(*args, init_ckpt=""):
    # 1. init block
    spatial_block = BasicTransformerBlock(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        dropout=0.,
        cross_attention_dim=1152,
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
    )

    param_dict = ms.load_checkpoint(init_ckpt)
    ms.load_param_into_net(spatial_block, param_dict)
    print(f"temporal block load checkpoint from `{init_ckpt}` success")

    out = spatial_block(*args)

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

    # zhy_test 1
    if hccl_info.rank == 0:
        _hidden_states = np.load("1_hs_after_sb_0_sp0.npy")
    elif hccl_info.rank == 1:
        _hidden_states = np.load("1_hs_after_sb_0_sp1.npy")
    _hidden_states = _hidden_states.reshape((2, 9, 1024, 1152)).transpose(1, 0, 2, 3).reshape((9, 2048, 1152))
    full_hidden_states = np.load("1_hs_after_sb_0.npy")
    full_hidden_states = full_hidden_states.reshape((2, 17, 1024, 1152)).transpose(0, 2, 1, 3).reshape((2048, 17, 1152))
    # _hidden_states = np.load("_bak_test_blocks/dump_data/step00/0_tem_b_0_hidden_states.npy") # (f // sp, b, N) ~ (9, 2048, 1152)
    # full_hidden_states = np.concatenate((_hidden_states[:], _hidden_states[:] * 0.3), axis=0)  # (f, b, N)
    # if hccl_info.rank == 1:
    #     _hidden_states = _hidden_states[:] * 0.3

    timestep_b6N = np.load("_bak_test_blocks/dump_data/step00/1_tem_b_4_timestep.npy")  # (6, b, N) ~ (6, 2048, 1152)
    attention_mask = None
    encoder_hidden_states = None
    encoder_attention_mask = None
    class_labels = None
    position_q = None
    position_k = None
    cross_attention_kwargs = None
    frame = 9
    init_ckpt = "_bak_test_blocks/temporal_block_random_init.ckpt"

    print("\n============== run sp ==============")
    hidden_states = Tensor(_hidden_states)

    # zhy_test
    if hccl_info.rank == 0:
        timestep = Tensor(np.load("3_temp_before_tb_0_sp0.npy"))
    else:
        timestep = Tensor(np.load("3_temp_before_tb_0_sp1.npy"))
    # timestep = Tensor(timestep_b6N)

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
    # zhy_test 2
    if hccl_info.rank == 1:
        out_sp = out_sp[:8]

    print("====================================")

    print("\n============== run no sp ==============")

    hidden_states = Tensor(full_hidden_states.transpose((1, 0, 2)))

    # zhy_test
    # timestep = Tensor(timestep_b6N.transpose((1, 0, 2)))
    timestep = Tensor(np.load("3_temp_before_tb_0.npy"))

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

    # zhy_test 3
    # out_no_sp = out_no_sp.transpose(1, 0, 2).chunk(2, axis=0)[hccl_info.rank%hccl_info.world_size]
    if hccl_info.rank == 0:
        out_no_sp = out_no_sp.transpose(1, 0, 2)[:9]
    else:
        out_no_sp = out_no_sp.transpose(1, 0, 2)[9:]

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
