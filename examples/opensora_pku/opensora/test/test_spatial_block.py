
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

    base_dir = "./_bak_test_blocks/dump_data_sb/"

    # zhy_test 1
    hidden_states_sp = np.load(base_dir + f"1_hidden_states_before_sb_0_sp{hccl_info.rank}.npy")
    attention_mask_sp = np.load(base_dir + f"2_attention_mask_before_sb_0_sp{hccl_info.rank}.npy")
    encoder_hs_sp = np.load(base_dir + f"3_encoder_hidden_states_spatial_before_sb_0_sp{hccl_info.rank}.npy")
    encoder_attention_mask_sp = np.load(base_dir + f"4_encoder_attention_mask_before_sb_0_sp{hccl_info.rank}.npy")
    timestep_sp = np.load(base_dir + f"5_timestep_spatial_before_sb_0_sp{hccl_info.rank}.npy")

    full_hidden_states = np.load(base_dir + f"1_hidden_states_before_sb_0.npy")
    full_attention_mask = np.load(base_dir + f"2_attention_mask_before_sb_0.npy")
    full_encoder_hs = np.load(base_dir + f"3_encoder_hidden_states_spatial_before_sb_0.npy")
    full_encoder_attention_mask = np.load(base_dir + f"4_encoder_attention_mask_before_sb_0.npy")
    full_timestep = np.load(base_dir + f"5_timestep_spatial_before_sb_0.npy")

    class_labels = None
    position_q = None
    position_k = None
    cross_attention_kwargs = None
    frame = 9
    init_ckpt = "_bak_test_blocks/spatial_block_from_trained_weight.ckpt"

    print("\n============== run sp ==============")

    out_sp = run_spatial_block(
        Tensor(hidden_states_sp),
        Tensor(attention_mask_sp),  # attention_mask
        Tensor(encoder_hs_sp),  # encoder_hidden_states
        Tensor(encoder_attention_mask_sp),  # encoder_attention_mask
        Tensor(timestep_sp),
        cross_attention_kwargs,
        class_labels,
        position_q,
        position_k,
        (32, 32),
        init_ckpt=init_ckpt
    )

    print("====================================")

    print("\n============== run no sp ==============")

    out_no_sp = run_spatial_block(
        Tensor(full_hidden_states),
        Tensor(full_attention_mask),  # attention_mask
        Tensor(full_encoder_hs),  # encoder_hidden_states
        Tensor(full_encoder_attention_mask),  # encoder_attention_mask
        Tensor(full_timestep),
        cross_attention_kwargs,
        class_labels,
        position_q,
        position_k,
        (32, 32),
        init_ckpt=init_ckpt
    )

    print("=======================================")

    out_no_sp, out_sp = out_no_sp.asnumpy(), out_sp.asnumpy()
    out_no_sp, out_sp = out_no_sp.reshape((2, 17, -1)), out_sp.reshape((2, 9, -1))
    if hccl_info.rank == 0:
        out_no_sp, out_sp = out_no_sp[:, :9, :], out_sp[:, :9, :]
    elif hccl_info.rank == 1:
        out_no_sp, out_sp = out_no_sp[:, 9:, :], out_sp[:, :8, :]

    diff_abs = np.abs(out_no_sp - out_sp).mean()
    diff_rel = (np.abs(out_no_sp - out_sp) / np.abs(out_sp)).mean()
    diff_rel_eps = (np.abs(out_no_sp - out_sp) / (np.abs(out_sp) + np.abs(out_sp.mean()))).mean()

    print("\n============== diff ==============")
    print(f"diff_abs: {diff_abs}")
    print(f"diff_rel: {diff_rel * 100:.2f}%")
    print(f"diff_rel_eps: {diff_rel_eps * 100:.2f}%")
    print("==================================")
