
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor

from opensora.test.init_env import init_env
from opensora.models.diffusion.latte.modules import BasicTransformerBlock_


if __name__ == '__main__':
    # 0. init env
    init_env(sp_size=2)

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
        FA_dtype=ms.bfloat16
    )

    # 2. load input
    hidden_states = Tensor(np.load("dump_data/step00/0_tem_b_0_hidden_states.npy"))
    timestep = Tensor(np.load("dump_data/step00/1_tem_b_4_timestep.npy"))
    attention_mask = None
    encoder_hidden_states = None
    encoder_attention_mask = None
    class_labels = None
    position_q = None
    position_k = None

    cross_attention_kwargs = None
    frame = 9

    # 3. run
    out = temporal_block(
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
    )

    print(f"out.shape: {out.shape}")
    print(f"out.mean: {out.mean()}")
    print(f"out.min: {out.min()}")
    print(f"out.max: {out.max()}")


