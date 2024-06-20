
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
    hidden_states = Tensor(np.load("xxx.npy"))
    attention_mask = Tensor(np.load("xxx.npy"))
    encoder_hidden_states = Tensor(np.load("xxx.npy"))
    encoder_attention_mask = Tensor(np.load("xxx.npy"))
    timestep = Tensor(np.load("xxx.npy"))
    class_labels = Tensor(np.load("xxx.npy"))
    position_q = Tensor(np.load("xxx.npy"))
    position_k = Tensor(np.load("xxx.npy"))

    cross_attention_kwargs = {xxx: xxx}
    frame = int(xxx)


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

