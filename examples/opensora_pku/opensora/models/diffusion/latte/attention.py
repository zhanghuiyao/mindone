from opensora.models.diffusion.latte.modules import *


from opensora.acceleration.communications import AllToAll_SBH
from opensora.acceleration.parallel_states import hccl_info


class MultiHeadAttention(nn.Cell):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to True):
            Set to `True` to upcast the softmax computation to `float32`.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        dtype=ms.float32,
        enable_flash_attention=False,
        use_rope: bool = False,
        rope_scaling: Optional[Dict] = None,
        compress_kv_factor: Optional[Tuple] = None,
        layout: Optional[str] = "BSH",
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.dropout = dropout
        self.heads = heads
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self.dtype = dtype
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.compress_kv_factor = compress_kv_factor
        self.only_cross_attention = only_cross_attention
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.layout = layout

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.added_kv_proj_dim = added_kv_proj_dim

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None."
                " Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if self.layout == "SBH":
            assert get_sequence_parallel_state()
            self.sp_size = hccl_info.world_size
            self.alltoall_sbh_q = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_k = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_v = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_out = AllToAll_SBH(scatter_dim=0, gather_dim=1)
        else:
            self.alltoall_sbh_q = None
            self.alltoall_sbh_k = None
            self.alltoall_sbh_v = None
            self.alltoall_sbh_out = None

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        assert not (
            self.use_rope and (self.compress_kv_factor is not None)
        ), "Can not both enable compressing kv and using rope"
        if self.compress_kv_factor is not None:
            self._init_compress()
        if self.use_rope:
            self._init_rope()

        self.to_q = nn.Dense(query_dim, self.inner_dim, has_bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
            self.to_v = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Dense(added_kv_proj_dim, self.inner_dim, has_bias=bias)
            self.add_v_proj = nn.Dense(added_kv_proj_dim, self.inner_dim, has_bias=bias)

        self.to_out = nn.SequentialCell(nn.Dense(self.inner_dim, query_dim, has_bias=out_bias), nn.Dropout(p=dropout))

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            self.flash_attention = MSFlashAttention(
                head_dim=dim_head, head_num=heads, fix_head_dims=[72], attention_dropout=attn_drop
            )
        else:
            self.attention = Attention(
                dim_head=dim_head, attn_drop=attn_drop, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax
            )

    def _init_compress(self):
        if len(self.compress_kv_factor) == 2:
            self.sr = nn.Conv2d(
                self.inner_dim,
                self.inner_dim,
                groups=self.inner_dim,
                kernel_size=self.compress_kv_factor,
                stride=self.compress_kv_factor,
            )
            weight = initializer("ones", self.sr.weight.shape) * (1 / self.compress_kv_factor[0] ** 2)
            self.sr.weight.set_data(weight)
        elif len(self.compress_kv_factor) == 1:
            self.kernel_size = self.compress_kv_factor[0]
            self.sr = nn.Conv1d(
                self.inner_dim,
                self.inner_dim,
                groups=self.inner_dim,
                kernel_size=self.compress_kv_factor[0],
                stride=self.compress_kv_factor[0],
            )
            weight = initializer("ones", self.sr.weight.shape) * (1 / self.compress_kv_factor[0])
            self.sr.weight.set_data(weight)
        bias = initializer("zeros", self.sr.bias.shape)
        self.sr.bias.set_data(bias)
        self.norm = LayerNorm(self.inner_dim)

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rope2d = RoPE2D()
            self.rope1d = RoPE1D()
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor_2d = self.rope_scaling["factor_2d"]
            scaling_factor_1d = self.rope_scaling["factor_1d"]
            if scaling_type == "linear":
                self.rope2d = LinearScalingRoPE2D(scaling_factor=scaling_factor_2d)
                self.rope1d = LinearScalingRoPE1D(scaling_factor=scaling_factor_1d)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> ms.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`ms.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `ms.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        assert (
            current_length == target_length
        ), "The attention mask length should be identical to encoder hidden states length"
        f", but got {current_length} and {current_length}"

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, 0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, 1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`ms.Tensor`): Hidden states of the encoder.

        Returns:
            `ms.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def apply_rope(self, query, key, value, position_q, position_k):
        assert self.use_rope, "use_rope must be True"
        head_dim = self.inner_dim // self.heads
        batch_size, seq_len, _ = query.shape
        # (b, n, h*d) -> (b, n, h, d) -> (b, h, n, d)
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        # require the shape of (batch_size x nheads x ntokens x dim)
        if position_q.ndim == 3:
            query = self.rope2d(query, position_q)
        elif position_q.ndim == 2:
            query = self.rope1d(query, position_q)
        else:
            raise NotImplementedError
        if position_k.ndim == 3:
            key = self.rope2d(key, position_k)
        elif position_k.ndim == 2:
            key = self.rope1d(key, position_k)
        else:
            raise NotImplementedError
        # change the to original shape
        # (b, h, n, d) -> (b, n, h, d) -> (b, n, h*d)
        query = query.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        key = key.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        value = value.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        return query, key, value

    def construct(
        self,
        hidden_states,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        scale: float = 1.0,
        position_q: Optional[ms.Tensor] = None,
        position_k: Optional[ms.Tensor] = None,
        last_shape: Tuple[int] = None,
    ):
        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        if self.compress_kv_factor is not None:
            batch_size = hidden_states.shape[0]
            if len(last_shape) == 2:
                encoder_hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, self.dim, *last_shape)
                encoder_hidden_states = (
                    self.sr(encoder_hidden_states).reshape(batch_size, self.dim, -1).permute(0, 2, 1)
                )
            elif len(last_shape) == 1:
                encoder_hidden_states = hidden_states.permute(0, 2, 1)
                if last_shape[0] % 2 == 1:
                    first_frame_pad = encoder_hidden_states[:, :, :1].repeat_interleave(self.kernel_size - 1, -1)
                    encoder_hidden_states = ops.concat((first_frame_pad, encoder_hidden_states), axis=2)
                encoder_hidden_states = self.sr(encoder_hidden_states).permute(0, 2, 1)
            else:
                raise NotImplementedError(f"NotImplementedError with last_shape {last_shape}")

            encoder_hidden_states = self.norm(encoder_hidden_states)

        if self.layout == "SBH":
            sequence_length, batch_size, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            sequence_length *= self.sp_size
        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

        if attention_mask is not None:
            out_dim = 4 if self.enable_flash_attention else 3
            attention_mask = self.prepare_attention_mask(
                attention_mask, sequence_length, batch_size, out_dim=out_dim
            )  # make attention mask a correct shape

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        x_dtype = hidden_states.dtype
        h = self.heads
        head_dim = self.inner_dim // self.heads
        mask = attention_mask

        q = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Layout BSH or SBH:
        # q k v: (b * frame/sp, h*w, hn*hd), BSH
        # q k v: (frame/sp, b * h*w, hn*hd), SBH

        # q k v: (frame/sp, b * h*w, hn*hd), SBH
        if self.layout == "SBH":

            q_f, q_b, _ = q.shape  # (frame/sp, b * h*w, hn*hd)
            k_f, k_b, _ = q.shape
            v_f, v_b, _ = q.shape

            # (frame/sp, b * h*w, hn*hd) -> (frame/sp * b * h*w, hn, hd)
            q = q.view(-1, h, head_dim)  # [s // sp, b, h * d] -> [s // sp * b, h, d]
            k = k.view(-1, h, head_dim)
            v = v.view(-1, h, head_dim)
            h_size = h * head_dim
            h_size_sp = h_size // self.sp_size

            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            # (frame/sp * b * h*w, hn, hd) -> (frame, b * h*w, hn * hd / sp)
            q = self.alltoall_sbh_q(q).view(-1, batch_size, h_size_sp)
            k = self.alltoall_sbh_k(k).view(-1, batch_size, h_size_sp)
            v = self.alltoall_sbh_v(v).view(-1, batch_size, h_size_sp)

            if self.use_rope:
                self.apply_rope(q, k, v, position_q, position_k)

            if self.enable_flash_attention:
                # reshape qkv shape ((s b hn*hd) -> (b*hd n hd)) and mask dtype for FA input format
                q = q.view(-1, q_b, h // self.sp_size, head_dim).transpose(1, 2, 0, 3).contiguous()
                k = k.view(-1, k_b, h // self.sp_size, head_dim).transpose(1, 2, 0, 3).contiguous()
                v = v.view(-1, v_b * batch_size, h // self.sp_size, head_dim).transpose(1, 2, 0, 3).contiguous()

                # (batch_size, hn, N, hd)
                if mask is not None:
                    assert mask.dim() == 4, f"Expect to have 4-dim mask for FA, but got mask shape {mask.shape}"
                    # (b, h, 1, k_n) - > (b, h, q_n, k_n), manual broadcast
                    if mask.shape[-2] == 1:
                        mask = mask.repeat(q.shape[-2], axis=-2)

                # (b * h*w, hn/sp, frame, hd)
                out = self.flash_attention(q, k, v, mask)
                b, h_, n, d = out.shape
                # (b * h*w * frame, hn/sp, hd)
                out = out.transpose(0, 2, 1, 3).view(-1, h_, d)
                # (b * h*w * frame, hn/sp, hd) -> (b * h*w * frame/sp, hn, hd) -> (frame/sp, b * h*w, hn*hd)
                out = self.alltoall_sbh_out(out).view(-1, batch_size, h_size)
            else:
                # (frame, b * h*w, hn * hd / sp) -> (b * h*w * hn/sp, frame, hd)
                q = q.view(-1, q_b, h // self.sp_size, head_dim
                           ).transpose(1, 2, 0, 3).view(q_b * h // self.sp_size, -1, head_dim).contiguous()
                k = k.view(-1, k_b, h // self.sp_size, head_dim
                           ).transpose(1, 2, 0, 3).view(k_b * h // self.sp_size, -1, head_dim).contiguous()
                v = v.view(-1, v_b, h // self.sp_size, head_dim
                           ).transpose(1, 2, 0, 3).view(v_b * h // self.sp_size, -1, head_dim).contiguous()

                # (batch_size, -1, attention_mask.shape[-1])
                if mask is not None:
                    assert (
                        mask.dim() == 3
                    ), f"Expect to have 3-dim mask for vanilla Attention, but got mask shape {mask.shape}"
                    assert (
                        mask.shape[0] == q.shape[0]
                    ), f"Expect to have the first dim (bs * num_heads) = {q.shape[0]},  but got {mask.shape[0]}"

                # (b * h*w * hn/sp, frame, hd)
                out = self.attention(q, k, v, mask)
                _, n, d = out.shape
                out = out.view(-1, h // self.sp_size, n, d).transpose(0, 2, 1, 3).view(-1, h // self.sp_size, d)
                out = self.alltoall_sbh_out(out).view(-1, batch_size, h_size)
        else:
            q_b, q_n, _ = q.shape  # (b n h*d)
            k_b, k_n, _ = k.shape
            v_b, v_n, _ = v.shape

            if self.use_rope:
                self.apply_rope(q, k, v, position_q, position_k)

            if self.enable_flash_attention:
                # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
                q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
                k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
                v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
                if mask is not None:
                    assert mask.dim() == 4, f"Expect to have 4-dim mask for FA, but got mask shape {mask.shape}"
                    # (b, h, 1, k_n) - > (b, h, q_n, k_n), manual broadcast
                    if mask.shape[-2] == 1:
                        mask = mask.repeat(q_n, axis=-2)
                out = self.flash_attention(q, k, v, mask)
                b, h, n, d = out.shape
                # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
                out = out.transpose(0, 2, 1, 3).view(b, n, -1)
            else:
                # (b, n, h*d) -> (b*h, n, d)
                q = self._rearange_in(q, h)
                k = self._rearange_in(k, h)
                v = self._rearange_in(v, h)
                if mask is not None:
                    assert (
                        mask.dim() == 3
                    ), f"Expect to have 3-dim mask for vanilla Attention, but got mask shape {mask.shape}"
                    assert (
                        mask.shape[0] == q.shape[0]
                    ), f"Expect to have the first dim (bs * num_heads) = {q.shape[0]},  but got {mask.shape[0]}"

                out = self.attention(q, k, v, mask)
                # (b*h, n, d) -> (b, n, h*d)
                out = self._rearange_out(out, h)

        hidden_states = self.to_out(out).to(x_dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states


class ori_MultiHeadAttention(nn.Cell):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to True):
            Set to `True` to upcast the softmax computation to `float32`.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        dtype=ms.float32,
        enable_flash_attention=False,
        use_rope: bool = False,
        rope_scaling: Optional[Dict] = None,
        compress_kv_factor: Optional[Tuple] = None,
        layout: Optional[str] = "BSH",
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.dropout = dropout
        self.heads = heads
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self.dtype = dtype
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.compress_kv_factor = compress_kv_factor
        self.only_cross_attention = only_cross_attention
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.layout = layout

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.added_kv_proj_dim = added_kv_proj_dim

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None."
                " Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        assert not (
            self.use_rope and (self.compress_kv_factor is not None)
        ), "Can not both enable compressing kv and using rope"
        if self.compress_kv_factor is not None:
            self._init_compress()
        if self.use_rope:
            self._init_rope()

        self.to_q = nn.Dense(query_dim, self.inner_dim, has_bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
            self.to_v = nn.Dense(self.cross_attention_dim, self.inner_dim, has_bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Dense(added_kv_proj_dim, self.inner_dim, has_bias=bias)
            self.add_v_proj = nn.Dense(added_kv_proj_dim, self.inner_dim, has_bias=bias)

        self.to_out = nn.SequentialCell(nn.Dense(self.inner_dim, query_dim, has_bias=out_bias), nn.Dropout(p=dropout))

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            self.flash_attention = MSFlashAttention(
                head_dim=dim_head, head_num=heads, fix_head_dims=[72], attention_dropout=attn_drop
            )
        else:
            self.attention = Attention(
                dim_head=dim_head, attn_drop=attn_drop, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax
            )

    def _init_compress(self):
        if len(self.compress_kv_factor) == 2:
            self.sr = nn.Conv2d(
                self.inner_dim,
                self.inner_dim,
                groups=self.inner_dim,
                kernel_size=self.compress_kv_factor,
                stride=self.compress_kv_factor,
            )
            weight = initializer("ones", self.sr.weight.shape) * (1 / self.compress_kv_factor[0] ** 2)
            self.sr.weight.set_data(weight)
        elif len(self.compress_kv_factor) == 1:
            self.kernel_size = self.compress_kv_factor[0]
            self.sr = nn.Conv1d(
                self.inner_dim,
                self.inner_dim,
                groups=self.inner_dim,
                kernel_size=self.compress_kv_factor[0],
                stride=self.compress_kv_factor[0],
            )
            weight = initializer("ones", self.sr.weight.shape) * (1 / self.compress_kv_factor[0])
            self.sr.weight.set_data(weight)
        bias = initializer("zeros", self.sr.bias.shape)
        self.sr.bias.set_data(bias)
        self.norm = LayerNorm(self.inner_dim)

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rope2d = RoPE2D()
            self.rope1d = RoPE1D()
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor_2d = self.rope_scaling["factor_2d"]
            scaling_factor_1d = self.rope_scaling["factor_1d"]
            if scaling_type == "linear":
                self.rope2d = LinearScalingRoPE2D(scaling_factor=scaling_factor_2d)
                self.rope1d = LinearScalingRoPE1D(scaling_factor=scaling_factor_1d)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> ms.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`ms.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `ms.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        assert (
            current_length == target_length
        ), "The attention mask length should be identical to encoder hidden states length"
        f", but got {current_length} and {current_length}"

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, 0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, 1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`ms.Tensor`): Hidden states of the encoder.

        Returns:
            `ms.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def apply_rope(self, query, key, value, position_q, position_k):
        assert self.use_rope, "use_rope must be True"
        head_dim = self.inner_dim // self.heads
        batch_size, seq_len, _ = query.shape
        # (b, n, h*d) -> (b, n, h, d) -> (b, h, n, d)
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        # require the shape of (batch_size x nheads x ntokens x dim)
        if position_q.ndim == 3:
            query = self.rope2d(query, position_q)
        elif position_q.ndim == 2:
            query = self.rope1d(query, position_q)
        else:
            raise NotImplementedError
        if position_k.ndim == 3:
            key = self.rope2d(key, position_k)
        elif position_k.ndim == 2:
            key = self.rope1d(key, position_k)
        else:
            raise NotImplementedError
        # change the to original shape
        # (b, h, n, d) -> (b, n, h, d) -> (b, n, h*d)
        query = query.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        key = key.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        value = value.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        return query, key, value

    def construct(
        self,
        hidden_states,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        scale: float = 1.0,
        position_q: Optional[ms.Tensor] = None,
        position_k: Optional[ms.Tensor] = None,
        last_shape: Tuple[int] = None,
    ):
        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        if self.compress_kv_factor is not None:
            batch_size = hidden_states.shape[0]
            if len(last_shape) == 2:
                encoder_hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, self.dim, *last_shape)
                encoder_hidden_states = (
                    self.sr(encoder_hidden_states).reshape(batch_size, self.dim, -1).permute(0, 2, 1)
                )
            elif len(last_shape) == 1:
                encoder_hidden_states = hidden_states.permute(0, 2, 1)
                if last_shape[0] % 2 == 1:
                    first_frame_pad = encoder_hidden_states[:, :, :1].repeat_interleave(self.kernel_size - 1, -1)
                    encoder_hidden_states = ops.concat((first_frame_pad, encoder_hidden_states), axis=2)
                encoder_hidden_states = self.sr(encoder_hidden_states).permute(0, 2, 1)
            else:
                raise NotImplementedError(f"NotImplementedError with last_shape {last_shape}")

            encoder_hidden_states = self.norm(encoder_hidden_states)

        batch_size, key_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            out_dim = 4 if self.enable_flash_attention else 3
            attention_mask = self.prepare_attention_mask(
                attention_mask, key_length, batch_size, out_dim=out_dim
            )  # make attention mask a correct shape
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            # attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        x_dtype = hidden_states.dtype
        h = self.heads
        mask = attention_mask

        q = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Layout BSH or SBH:
        # q k v: (frame/sp, b * h*w, hn*hd), SBH
        # q k v: (b * frame/sp, h*w, hn*hd), BSH

        if self.layout == "SBH":
            q_f, q_bhw, _ = q.shape
            k_f, k_bhw, _ = k.shape
            v_f, v_bhw, _ = v.shape

            if self.use_rope:
                self.apply_rope(q, k, v, position_q, position_k)

            if self.enable_flash_attention:
                # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
                q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
                k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
                v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
                if mask is not None:
                    assert mask.dim() == 4, f"Expect to have 4-dim mask for FA, but got mask shape {mask.shape}"
                    # (b, h, 1, k_n) - > (b, h, q_n, k_n), manual broadcast
                    if mask.shape[-2] == 1:
                        mask = mask.repeat(q_n, axis=-2)
                out = self.flash_attention(q, k, v, mask)
                b, h, n, d = out.shape
                # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
                out = out.transpose(0, 2, 1, 3).view(b, n, -1)
            else:
                # (b, n, h*d) -> (b*h, n, d)
                q = self._rearange_in(q, h)
                k = self._rearange_in(k, h)
                v = self._rearange_in(v, h)
                if mask is not None:
                    assert (
                            mask.dim() == 3
                    ), f"Expect to have 3-dim mask for vanilla Attention, but got mask shape {mask.shape}"
                    assert (
                            mask.shape[0] == q.shape[0]
                    ), f"Expect to have the first dim (bs * num_heads) = {q.shape[0]},  but got {mask.shape[0]}"

                out = self.attention(q, k, v, mask)
                # (b*h, n, d) -> (b, n, h*d)
                out = self._rearange_out(out, h)

        else:
            q_b, q_n, _ = q.shape  # (b n h*d)
            k_b, k_n, _ = k.shape
            v_b, v_n, _ = v.shape

            if self.use_rope:
                self.apply_rope(q, k, v, position_q, position_k)

            if self.enable_flash_attention:
                # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
                q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
                k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
                v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
                if mask is not None:
                    assert mask.dim() == 4, f"Expect to have 4-dim mask for FA, but got mask shape {mask.shape}"
                    # (b, h, 1, k_n) - > (b, h, q_n, k_n), manual broadcast
                    if mask.shape[-2] == 1:
                        mask = mask.repeat(q_n, axis=-2)
                out = self.flash_attention(q, k, v, mask)
                b, h, n, d = out.shape
                # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
                out = out.transpose(0, 2, 1, 3).view(b, n, -1)
            else:
                # (b, n, h*d) -> (b*h, n, d)
                q = self._rearange_in(q, h)
                k = self._rearange_in(k, h)
                v = self._rearange_in(v, h)
                if mask is not None:
                    assert (
                        mask.dim() == 3
                    ), f"Expect to have 3-dim mask for vanilla Attention, but got mask shape {mask.shape}"
                    assert (
                        mask.shape[0] == q.shape[0]
                    ), f"Expect to have the first dim (bs * num_heads) = {q.shape[0]},  but got {mask.shape[0]}"

                out = self.attention(q, k, v, mask)
                # (b*h, n, d) -> (b, n, h*d)
                out = self._rearange_out(out, h)



        hidden_states = self.to_out(out).to(x_dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states
