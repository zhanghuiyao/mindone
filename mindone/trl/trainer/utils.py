import numpy as np

import mindspore
from mindspore import nn, ops, mint
from mindspore.ops import operations as P


def disable_dropout_in_model(model: nn.Cell) -> None:

    for cell in model.name_cells().values():
        if isinstance(cell, nn.Dropout):
            cell.p = 0
            cell.keep_prob = 1.0
            cell.dropout = P.Dropout(1.0)


def pad(tensors: list[mindspore.Tensor], padding_value: int = 0, padding_side: str = "right") -> mindspore.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[mindspore.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `mindspore.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import mindspore
        >>> pad([mindspore.Tensor([1, 2, 3]), mindspore.Tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([mindspore.Tensor([[1, 2], [3, 4]]), mindspore.Tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = ops.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [mindspore.float32, mindspore.float64]:
        selected_logits = mint.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = mint.stack([mint.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []

        # FIXME: graph mode not support `zip` on MindSpore 2.5.0
        # for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
        for i in range(logits.shape[0]):
            row_logits, row_labels = logits[i], index[i]

            row_logps = ops.log_softmax(row_logits, axis=-1)
            row_per_token_logps = mint.gather(row_logps, dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = mint.stack(per_token_logps)
    return per_token_logps
