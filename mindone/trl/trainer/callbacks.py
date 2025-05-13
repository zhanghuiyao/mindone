import mindspore
from mindspore import nn, ops

from typing import Optional, Union

from mindone.transformers.trainer import TrainerCallback
from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel


class SyncRefModelCallback(TrainerCallback):
    """
    Callback to synchronize the model with a reference model.
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Cell],
    ):
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        for target_param, copy_param in zip(target_model.get_parameters(), model.get_parameters()):
            ops.assign(target_param, target_param * (1.0 - alpha) + copy_param * alpha)

    @staticmethod
    def sync_target_model(model, target_model, alpha):
        # TODO: support zero3
        SyncRefModelCallback._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)
