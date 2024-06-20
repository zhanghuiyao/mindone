
import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init

from opensora.acceleration.parallel_states import initialize_sequence_parallel_state

def init_env(sp_size=1):
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O1"})

    init()
    device_num = get_group_size()
    rank_id = get_rank()
    ms.reset_auto_parallel_context()

    ms.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.DATA_PARALLEL,
        gradients_mean=True,
        device_num=device_num,
    )

    initialize_sequence_parallel_state(sp_size)

