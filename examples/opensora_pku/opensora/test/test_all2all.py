
import os, sys
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

from opensora.test.init_env import init_env
from opensora.models.diffusion.latte.modules import MultiHeadAttention

from opensora.acceleration.communications import AllToAll_SBH
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info


def run_all2all_sp(q):

    alltoall_sbh_q = AllToAll_SBH(scatter_dim=1, gather_dim=0)

    # 3. run
    _q = alltoall_sbh_q(q)

    print(f"input q.shape: {q.shape}")
    print(f"_q_out.shape: {_q.shape}")
    print(f"_q_out.mean: {_q.mean()}")
    print(f"_q_out.min: {_q.min()}")
    print(f"_q_out.max: {_q.max()}")

    return _q


if __name__ == '__main__':
    # 0. init env
    init_env(sp_size=2)

    # (f // sp, b, N) -> (f, b, N)
    norm_hidden_states = np.load("dump_data/step00/2_tem_b_MHA1_0_norm_hidden_states.npy")
    norm_hidden_states = np.concatenate(
        (norm_hidden_states[:, ...], norm_hidden_states[:, ...] * 0.3), axis=0
    )
    f_, bhw, N = norm_hidden_states.shape
    h, d = 16, 72
    norm_hidden_states = Tensor(norm_hidden_states)

    print("\n============== run sp ==============")
    # (f, b, N) -> (f // sp * b, h, d)
    norm_hidden_states_sp = norm_hidden_states.chunk(2, axis=0)[hccl_info.rank%hccl_info.world_size].reshape(-1, h, d)
    out_sp = run_all2all_sp(norm_hidden_states_sp)  # (f // sp * b, h, d) -> (f * b, h // sp, d)
    print("====================================")

    print("\n============== run no sp ==============")
    # (f, b, N) -> (f * b, h // sp, d)
    out = norm_hidden_states.view(-1, h, d).chunk(2, axis=1)[hccl_info.rank%hccl_info.world_size]
    print("=======================================")

    out_sp, out = out_sp.asnumpy(), out.asnumpy()
    diff_abs = np.abs(out_sp - out).mean()
    diff_rel = (np.abs(out_sp - out) / np.abs(out)).mean()
    diff_rel_eps = (np.abs(out_sp - out) / (np.abs(out) + np.abs(out.mean()))).mean()

    print("\n============== diff ==============")
    print(f"diff_abs: {diff_abs}")
    print(f"diff_rel: {diff_rel * 100:.2f}%")
    print(f"diff_rel_eps: {diff_rel_eps * 100:.2f}%")
    print("==================================")




