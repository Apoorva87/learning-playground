# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__unsafe_view_mul_silu_0 = async_compile.cpp_pybinding(['at::BFloat16*', 'const at::BFloat16*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(at::BFloat16* in_out_ptr0,
                       const at::BFloat16* in_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(278528LL); x0+=static_cast<int64_t>(8LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(278528LL)))
                {
                    auto tmp0 = at::vec::Vectorized<at::BFloat16>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp9 = at::vec::Vectorized<at::BFloat16>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::convert<float,2,at::BFloat16,1>(tmp0);
                    auto tmp2 = tmp1.neg();
                    auto tmp3 = tmp2.exp();
                    auto tmp4 = static_cast<float>(1.0);
                    auto tmp5 = at::vec::VectorizedN<float,2>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp1 / tmp6;
                    auto tmp8 = (tmp7);
                    auto tmp10 = at::vec::convert<float,2,at::BFloat16,1>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = at::vec::convert<at::BFloat16,1,float,2>(tmp11);
                    tmp12.store(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 16, 5120), (81920, 5120, 1))
        assert_size_stride(arg1_1, (17408, 5120), (5120, 1))
        assert_size_stride(arg2_1, (17408, 5120), (5120, 1))
        assert_size_stride(arg3_1, (5120, 17408), (17408, 1))
        buf0 = empty_strided_cpu((16, 17408), (17408, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
        # [Provenance debug handles] extern_kernels.mm:1
        extern_kernels.mm(reinterpret_tensor(arg0_1, (16, 5120), (5120, 1), 0), reinterpret_tensor(arg1_1, (5120, 17408), (1, 5120), 0), out=buf0)
        del arg1_1
        buf1 = empty_strided_cpu((16, 17408), (17408, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
        # [Provenance debug handles] extern_kernels.mm:2
        extern_kernels.mm(reinterpret_tensor(arg0_1, (16, 5120), (5120, 1), 0), reinterpret_tensor(arg2_1, (5120, 17408), (1, 5120), 0), out=buf1)
        del arg0_1
        del arg2_1
        buf2 = reinterpret_tensor(buf0, (1, 16, 17408), (278528, 17408, 1), 0); del buf0  # reuse
        # [Provenance debug handles] cpp_fused__unsafe_view_mul_silu_0:3
        cpp_fused__unsafe_view_mul_silu_0(buf2, buf1)
        del buf1
        buf3 = empty_strided_cpu((16, 5120), (5120, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear, silu, linear_1, mul, linear_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
        # [Provenance debug handles] extern_kernels.mm:4
        extern_kernels.mm(reinterpret_tensor(buf2, (16, 17408), (17408, 1), 0), reinterpret_tensor(arg3_1, (17408, 5120), (1, 17408), 0), out=buf3)
        del arg3_1
        return (reinterpret_tensor(buf3, (1, 16, 5120), (81920, 5120, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 16, 5120), (81920, 5120, 1), device='cpu', dtype=torch.bfloat16)
    arg1_1 = rand_strided((17408, 5120), (5120, 1), device='cpu', dtype=torch.bfloat16)
    arg2_1 = rand_strided((17408, 5120), (5120, 1), device='cpu', dtype=torch.bfloat16)
    arg3_1 = rand_strided((5120, 17408), (17408, 1), device='cpu', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
