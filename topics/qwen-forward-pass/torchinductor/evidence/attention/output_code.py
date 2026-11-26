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


cpp_fused__scaled_dot_product_flash_attention_for_cpu_clone_expand_unsqueeze_view_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(24LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(4352LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(4352LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 4352LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(6LL)))), static_cast<int64_t>(4));
                        tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 4352LL*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(24LL); x0+=static_cast<int64_t>(1LL))
        {
            for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(4352LL); x1+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(4352LL)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 4352LL*(c10::div_floor_integer(static_cast<int64_t>(x0), static_cast<int64_t>(6LL)))), static_cast<int64_t>(4));
                        tmp0.store(out_ptr1 + static_cast<int64_t>(x1 + 4352LL*x0));
                    }
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 4, 17, 256), (17408, 4352, 256, 1))
        assert_size_stride(arg1_1, (1, 4, 17, 256), (17408, 4352, 256, 1))
        assert_size_stride(arg2_1, (1, 24, 1, 256), (6144, 256, 256, 1))
        buf0 = empty_strided_cpu((1, 24, 17, 256), (104448, 4352, 256, 1), torch.float32)
        buf1 = empty_strided_cpu((1, 24, 17, 256), (104448, 4352, 256, 1), torch.float32)
        # [Provenance debug handles] cpp_fused__scaled_dot_product_flash_attention_for_cpu_clone_expand_unsqueeze_view_0:1
        cpp_fused__scaled_dot_product_flash_attention_for_cpu_clone_expand_unsqueeze_view_0(arg0_1, arg1_1, buf0, buf1)
        del arg0_1
        del arg1_1
        # Topologically Sorted Source Nodes: [keys, values, ], Original ATen: [aten.unsqueeze, aten.expand, aten.clone, aten.view, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:2
        buf2 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg2_1, buf0, buf1, scale=0.0625)
        del arg2_1
        del buf0
        del buf1
        buf3 = buf2[0]
        assert_size_stride(buf3, (1, 24, 1, 256), (6144, 256, 256, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        # buffer buf3 (op: torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default) is assumed to be not aligned
        return (buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 4, 17, 256), (17408, 4352, 256, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 4, 17, 256), (17408, 4352, 256, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 24, 1, 256), (6144, 256, 256, 1), device='cpu', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
