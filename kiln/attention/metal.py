"""Paged-attention decode as a hand-written Metal kernel.

The PyTorch backend gathers each sequence's K/V out of the paged pool into a temporary [B, L, Hkv, D]
tensor and then runs SDPA, so every decode step copies the whole KV history of every sequence, in
every layer. This kernel reads K/V straight from the pool through the block table instead.

One threadgroup computes one (sequence, query head). Its 4 SIMD groups of 32 threads take every 4th
token. Within a SIMD group each thread owns D/32 of the head's dimensions: the q·k dot product is a
`simd_sum` across the 32 threads. Each SIMD group keeps an online softmax (running max m, running
denominator l, running weighted sum of V; FlashAttention's trick), so scores are never stored. At the
end the 4 partial results are rescaled to a common max and merged through threadgroup memory.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

THREADS = 128  # 4 SIMD groups of 32
_SOURCE = """
#include <metal_stdlib>
using namespace metal;
#define D __HEAD_DIM__
#define PER (D / 32)
#define NSG 4
typedef __SCALAR__ T;

kernel void paged_decode(
    device const T* q              [[buffer(0)]],   // [B, H, D]
    device const T* k_pool         [[buffer(1)]],   // [slots, Hkv, D]
    device const T* v_pool         [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],   // [B, max_blocks]
    device const int* seq_lens     [[buffer(4)]],   // [B]
    device T* out                  [[buffer(5)]],   // [B, H, D]
    constant uint& num_heads       [[buffer(6)]],
    constant uint& num_kv_heads    [[buffer(7)]],
    constant uint& max_blocks      [[buffer(8)]],
    constant uint& block_size      [[buffer(9)]],
    constant float& scale          [[buffer(10)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint b = gid / num_heads;
    const uint h = gid % num_heads;
    const uint kvh = h / (num_heads / num_kv_heads);  // grouped-query attention
    const int L = seq_lens[b];
    device const int* table = block_tables + b * max_blocks;

    float qv[PER];
    for (uint i = 0; i < PER; i++) qv[i] = float(q[(b * num_heads + h) * D + lane * PER + i]) * scale;

    float m = -INFINITY, l = 0.0f, acc[PER];
    for (uint i = 0; i < PER; i++) acc[i] = 0.0f;

    for (int t = int(sg); t < L; t += NSG) {
        const uint slot = uint(table[t / int(block_size)]) * block_size + uint(t) % block_size;
        const uint base = (slot * num_kv_heads + kvh) * D + lane * PER;
        float s = 0.0f;
        for (uint i = 0; i < PER; i++) s += qv[i] * float(k_pool[base + i]);
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float correction = exp(m - m_new);  // rescale what was accumulated under the old max
        const float p = exp(s - m_new);
        l = l * correction + p;
        for (uint i = 0; i < PER; i++) acc[i] = acc[i] * correction + p * float(v_pool[base + i]);
        m = m_new;
    }

    threadgroup float tm[NSG], tl[NSG];
    threadgroup float tacc[NSG][D];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    for (uint i = 0; i < PER; i++) tacc[sg][lane * PER + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float M = -INFINITY;
        for (uint j = 0; j < NSG; j++) M = max(M, tm[j]);
        float denom = 0.0f;
        float w[NSG];
        for (uint j = 0; j < NSG; j++) {
            w[j] = tm[j] == -INFINITY ? 0.0f : exp(tm[j] - M);  // a SIMD group that saw no tokens
            denom += tl[j] * w[j];
        }
        for (uint i = 0; i < PER; i++) {
            float o = 0.0f;
            for (uint j = 0; j < NSG; j++) o += tacc[j][lane * PER + i] * w[j];
            out[(b * num_heads + h) * D + lane * PER + i] = T(o / denom);
        }
    }
}
"""
_SCALARS = {torch.float16: "half", torch.float32: "float"}
_libs: dict[tuple[torch.dtype, int], Any] = {}


def supported(dtype: torch.dtype, head_dim: int) -> bool:
    return torch.backends.mps.is_available() and dtype in _SCALARS and head_dim % 32 == 0 and head_dim <= 256


def _lib(dtype: torch.dtype, head_dim: int) -> Any:
    key = (dtype, head_dim)
    if key not in _libs:
        source = _SOURCE.replace("__HEAD_DIM__", str(head_dim)).replace("__SCALAR__", _SCALARS[dtype])
        _libs[key] = torch.mps.compile_shader(source)
    return _libs[key]


def paged_decode_attention(q: Tensor, k_pool: Tensor, v_pool: Tensor, block_tables: Tensor, seq_lens: Tensor,
                           block_size: int) -> Tensor:
    """q: [B, H, D]; pools: [slots, Hkv, D]; block_tables: [B, max_blocks] int32; seq_lens: [B] int32.
    Each sequence's query attends to its first seq_lens[b] positions. Returns [B, H, D]."""
    batch, heads, head_dim = q.shape
    out = torch.empty_like(q)
    _lib(q.dtype, head_dim).paged_decode(
        q.contiguous(), k_pool, v_pool, block_tables, seq_lens, out, heads, k_pool.shape[1],
        block_tables.shape[1], block_size, head_dim ** -0.5, threads=batch * heads * THREADS, group_size=THREADS)
    return out
