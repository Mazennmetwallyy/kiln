"""Paged-attention decode as a hand-written Metal kernel.

The PyTorch backend gathers each sequence's K/V out of the paged pool into a temporary [B, L, Hkv, D]
tensor and then runs SDPA, so every decode step copies the whole KV history of every sequence, in
every layer. This kernel reads K/V straight from the pool through the block table instead.

Work split. One threadgroup computes one (sequence, query head, context partition). Its 4 SIMD groups
of 32 threads take every 4th token of the partition. Within a SIMD group each thread owns D/32 of the
head's dimensions, so the q·k dot product is a `simd_sum` across the 32 threads. Each SIMD group keeps
an online softmax (running max m, running denominator l, running weighted sum of V; FlashAttention's
trick), so scores are never stored. The 4 SIMD groups' results are rescaled to a common max and merged
through threadgroup memory.

Split-K (FlashDecoding). With one request, (sequence, head) alone gives only H threadgroups, far too few
to fill the GPU, and each would walk the whole context serially. So the context is cut into partitions
of PARTITION tokens, each handled by its own threadgroup. Each writes an unnormalised partial result
(m, l, acc); a second small kernel merges the partitions the same way the SIMD groups were merged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

THREADS = 128  # 4 SIMD groups of 32
PARTITION = 256  # context tokens per threadgroup when the context is split
_SOURCE = """
#include <metal_stdlib>
using namespace metal;
#define D __HEAD_DIM__
#define PER (D / 32)
#define NSG 4
typedef __SCALAR__ T;

// Grid: B * H * num_splits threadgroups of 128 threads.
kernel void paged_decode(
    device const T* q              [[buffer(0)]],   // [B, H, D]
    device const T* k_pool         [[buffer(1)]],   // [slots, Hkv, D]
    device const T* v_pool         [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],   // [B, max_blocks]
    device const int* seq_lens     [[buffer(4)]],   // [B]
    device T* out                  [[buffer(5)]],   // [B, H, D]               (num_splits == 1)
    device float* part_acc         [[buffer(6)]],   // [B, H, num_splits, D]   (num_splits > 1)
    device float* part_ml          [[buffer(7)]],   // [B, H, num_splits, 2]   running max, denominator
    constant uint& num_heads       [[buffer(8)]],
    constant uint& num_kv_heads    [[buffer(9)]],
    constant uint& max_blocks      [[buffer(10)]],
    constant uint& block_size      [[buffer(11)]],
    constant float& scale          [[buffer(12)]],
    constant uint& num_splits      [[buffer(13)]],
    constant uint& partition       [[buffer(14)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint split = gid % num_splits;
    const uint bh = gid / num_splits;
    const uint b = bh / num_heads;
    const uint h = bh % num_heads;
    const uint kvh = h / (num_heads / num_kv_heads);  // grouped-query attention
    const int L = seq_lens[b];
    const int start = int(split * partition);
    const int end = min(L, start + int(partition));
    device const int* table = block_tables + b * max_blocks;

    float qv[PER];
    for (uint i = 0; i < PER; i++) qv[i] = float(q[bh * D + lane * PER + i]) * scale;

    float m = -INFINITY, l = 0.0f, acc[PER];
    for (uint i = 0; i < PER; i++) acc[i] = 0.0f;

    for (int t = start + int(sg); t < end; t += NSG) {
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
    if (sg != 0) return;

    float M = -INFINITY;
    for (uint j = 0; j < NSG; j++) M = max(M, tm[j]);
    float w[NSG], denom = 0.0f;
    for (uint j = 0; j < NSG; j++) {
        w[j] = tm[j] == -INFINITY ? 0.0f : exp(tm[j] - M);  // a SIMD group that saw no tokens
        denom += tl[j] * w[j];
    }
    for (uint i = 0; i < PER; i++) {
        float o = 0.0f;
        for (uint j = 0; j < NSG; j++) o += tacc[j][lane * PER + i] * w[j];
        if (num_splits == 1) {
            out[bh * D + lane * PER + i] = T(o / denom);
        } else {
            part_acc[(gid) * D + lane * PER + i] = o;  // unnormalised, relative to max M
        }
    }
    if (num_splits > 1 && lane == 0) {
        part_ml[gid * 2] = M;  // -INFINITY if this partition is past the end of the sequence
        part_ml[gid * 2 + 1] = denom;
    }
}

// Merge the partitions of each (sequence, head). Grid: B * H threadgroups of D threads.
kernel void merge_partitions(
    device const float* part_acc [[buffer(0)]],
    device const float* part_ml  [[buffer(1)]],
    device T* out                [[buffer(2)]],
    constant uint& num_splits    [[buffer(3)]],
    uint bh [[threadgroup_position_in_grid]],
    uint d  [[thread_position_in_threadgroup]])
{
    device const float* ml = part_ml + bh * num_splits * 2;
    float M = -INFINITY;
    for (uint s = 0; s < num_splits; s++) M = max(M, ml[s * 2]);
    float o = 0.0f, denom = 0.0f;
    for (uint s = 0; s < num_splits; s++) {
        if (ml[s * 2] == -INFINITY) continue;
        const float w = exp(ml[s * 2] - M);
        denom += ml[s * 2 + 1] * w;
        o += part_acc[(bh * num_splits + s) * D + d] * w;
    }
    out[bh * D + d] = T(o / denom);
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
                           block_size: int, max_seq_len: int, partition: int = PARTITION) -> Tensor:
    """q: [B, H, D]; pools: [slots, Hkv, D]; block_tables: [B, max_blocks] int32; seq_lens: [B] int32.
    Each sequence's query attends to its first seq_lens[b] positions; max_seq_len = max(seq_lens)
    (passed as a Python int so the launch never reads back from the GPU). Returns [B, H, D]."""
    batch, heads, head_dim = q.shape
    lib = _lib(q.dtype, head_dim)
    splits = -(-max_seq_len // partition)
    out = torch.empty_like(q)
    if splits > 1:
        part_acc = torch.empty(batch * heads * splits * head_dim, dtype=torch.float32, device=q.device)
        part_ml = torch.empty(batch * heads * splits * 2, dtype=torch.float32, device=q.device)
    else:
        part_acc = part_ml = out  # unused
    lib.paged_decode(q.contiguous(), k_pool, v_pool, block_tables, seq_lens, out, part_acc, part_ml, heads,
                     k_pool.shape[1], block_tables.shape[1], block_size, head_dim ** -0.5, splits, partition,
                     threads=batch * heads * splits * THREADS, group_size=THREADS)
    if splits > 1:
        lib.merge_partitions(part_acc, part_ml, out, splits, threads=batch * heads * head_dim, group_size=head_dim)
    return out
