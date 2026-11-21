import pytest
import torch

from kiln.attention.metadata import build_metadata
from kiln.attention.torch_paged import PagedContext
from kiln.cache import BlockManager
from kiln.cache.paged import PagedKVCache
from kiln.model.kv_cache import ContiguousKVCache
from kiln.model.loader import load_model
from tests.model.tiny import make_tiny_hf_model

torch.set_grad_enabled(False)


@pytest.fixture
def tiny(tmp_path):
    path, _ = make_tiny_hf_model("qwen2", tmp_path)
    return load_model(path)


def contiguous_logits(model, config, ids):
    cache = ContiguousKVCache(config, len(ids), torch.float32, "cpu")
    cache.set_positions(0, len(ids))
    return model(torch.tensor(ids), torch.arange(len(ids)), cache)


class Runner:
    """Minimal driver: grows sequences in the block manager, builds metadata, runs one forward."""

    def __init__(self, model, config, num_blocks=32, block_size=4):
        self.model, self.config = model, config
        self.bm = BlockManager(num_blocks, block_size)
        self.cache = PagedKVCache(config, num_blocks, block_size, torch.float32, "cpu")

    def step(self, batch):  # batch: list of (seq_id, new_token_ids)
        ids, positions = [], []
        for sid, toks in batch:
            if sid not in self.bm.seq_ids:
                self.bm.allocate(sid, len(toks))
            else:
                self.cache.copy_blocks(self.bm.append_slots(sid, len(toks)))
            end = self.bm.seq_len(sid)
            ids += toks
            positions += list(range(end - len(toks), end))
        meta = build_metadata(self.bm, [s for s, _ in batch], [len(t) for _, t in batch], "cpu")
        return self.model(torch.tensor(ids), torch.tensor(positions), PagedContext(self.cache, meta))


def test_paged_matches_contiguous_single_seq(tiny):
    model, config = tiny
    ids = list(range(3, 23))  # 20 tokens, crosses several 4-token blocks
    ref = contiguous_logits(model, config, ids)
    r = Runner(model, config)
    rows = [r.step([(1, ids[:10])])]
    rows += [r.step([(1, [t])]) for t in ids[10:]]
    torch.testing.assert_close(torch.cat(rows), ref, atol=1e-5, rtol=0)


def test_two_sequences_in_one_flat_batch(tiny):
    model, config = tiny
    a, b = list(range(10, 19)), list(range(40, 45))  # lengths 9 and 5
    out = Runner(model, config).step([(1, a), (2, b)])
    torch.testing.assert_close(out[:9], contiguous_logits(model, config, a), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[9:], contiguous_logits(model, config, b), atol=1e-5, rtol=0)


def test_mixed_prefill_and_decode_batch(tiny):
    model, config = tiny
    a, b = list(range(10, 19)), list(range(40, 52))
    r = Runner(model, config)
    r.step([(1, a)])
    out = r.step([(1, [77]), (2, b)])  # seq 1 decodes one token while seq 2 prefills
    torch.testing.assert_close(out[0], contiguous_logits(model, config, [*a, 77])[-1], atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1:], contiguous_logits(model, config, b), atol=1e-5, rtol=0)


def test_decode_batch_matches_individual(tiny):
    model, config = tiny
    prompts = {1: list(range(5, 12)), 2: list(range(60, 63)), 3: list(range(100, 117))}
    r = Runner(model, config)
    r.step(list(prompts.items()))
    out = r.step([(s, [9]) for s in prompts])  # pure decode: the vectorised path
    for i, (_, p) in enumerate(prompts.items()):
        torch.testing.assert_close(out[i], contiguous_logits(model, config, [*p, 9])[-1], atol=1e-5, rtol=0)


def test_cow_does_not_corrupt_sibling(tiny):
    model, config = tiny
    prompt = list(range(20, 26))  # 6 tokens, block size 4 -> second block is shared and partial
    r = Runner(model, config)
    r.step([(1, prompt)])
    r.bm.fork(1, 2)
    out = r.step([(1, [50]), (2, [51])])
    torch.testing.assert_close(out[0], contiguous_logits(model, config, [*prompt, 50])[-1], atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1], contiguous_logits(model, config, [*prompt, 51])[-1], atol=1e-5, rtol=0)
    out = r.step([(1, [60]), (2, [61])])
    torch.testing.assert_close(out[0], contiguous_logits(model, config, [*prompt, 50, 60])[-1],
                               atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1], contiguous_logits(model, config, [*prompt, 51, 61])[-1],
                               atol=1e-5, rtol=0)


def test_blocks_for_memory(tiny):
    _, config = tiny  # 2 layers, 2 kv heads, head_dim 16
    per_block = 2 * config.num_layers * 16 * config.num_kv_heads * config.head_dim * 4  # K+V, fp32, bs 16
    assert PagedKVCache.blocks_for_memory(config, per_block * 10 + 5, 16, torch.float32) == 10
