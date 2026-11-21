import torch

from bench.hf_server import static_batch_generate
from tests.model.tiny import make_tiny_hf_model


def test_static_batch_matches_individual_greedy(tmp_path):
    _, hf = make_tiny_hf_model("qwen2", tmp_path)
    prompts = [[5, 6, 7], [9] * 11, [40, 41]]
    out_lens = [4, 9, 2]
    batched = static_batch_generate(hf, prompts, out_lens, pad_id=0)
    assert [len(b) for b in batched] == out_lens
    for p, n, b in zip(prompts, out_lens, batched, strict=True):
        alone = static_batch_generate(hf, [p], [n], pad_id=0)[0]
        assert b == alone
        with torch.no_grad():
            ids = torch.tensor([p])
            for _ in range(n):
                ids = torch.cat([ids, hf(ids).logits[:, -1:].argmax(-1)], dim=1)
        assert b == ids[0, len(p):].tolist()
