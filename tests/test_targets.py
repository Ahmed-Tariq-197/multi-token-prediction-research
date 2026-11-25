"""Head targets: offset alignment, window layout, and document-aware valid-position masking."""
import torch

from mtp.targets import IGNORE_INDEX as IGN, build_head_targets, document_segments


def reference_targets(window, T, k, eos, mask=True):
    """Slow, obviously-correct python-loop reference."""
    B = window.shape[0]
    out = torch.full((B, k, T), IGN, dtype=torch.long)
    for b in range(B):
        toks = window[b].tolist()
        for t in range(T):
            for i in range(1, k + 1):
                j = t + i
                if mask and eos in toks[t:j]:        # an EOS at positions t..j-1 closes the document before j
                    continue
                out[b, i - 1, t] = toks[j]
    return out


def test_offsets_explicit_example():
    # window = T + H tokens; no EOS present (eos_id=0 never appears)
    window = torch.tensor([[11, 12, 13, 14, 15, 16, 17]])          # T=4, H=3
    tg = build_head_targets(window, seq_len=4, k=3, eos_id=0)
    assert tg.shape == (1, 3, 4)
    assert tg[0, 0].tolist() == [12, 13, 14, 15]                   # head 1: +1
    assert tg[0, 1].tolist() == [13, 14, 15, 16]                   # head 2: +2
    assert tg[0, 2].tolist() == [14, 15, 16, 17]                   # head 3: +3


def test_head1_equals_standard_shifted_next_token():
    window = torch.randint(1, 50, (3, 20))
    ids, tg = window[:, :16], build_head_targets(window, 16, 1, eos_id=0)
    assert torch.equal(tg[:, 0], window[:, 1:17])


def test_head_count_smaller_than_horizon_window():
    window = torch.randint(1, 50, (2, 16 + 8))                      # horizon 8, but k=2
    tg = build_head_targets(window, 16, 2, eos_id=0)
    assert tg.shape == (2, 2, 16)
    assert torch.equal(tg[:, 1], window[:, 2:18])


def test_cross_document_masking_explicit():
    # EOS (=0) at window index 3. T=4, k=3.
    window = torch.tensor([[5, 6, 7, 0, 8, 9, 10]])
    tg = build_head_targets(window, 4, 3, eos_id=0)
    # t=0: head1->6, head2->7, head3->0 (the EOS ending the doc is a legal target)
    assert tg[0, :, 0].tolist() == [6, 7, 0]
    # t=1: head1->7, head2->0 (EOS), head3-> x[4]=8 is in the NEXT document -> ignored
    assert tg[0, :, 1].tolist() == [7, 0, IGN]
    # t=2: head1->0 (EOS), heads 2,3 cross the boundary -> ignored
    assert tg[0, :, 2].tolist() == [0, IGN, IGN]
    # t=3 is the EOS token itself; everything it would predict is in the next doc -> all ignored
    assert tg[0, :, 3].tolist() == [IGN, IGN, IGN]


def test_mask_disabled_keeps_everything():
    window = torch.tensor([[5, 6, 7, 0, 8, 9, 10]])
    tg = build_head_targets(window, 4, 3, eos_id=0, mask_cross_document=False)
    assert (tg != IGN).all()


def test_matches_reference_on_random_windows():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        T, k, H = 12, 4, 6
        window = torch.randint(1, 5, (4, T + H), generator=g)
        eos_pos = torch.rand(4, T + H, generator=g) < 0.15
        window[eos_pos] = 0
        assert torch.equal(build_head_targets(window, T, k, 0), reference_targets(window, T, k, 0))


def test_segments_definition():
    w = torch.tensor([[1, 0, 2, 3, 0, 4]])
    assert document_segments(w, 0).tolist() == [[0, 0, 1, 1, 1, 2]]


def test_window_too_short_raises():
    import pytest
    with pytest.raises(AssertionError):
        build_head_targets(torch.ones(1, 10, dtype=torch.long), seq_len=8, k=4, eos_id=0)
