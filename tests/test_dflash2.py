import torch
import torch.nn.functional as F
from types import SimpleNamespace

from allamo.model.dflash.model import DFlash2CandidateSelector, DFlash2DynamicConv


def _config(taps=2, group_size=2, block_size=4, hidden=8, vocab=32, rank=4, top_k=3):
    return SimpleNamespace(
        n_embd=hidden,
        vocab_size=vocab,
        dflash_config={
            "block_size": block_size,
            "conv_kernel_size": taps,
            "conv_group_size": group_size,
            "selector_rank": rank,
            "selector_top_k": top_k,
        },
    )


def _vllm_grouped_conv(hidden, delta, base, block_size, group_size):
    """CPU reference copied from vLLM's DFlash 2 grouped convolution."""
    num_groups = hidden.shape[-1] // group_size
    taps = base.shape[0]
    blocks = hidden.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden.shape[0], device=hidden.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def test_parameter_layout_matches_vllm():
    taps, groups, hidden = 3, 4, 8
    conv = DFlash2DynamicConv(_config(taps=taps, group_size=2, hidden=hidden))

    assert conv.base_kernel.shape == (2, taps, hidden)
    assert conv.kernel_projection.weight.shape == (2 * taps * groups, hidden)
    assert conv.kernel_projection.bias is None


def test_identity_initialization_is_a_noop():
    conv = DFlash2DynamicConv(_config())
    conv.init_weights()
    x = torch.randn(2, 8, 8)
    prepared, side1 = conv.prepare(x)
    finished = conv.finish(torch.randn_like(x), side1)

    assert torch.allclose(prepared, x)
    assert torch.allclose(finished, conv.finish(finished, side1))
    direct = conv.finish(x, side1)
    assert torch.allclose(direct, x)


def test_convolution_matches_vllm_reference():
    torch.manual_seed(0)
    block_size, taps, group_size, hidden = 4, 3, 2, 8
    conv = DFlash2DynamicConv(_config(taps=taps, group_size=group_size, block_size=block_size, hidden=hidden))
    conv.base_kernel.data.normal_()
    conv.kernel_projection.weight.data.normal_()
    x = torch.randn(2, 2 * block_size, hidden)

    prepared, side1 = conv.prepare(x)
    rows = x.reshape(-1, hidden)
    coefficients = conv.kernel_projection(rows).reshape(rows.shape[0], 2, taps, hidden // group_size)
    expected_prepare = _vllm_grouped_conv(
        rows, coefficients[:, 0], conv.base_kernel[0], block_size, group_size
    ).reshape_as(x)
    sublayer_out = torch.randn_like(x)
    finished = conv.finish(sublayer_out, side1)
    expected_finish = _vllm_grouped_conv(
        sublayer_out.reshape(-1, hidden),
        coefficients[:, 1],
        conv.base_kernel[1],
        block_size,
        group_size,
    ).reshape_as(x)

    assert torch.allclose(prepared, expected_prepare, atol=1e-6)
    assert torch.allclose(finished, expected_finish, atol=1e-6)


def test_finish_reuses_prepare_coefficients():
    torch.manual_seed(1)
    conv = DFlash2DynamicConv(_config())
    conv.init_weights()
    conv.kernel_projection.weight.data.normal_()
    x = torch.randn(1, 8, 8)
    _, side1 = conv.prepare(x)
    conv.kernel_projection.weight.data.zero_()
    sublayer_out = torch.randn_like(x)

    finished = conv.finish(sublayer_out, side1)
    expected = _vllm_grouped_conv(
        sublayer_out.reshape(-1, 8),
        side1.reshape(-1, 2, 4),
        conv.base_kernel[1],
        conv.draft_block_size,
        conv.group_size,
    ).reshape_as(sublayer_out)

    assert torch.allclose(finished, expected, atol=1e-6)
    assert not torch.allclose(finished, sublayer_out)


def test_taps_do_not_cross_block_boundaries():
    conv = DFlash2DynamicConv(_config(taps=2, group_size=1, block_size=4, hidden=4))
    with torch.no_grad():
        conv.base_kernel.zero_()
        conv.base_kernel[:, 1, :] = 1.0
        conv.kernel_projection.weight.zero_()
    x = torch.zeros(1, 8, 4)
    x[:, 3, :] = 1.0
    prepared, _ = conv.prepare(x)

    assert torch.allclose(prepared[:, 4], torch.zeros(4))
    assert torch.allclose(prepared[:, 3], x[:, 2])


def test_selector_checkpoint_names_and_score():
    selector = DFlash2CandidateSelector(_config())
    selector.init_weights()
    keys = selector.state_dict().keys()

    assert "predecessor_codebook" in keys
    assert "successor_codebook" in keys
    assert "predecessor_codebook.weight" not in keys
    assert "successor_codebook.weight" not in keys
    assert selector.predecessor_codebook.shape == (32, 4)

    hidden = torch.randn(2, 8)
    pred_ids = torch.randint(0, 32, (2,))
    cand_ids = torch.randint(0, 32, (2, 3))
    cand_logits = torch.randn(2, 3)
    scores = selector(hidden, pred_ids, cand_ids, cand_logits)
    ctx = selector.hidden_projection(hidden).unsqueeze(-2)
    expected = cand_logits + (
        (selector.predecessor_codebook[pred_ids].unsqueeze(-2) * ctx)
        * selector.successor_codebook[cand_ids]
    ).sum(-1)

    assert torch.allclose(scores, expected)
