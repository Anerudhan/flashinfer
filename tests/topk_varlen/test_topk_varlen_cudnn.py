"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for the cuDNN DSA backend of ``top_k_varlen``.

Covered here and nowhere else:

* the DSv3.2 / GLM-5.3 operating point (``top_k = 2048``, fp32), which sits
  exactly on cuDNN's ``indexer_topk_max_k`` ceiling rather than under it;
* the ``next_n`` stagger, where cuDNN and FlashInfer must agree on which row of
  a group sees which prefix — the one place the two APIs could silently
  disagree while both looking correct;
* that ``backend="auto"`` still never yields ``"cudnn"``, so registering it
  cannot perturb an existing deployment's backend choice.

``_check_correct`` and ``_make_varlen_inputs`` mirror the helpers in
``test_topk_varlen.py`` so the cuDNN backend is held to exactly the same
contract as the others. They are duplicated rather than imported because the
modules in this directory are self-contained (no ``__init__.py``, no
cross-imports), which is what keeps them runnable individually.
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE, reason="flashinfer not installed"
)


def _make_varlen_inputs(seq_len_list, N, dtype, seed):
    """Ragged batch: ``(logits[batch, N], seq_lens[batch] int32)``."""
    batch_size = len(seq_len_list)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
    return logits, seq_lens


def _check_correct(
    indices,
    logits,
    seq_lens,
    top_k,
    next_n=1,
    compress_ratio=1,
    require_all_checked=False,
):
    """Every selected value must be >= the k-th largest in its row.

    ``require_all_checked=True`` turns the otherwise-silent "skip degenerate
    row" branch into a hard failure, so a mis-parametrized test cannot quietly
    verify nothing.
    """
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    n_checked = 0
    for row in range(indices.shape[0]):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            if require_all_checked:
                raise AssertionError(
                    f"row={row}: N_eff={N_eff} < top_k={top_k} — degenerate row "
                    f"not allowed under require_all_checked"
                )
            continue
        row_logits = logits_f32[row, :N_eff]
        kth_value = torch.topk(row_logits, k=top_k).values[-1].item()
        sel = [int(i) for i in indices[row].cpu().tolist() if i >= 0]
        assert len(sel) == top_k, f"row={row}: got {len(sel)} indices, want {top_k}"
        assert len(set(sel)) == len(sel), f"row={row}: duplicate indices"
        assert all(i < N_eff for i in sel), f"row={row}: out-of-range index"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        assert (sel_vals < kth_value).sum() == 0, (
            f"row={row}: some selected values below kth-rank ({kth_value:.6f})"
        )
        n_checked += 1
    if require_all_checked:
        assert n_checked == indices.shape[0], (
            f"only {n_checked}/{indices.shape[0]} rows were verified"
        )


def _cudnn_supported() -> bool:
    """Static registration + CC check, then the runtime cuDNN-FE probe.

    ``is_backend_supported`` answers only the static question (registered, and
    FlashInfer ships a kernel for this CC). The DSA submodule and the >= 1.28
    frontend floor are environment facts, so both have to be asked separately —
    see ``_cudnn_dsa_topk_ready``.
    """
    if not torch.cuda.is_available() or not _FLASHINFER_AVAILABLE:
        return False
    from flashinfer.topk_varlen.topk_varlen import _cudnn_dsa_topk_ready

    major, minor = get_compute_capability(torch.device("cuda"))
    return (
        flashinfer.top_k_varlen.is_backend_supported("cudnn", major * 10 + minor)
        and _cudnn_dsa_topk_ready()
    )


requires_cudnn = pytest.mark.skipif(
    not _cudnn_supported(),
    reason="cuDNN DSA top-K needs nvidia-cudnn-frontend>=1.28 on sm_90/100/103/107",
)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@requires_cudnn
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("N", [4096, 32768])
@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_cudnn_basic(dtype, top_k, N, batch_size):
    """Uniform-length batch: every selected value is >= the row's k-th largest."""
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=17)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="cudnn")
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_cudnn
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_cudnn_varlen(dtype, top_k):
    """Ragged batch: rows shorter than their buffer must not leak padding."""
    N = 8192
    seq_len_list = [8192, 6000, 4096, 2049, 8192, 3000, 2048, 7777]
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=23)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="cudnn")
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_cudnn
@pytest.mark.parametrize("next_n", [2, 4])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_cudnn_next_n(next_n, top_k):
    """Speculative-decode stagger.

    cuDNN documents the stagger as "first row sees the shortest prefix, last
    sees the full ``seq_lens``"; ``_check_correct`` independently derives the
    same relation as ``seq_lens[row // next_n] - next_n + (row % next_n) + 1``.
    If the two conventions disagreed — e.g. reversed within the group — rows
    would be validated against the wrong prefix and this fails.
    """
    dtype, N, num_groups = torch.float32, 8192, 8
    num_rows = num_groups * next_n
    torch.manual_seed(31)
    logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    seq_lens = torch.full((num_groups,), N, dtype=torch.int32, device="cuda")

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, next_n=next_n, backend="cudnn"
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )


@requires_cudnn
def test_cudnn_return_values():
    """Returned values must equal ``logits[row, indices]``."""
    dtype, top_k, N, batch_size = torch.float32, 1024, 8192, 4
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=41)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, return_values=True, backend="cudnn"
    )
    torch.cuda.synchronize()

    assert values is not None and values.shape == (batch_size, top_k)
    assert values.dtype == dtype
    for row in range(batch_size):
        expected = logits[row][indices[row].long()].float()
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


@requires_cudnn
def test_cudnn_preallocated_outputs():
    """Caller buffers are the destination.

    The cuDNN FE wrapper allocates its own outputs, so ``_run_cudnn`` copies
    into the caller's buffers. This asserts the copy actually lands *and* that
    the caller's own tensor objects come back — a CUDA-graph caller captures
    those addresses, so returning cuDNN's fresh tensors instead would replay
    into the wrong memory.
    """
    dtype, top_k, N, batch_size = torch.float32, 512, 4096, 4
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=53)

    out_i = torch.empty((batch_size, top_k), dtype=torch.int32, device="cuda")
    out_v = torch.empty((batch_size, top_k), dtype=dtype, device="cuda")
    out_i.fill_(-12345)

    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        return_values=True,
        out_indices=out_i,
        out_values=out_v,
        backend="cudnn",
    )
    torch.cuda.synchronize()

    assert ret_i is out_i, "returned indices must be the caller's buffer object"
    assert ret_v is out_v, "returned values must be the caller's buffer object"
    assert (out_i != -12345).all(), "caller buffer was not written"
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


@requires_cudnn
def test_cudnn_matches_torch_topk():
    """cuDNN selects the same top-K value multiset as ``torch.topk``.

    Compares sorted *values* rather than indices so ties cannot cause spurious
    failures. ``torch.topk`` is the reference rather than a sibling backend
    because it is ground truth and needs no JIT toolchain — the CUTLASS radix
    backends require a working ``nvcc``/ninja, which not every environment
    running this test will have.
    """
    dtype, top_k, N, batch_size = torch.float32, 1024, 8192, 8
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=67)

    idx_cudnn, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="cudnn")
    torch.cuda.synchronize()

    lf = logits.float()
    ref_vals = torch.topk(lf, k=top_k, dim=-1).values  # already descending
    for row in range(batch_size):
        v_cudnn = lf[row][idx_cudnn[row].long()].sort(descending=True).values
        assert torch.allclose(v_cudnn, ref_vals[row], rtol=1e-4, atol=1e-4), (
            f"row={row}: cudnn vs torch.topk value multisets differ"
        )


@requires_cudnn
def test_cudnn_glm53_operating_point():
    """GLM-5.3 / DeepSeek-V3.2 shape: top-2048 over a long context, fp32.

    ``index_topk = 2048`` is cuDNN's maximum, not a value comfortably inside
    its range, so this is the configuration most likely to regress if that
    bound ever tightens.
    """
    top_k, N, batch_size = 2048, 131072, 4
    logits, seq_lens = _make_varlen_inputs(
        [N] * batch_size, N, torch.float32, seed=2048
    )
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="cudnn")
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# Rejections — each mirrors a hard cuDNN constraint
#
# ``@backend_requirement`` separates two rejection kinds and raises a different
# exception for each: a backend that is unregistered or unsupported on this
# compute capability gives ``BackendSupportedError``, while a *registered*
# backend whose checker declines the specific call gives
# ``ValueError("Problem size is not supported ...")``. Everything below is the
# second kind, so ``ValueError`` is the contract being asserted.
# ---------------------------------------------------------------------------


@requires_cudnn
@pytest.mark.parametrize("top_k", [2049, 4096])
def test_cudnn_rejects_top_k_above_2048(top_k):
    """> 2048 exceeds cuDNN's ``indexer_topk_max_k``; must fail at validation."""
    logits, seq_lens = _make_varlen_inputs([8192] * 4, 8192, torch.float32, seed=71)
    with pytest.raises(ValueError, match="Problem size is not supported"):
        flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="cudnn")


@requires_cudnn
def test_cudnn_rejects_compress_ratio():
    """cuDNN's wrapper has no compression parameter — reject rather than ignore.

    Silently ignoring it would be a correctness bug: ``seq_lens`` is in
    uncompressed-token space when ``compress_ratio > 1``, so the kernel would
    search a window ``compress_ratio`` times too long.
    """
    logits, seq_lens = _make_varlen_inputs([8192] * 4, 8192, torch.float32, seed=73)
    with pytest.raises(ValueError, match="Problem size is not supported"):
        flashinfer.top_k_varlen(
            logits, seq_lens, 512, compress_ratio=4, backend="cudnn"
        )


# ---------------------------------------------------------------------------
# Registration must not perturb "auto"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cudnn_never_selected_by_auto():
    """``auto`` must not pick cuDNN.

    ``_top_k_varlen_heuristic`` filters against an explicit ``order`` list, and
    "cudnn" is deliberately absent from it. This pins that: registering the
    backend must not change any existing deployment's ``auto`` choice.
    """
    dtype, top_k, N, batch_size = torch.float32, 1024, 8192, 4
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=79)
    flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="auto")
    assert "cudnn" not in flashinfer.top_k_varlen.suitable_auto_backends, (
        "cudnn leaked into the auto ranking"
    )
