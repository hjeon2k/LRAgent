import math

import pytest
import torch
from einops import rearrange, repeat
from flash_attn import (
    flash_lora_attn_func,
    flash_attn_func,
)


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False, key_leftpad=None
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        if key_leftpad is not None:
            key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
            col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_lora_ref(
    q,
    k,
    v,
    lr_v,
    lr_B,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Reference implementation for flash_lora_attn_func.
    
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        lr_v: (batch_size, seqlen_k, lora_dim)
        lr_B: (nheads_k, lora_dim, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32
        reorder_ops: whether to change the order of operations
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        lr_v = lr_v.float()
        lr_B = lr_B.float()
    
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    nheads = q.shape[2]
    nheads_k = k.shape[2]
    
    # Compute LoRA delta_v: lr_v @ lr_B -> (batch, seqlen_k, nheads_k, headdim)
    # lr_v: (batch, seqlen_k, lora_dim), lr_B: (nheads_k, lora_dim, headdim)
    delta_v = torch.einsum("bsr,hrd->bshd", lr_v, lr_B)
    
    # Add LoRA to v
    v_lora = v + delta_v
    
    # Repeat for GQA/MQA
    k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_lora = repeat(v_lora, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    
    if attn_bias is not None:
        scores = scores + attn_bias
    
    attention = torch.softmax(scores, dim=-1).to(v_lora.dtype)
    
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    
    # We want to mask here so that the attention matrix doesn't have any NaNs
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v_lora * dropout_scaling)
    
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


# Test parameters
# Fixed: batch_size=1, d=128, dropout_p=0.0
# SM80 only: bfloat16 and float16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
def test_flash_lora_attn_output(
    seqlen_q, seqlen_k, causal, local, alibi, deterministic, mha_type, dtype
):
    """
    Test flash_lora_attn_func forward pass.
    Fixed: batch_size=1, d=128, dropout_p=0.0
    """
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128  # head_dim fixed to 128
    dropout_p = 0.0
    
    # Variable parameters
    nheads = 8
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    
    # Create Q, K, V
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    
    # Create LoRA parameters
    # lr_v: (batch_size, seqlen_k, lora_dim)
    # lr_B: (nheads_k, lora_dim, headdim)
    r = 8  # Fixed lora_rank
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads_k, r, d, device=device, dtype=dtype) * 0.1
    
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    
    # Run flash_lora_attn_func
    out = flash_lora_attn_func(
        q,
        k,
        v,
        lr_v,
        lr_B,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=False,
    )
    
    # Reference implementation
    out_ref, attn_ref = attention_lora_ref(
        q,
        k,
        v,
        lr_v,
        lr_B,
        None,
        None,
        attn_bias,
        dropout_p,
        None,
        causal=causal,
        window_size=window_size,
    )
    
    # PyTorch reference (with reorder_ops for numerical error estimation)
    out_pt, attn_pt = attention_lora_ref(
        q,
        k,
        v,
        lr_v,
        lr_B,
        None,
        None,
        attn_bias,
        dropout_p,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most four times the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
def test_flash_lora_attn_causal(seqlen_q, seqlen_k, swap_sq_sk, local, causal, dtype):
    """
    Test flash_lora_attn_func with various causal/local configurations.
    Fixed: batch_size=1, d=128, dropout_p=0.0
    """
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128
    nheads = 6
    nheads_k = nheads  # MHA for simplicity
    
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    
    # LoRA parameters
    r = 8  # Fixed lora_rank
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads_k, r, d, device=device, dtype=dtype) * 0.1
    
    out = flash_lora_attn_func(
        q, k, v, lr_v, lr_B, 0.0, causal=causal, window_size=window_size
    )
    
    out_ref, attn_ref = attention_lora_ref(
        q, k, v, lr_v, lr_B, None, None, None, 0.0, None, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_lora_ref(
        q,
        k,
        v,
        lr_v,
        lr_B,
        None,
        None,
        None,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most four times the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (3, 1024),
        (1, 339),
        (64, 800),
        (3, 799),
        (64, 2048),
        (128, 128),
        (256, 256),
    ],
)
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
def test_flash_lora_attn_splitkv(
    seqlen_q, seqlen_k, swap_sq_sk, causal, local, alibi, dtype
):
    """
    Test flash_lora_attn_func for split-KV scenarios (small seqlen_q, large seqlen_k).
    Fixed: batch_size=1, d=128, dropout_p=0.0
    """
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128
    nheads = 8
    nheads_k = nheads
    
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    
    # LoRA parameters
    r = 8  # Fixed lora_rank
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads_k, r, d, device=device, dtype=dtype) * 0.1
    
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    
    out, lse, _ = flash_lora_attn_func(
        q,
        k,
        v,
        lr_v,
        lr_B,
        0.0,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=False,
        return_attn_probs=True,
    )
    
    out_ref, attn_ref = attention_lora_ref(
        q, k, v, lr_v, lr_B, None, None, attn_bias, 0.0, None, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_lora_ref(
        q,
        k,
        v,
        lr_v,
        lr_B,
        None,
        None,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most four times the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("nheads", [1, 4, 8, 12, 16])
# @pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ],
)
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
def test_flash_lora_attn_nheads(seqlen_q, seqlen_k, nheads, dtype):
    """
    Test flash_lora_attn_func with various number of heads.
    Fixed: batch_size=1, d=128, dropout_p=0.0, causal=True
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128
    causal = True
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # LoRA parameters
    r = 8  # Fixed lora_rank
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads, r, d, device=device, dtype=dtype) * 0.1
    
    out = flash_lora_attn_func(
        q, k, v, lr_v, lr_B, 0.0, causal=causal
    )
    
    out_ref, attn_ref = attention_lora_ref(
        q, k, v, lr_v, lr_B, None, None, None, 0.0, None, causal=causal
    )
    out_pt, attn_pt = attention_lora_ref(
        q, k, v, lr_v, lr_B, None, None, None, 0.0, None, causal=causal, upcast=False, reorder_ops=True
    )
    
    print(f"nheads={nheads}")
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (384, 384),
        (512, 512),
    ],
)
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
def test_flash_lora_attn_deterministic(seqlen_q, seqlen_k, causal, dtype):
    """
    Test that flash_lora_attn_func returns consistent results.
    Fixed: batch_size=1, d=128, dropout_p=0.0
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128
    nheads = 8
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # LoRA parameters
    r = 8  # Fixed lora_rank
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads, r, d, device=device, dtype=dtype) * 0.1
    
    out0 = flash_lora_attn_func(
        q, k, v, lr_v, lr_B, 0.0, causal=causal, deterministic=True
    )
    
    for i in range(50):
        out = flash_lora_attn_func(
            q, k, v, lr_v, lr_B, 0.0, causal=causal, deterministic=True
        )
        assert torch.equal(out, out0), f"Mismatch at iteration {i}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 512),
        (512, 256),
        (1024, 1024),
    ],
)
# @pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128)])
def test_flash_lora_attn_softcap(seqlen_q, seqlen_k, causal, softcap, dtype):
    """
    Test flash_lora_attn_func with softcap.
    Fixed: batch_size=1, d=128, dropout_p=0.0
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128
    nheads = 8
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q = q * softcap
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # LoRA parameters
    r = 8  # Fixed lora_rank
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads, r, d, device=device, dtype=dtype) * 0.1
    
    out = flash_lora_attn_func(
        q, k, v, lr_v, lr_B, 0.0, causal=causal, softcap=softcap
    )
    
    out_ref, attn_ref = attention_lora_ref(
        q, k, v, lr_v, lr_B, None, None, None, 0.0, None, causal=causal, softcap=softcap
    )
    out_pt, attn_pt = attention_lora_ref(
        q, k, v, lr_v, lr_B, None, None, None, 0.0, None, causal=causal, softcap=softcap, upcast=False, reorder_ops=True
    )
    
    print(f"softcap={softcap}")
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


# =============================================================================
# Benchmark Tests - Kernel Execution Time Measurement
# =============================================================================

def benchmark_forward(fn, *args, warmup=25, rep=100, **kwargs):
    """
    Benchmark a function using CUDA events for accurate timing.
    
    Args:
        fn: Function to benchmark
        *args: Arguments to pass to fn
        warmup: Number of warmup iterations
        rep: Number of timed iterations
        **kwargs: Keyword arguments to pass to fn
    
    Returns:
        Tuple of (mean_ms, min_ms, max_ms)
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    
    for i in range(rep):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    
    return mean_ms, min_ms, max_ms


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
        (65536, 65536),
    ],
)
def test_flash_lora_attn_benchmark(seqlen_q, seqlen_k, causal, dtype):
    """
    Benchmark flash_lora_attn_func kernel execution time.
    Fixed: batch_size=1, d=128, nheads=8, lora_rank=8
    
    Run with: pytest test_flash_lora_attn.py::test_flash_lora_attn_benchmark -v -s
    """
    if (
        max(seqlen_q, seqlen_k) >= 8192
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Skip for low memory GPUs
    
    device = "cuda"
    torch.random.manual_seed(0)
    
    # Fixed parameters
    batch_size = 1
    d = 128
    nheads = 32
    nheads_k = 8
    r = 8  # lora_rank
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads_k, r, d, device=device, dtype=dtype) * 0.1
    
    # Benchmark flash_lora_attn_func
    mean_ms, min_ms, max_ms = benchmark_forward(
        flash_lora_attn_func,
        q, k, v, lr_v, lr_B, 0.0,
        causal=causal,
    )

    ref_mean_ms, ref_min_ms, ref_max_ms = benchmark_forward(
        flash_attn_func,
        q, k, v, 0.0,
        causal=causal,
    )
    
    # Calculate FLOPS (approximate)
    # Attention FLOPs: 4 * batch * nheads * seqlen_q * seqlen_k * d (for QK^T and softmax@V)
    # LoRA FLOPs: 2 * batch * seqlen_k * r * nheads_k * d (for lr_v @ lr_B)
    attn_flops = 4 * batch_size * nheads * seqlen_q * seqlen_k * d
    lora_flops = 2 * batch_size * nheads * seqlen_q * seqlen_k * r + 2 * batch_size * nheads * seqlen_q * r * d
    total_flops = attn_flops + lora_flops
    tflops = total_flops / (mean_ms * 1e-3) / 1e12
    ref_tflops = attn_flops / (ref_mean_ms * 1e-3) / 1e12
    
    print(f"\n{'='*60}")
    print(f"Benchmark: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, dtype={dtype}")
    print(f"{'='*60}")
    print(f"  Mean time: {mean_ms:.3f} ms")
    print(f"  Min time:  {min_ms:.3f} ms")
    print(f"  Max time:  {max_ms:.3f} ms")
    print(f"  Throughput: {tflops:.2f} TFLOPS")
    print(f"{'='*60}\n")
    print(f"Ref:")
    print(f"  Mean time: {ref_mean_ms:.3f} ms")
    print(f"  Min time:  {ref_min_ms:.3f} ms")
    print(f"  Max time:  {ref_max_ms:.3f} ms")
    print(f"  Throughput: {ref_tflops:.2f} TFLOPS")
    print(f"{'='*60}\n")


# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
# @pytest.mark.parametrize("nheads", [8, 16, 32])
# def test_flash_lora_attn_benchmark_nheads(seqlen, nheads, dtype):
#     """
#     Benchmark flash_lora_attn_func with varying number of heads.
    
#     Run with: pytest test_flash_lora_attn.py::test_flash_lora_attn_benchmark_nheads -v -s
#     """
#     device = "cuda"
#     torch.random.manual_seed(0)
    
#     batch_size = 1
#     d = 128
#     r = 8
#     causal = True
    
#     q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
#     k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
#     v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
#     lr_v = torch.randn(batch_size, seqlen, r, device=device, dtype=dtype) * 0.1
#     lr_B = torch.randn(nheads, r, d, device=device, dtype=dtype) * 0.1
    
#     mean_ms, min_ms, max_ms = benchmark_forward(
#         flash_lora_attn_func,
#         q, k, v, lr_v, lr_B, 0.0,
#         causal=causal,
#     )
    
#     print(f"\n[seqlen={seqlen}, nheads={nheads}] Mean: {mean_ms:.3f} ms, Min: {min_ms:.3f} ms, Max: {max_ms:.3f} ms")

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "seqlen_k",
    [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize("nheads", [32])
def test_flash_lora_attn_benchmark_decode(seqlen_k, nheads, dtype):
    """
    Benchmark flash_lora_attn_func for decode step (seqlen_q=1, causal=False).
    This simulates the autoregressive decoding scenario where we generate one token at a time.
    
    Fixed: batch_size=1, d=128, seqlen_q=1, causal=False, lora_rank=8
    
    Run with: pytest test_flash_lora_attn.py::test_flash_lora_attn_benchmark_decode -v -s
    """
    device = "cuda"
    torch.random.manual_seed(0)
    
    # Decode step parameters
    batch_size = 1
    seqlen_q = 1  # Single token generation
    d = 128
    r = 8
    nheads_kv = 8
    causal = False  # No causal mask needed for decode (all previous tokens are visible)
    
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    lr_v = torch.randn(batch_size, seqlen_k, r, device=device, dtype=dtype) * 0.1
    lr_B = torch.randn(nheads_kv, r, d, device=device, dtype=dtype) * 0.1
    
    mean_ms, min_ms, max_ms = benchmark_forward(
        flash_lora_attn_func,
        q, k, v, lr_v, lr_B, 0.0,
        causal=causal,
    )

    ref_mean_ms, ref_min_ms, ref_max_ms = benchmark_forward(
        flash_attn_func,
        q, k, v, 0.0,
        causal=causal,
    )
    
    # Calculate memory bandwidth (approximate)
    # Read: Q, K, V, lr_v, lr_B  |  Write: Output
    bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    q_bytes = batch_size * seqlen_q * nheads * d * bytes_per_elem
    k_bytes = batch_size * seqlen_k * nheads_kv * d * bytes_per_elem
    v_bytes = batch_size * seqlen_k * nheads_kv * d * bytes_per_elem
    lr_v_bytes = batch_size * seqlen_k * r * bytes_per_elem
    lr_B_bytes = nheads_kv * r * d * bytes_per_elem
    out_bytes = batch_size * seqlen_q * nheads * d * bytes_per_elem
    total_bytes = q_bytes + k_bytes + v_bytes + lr_v_bytes + lr_B_bytes + out_bytes
    bandwidth_gb_s = total_bytes / (mean_ms * 1e-3) / 1e9
    ref_bandwidth_gb_s = (q_bytes + k_bytes + v_bytes + out_bytes) / (ref_mean_ms * 1e-3) / 1e9
    
    print(f"\n{'='*70}")
    print(f"DECODE Benchmark: seqlen_q=1, seqlen_k={seqlen_k}, nheads={nheads}, dtype={dtype}")
    print(f"{'='*70}")
    print(f"  Mean time: {mean_ms:.4f} ms")
    print(f"  Min time:  {min_ms:.4f} ms")
    print(f"  Max time:  {max_ms:.4f} ms")
    print(f"  Memory Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"{'='*70}\n")
    print(f"Ref:")
    print(f"  Mean time: {ref_mean_ms:.4f} ms")
    print(f"  Min time:  {ref_min_ms:.4f} ms")
    print(f"  Max time:  {ref_max_ms:.4f} ms")
    print(f"  Memory Bandwidth: {ref_bandwidth_gb_s:.2f} GB/s")
    print(f"{'='*70}\n")

def test_benchmark_simple(q_seq, k_seq, warmup = 20, rep = 100):
    device = "cuda"
    torch.random.manual_seed(0)
    
    # Decode step parameters
    batch_size = 1
    d = 128
    r = 8
    n_qheads = 32
    n_kvheads = 8
    if (q_seq == 1):
        causal = False
    else:
        causal = True
    
    q = torch.randn(batch_size, q_seq, n_qheads, d, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, k_seq, n_kvheads, d, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, k_seq, n_kvheads, d, device=device, dtype=torch.float16)
    lr_v = torch.randn(batch_size, k_seq, r, device=device, dtype=torch.float16) * 0.1
    lr_B = torch.randn(n_kvheads, r, d, device=device, dtype=torch.float16) * 0.1

    # Benchmark flash_lora_attn_func
    mean_ms, min_ms, max_ms = benchmark_forward(
        flash_lora_attn_func,
        q, k, v, lr_v, lr_B, 0.0,
        causal=causal,
        warmup=warmup,
        rep=rep
    )

    ref_mean_ms, ref_min_ms, ref_max_ms = benchmark_forward(
        flash_attn_func,
        q, k, v, 0.0,
        causal=causal,
        warmup=warmup,
        rep=rep
    )

    if min_ms < max_ms * 0.5:
        assert False, "Unstable benchmark results detected."
    if ref_min_ms < ref_max_ms * 0.5:
        assert False, "Unstable benchmark results detected."

    attn_flops = 4 * batch_size * n_qheads * q_seq * k_seq * d
    lora_flops = 2 * batch_size * n_qheads * q_seq * k_seq * r + 2 * batch_size * n_qheads * q_seq * r * d
    total_flops = attn_flops + lora_flops
    tflops = total_flops / (mean_ms * 1e-3) / 1e12
    ref_flops = attn_flops / (ref_mean_ms * 1e-3) / 1e12

    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "tflops": tflops,
        "ref_mean_ms": ref_mean_ms,
        "ref_min_ms": ref_min_ms,
        "ref_max_ms": ref_max_ms,
        "ref_tflops": ref_flops
    }

if __name__ == "__main__":

    for seq in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        print(f"=== Benchmarking Prefill seqlen={seq} ===")
        result = test_benchmark_simple(seq, seq, warmup=25, rep=100)
        print(f"  Mean time: {result['mean_ms']:.3f} ms")
        print(f"  Min time:  {result['min_ms']:.3f} ms")
        print(f"  Max time:  {result['max_ms']:.3f} ms")
        print(f"  Throughput: {result['tflops']:.2f} TFLOPS")
        print(f"{'='*70}\n")
        print(f"Ref:")
        print(f"  Mean time: {result['ref_mean_ms']:.3f} ms")
        print(f"  Min time:  {result['ref_min_ms']:.3f} ms")
        print(f"  Max time:  {result['ref_max_ms']:.3f} ms")
        print(f"  Throughput: {result['ref_tflops']:.2f} TFLOPS")
        print(f"{'='*70}\n")
    
    for seq in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        print(f"=== Benchmarking Decode seqlen={seq} ===")
        result = test_benchmark_simple(1, seq, warmup=25, rep=100)
        print(f"  Mean time: {result['mean_ms']:.3f} ms")
        print(f"  Min time:  {result['min_ms']:.3f} ms")
        print(f"  Max time:  {result['max_ms']:.3f} ms")
        print(f"  Throughput: {result['tflops']:.2f} TFLOPS")
        print(f"{'='*70}\n")
        print(f"Ref:")
        print(f"  Mean time: {result['ref_mean_ms']:.3f} ms")
        print(f"  Min time:  {result['ref_min_ms']:.3f} ms")
        print(f"  Max time:  {result['ref_max_ms']:.3f} ms")
        print(f"  Throughput: {result['ref_tflops']:.2f} TFLOPS")
        print(f"{'='*70}\n")


