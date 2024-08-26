import torch
import torch.nn.functional as F


def single_seq_attn():
    N = 3  # batch size
    L = 6  # seq length

    # batch size one case, works!
    # default causal attn mask
    mask = torch.tril(torch.ones(L, L))
    mask_q = torch.tril(torch.ones(L, L), diagonal=-1)
    mask_q[0, 0] = 1

    # generate permutation per sample in batch
    noise = torch.rand(1, L)
    perm = torch.argsort(noise, dim=1)
    restore = torch.argsort(perm, dim=1)

    # generate attn mask for each sample in batch
    # output shape should be (N, num_heads, L, L)
    # good enough to be (N, 1, L, L)
    idx = perm.repeat(L, 1)
    idx_r = perm.view(L, 1).repeat(1, L)
    # permute the columns
    attn_partial = torch.gather(mask, dim=1, index=idx)
    attn_partial_q = torch.gather(mask_q, dim=1, index=idx)
    # then permute the rows
    attn = torch.gather(attn_partial, dim=0, index=idx_r)
    attn_q = torch.gather(attn_partial_q, dim=0, index=idx_r)

    print("mask: ", mask)
    print("perm: ", perm)
    print("restore: ", restore)
    print("causal attn: ", mask)
    print("partial attn: ", attn_partial)
    print("attn: ", attn)
    print("attn_q: ", attn_q)

    # Then query attention
    torch.diagonal(attn, dim1=0, dim2=1).zero_()
    # Then add back the solo
    attn[perm[0][0], perm[0][0]] = 1.0
    print("padded query attn: ", attn)

    # Pad on only top?
    attn = F.pad(attn, (1, 0, 1, 0), value=0.0)
    attn[0, :] = 1.0
    print("padded attn: ", attn)

    skip = 2
    print("perm tgt:", perm >= skip)
    print("restore tgt:", restore >= skip)


def batch_attn():
    N = 3  # batch size
    L = 6  # seq length

    # default causal attn mask
    mask = torch.tril(torch.ones(L, L)).view(1, L, L)

    # generate permutation per sample in batch
    noise = torch.rand(N, L)
    perm = torch.argsort(noise, dim=1)
    restore = torch.argsort(perm, dim=1)

    # generate attn mask for each sample in batch
    # output shape should be (N, num_heads, L, L)
    # good enough to be (N, 1, L, L)
    idx = perm.view(N, 1, L).repeat(1, L, 1)
    idx_r = restore.view(N, L, 1).repeat(1, 1, L)
    attn_partial = torch.gather(mask.repeat(N, 1, 1), dim=2, index=idx)
    attn = torch.gather(attn_partial, dim=1, index=idx_r)

    print("mask: ", mask)
    print("perm: ", perm)
    print("causal attn: ", mask)
    print("partial attn: ", attn_partial)
    print("attn: ", attn)


def batch_head_attn():
    N = 3  # batch size
    L = 6  # seq length

    # default causal attn mask
    mask = torch.tril(torch.ones(L, L)).view(1, 1, L, L)

    # generate permutation per sample in batch
    noise = torch.rand(N, L)
    perm = torch.argsort(noise, dim=1)
    restore = torch.argsort(perm, dim=1)

    # generate attn mask for each sample in batch
    # output shape should be (N, num_heads, L, L)
    # good enough to be (N, 1, L, L)
    idx = perm.view(N, 1, 1, L).repeat(1, 1, L, 1)
    idx_r = restore.view(N, 1, L, 1).repeat(1, 1, 1, L)
    attn_partial = torch.gather(mask.repeat(N, 1, 1, 1), dim=3, index=idx)
    attn = torch.gather(attn_partial, dim=2, index=idx_r)

    print("mask: ", mask)
    print("perm: ", perm)
    print("causal attn: ", mask)
    print("partial attn: ", attn_partial)
    print("attn: ", attn)


def debug():
    """
    torch.save(nan_mask.nonzero(), "nan_mask.pt")
    torch.save(out[nan_mask.nonzero()[:, 0].unique(
        sorted=True)], "out_vals.pt")
    """
    # nan_mask = torch.load("nan_mask.pt")
    # out_vals = torch.load("out_vals.pt")
    # print("nan_mask: ", nan_mask.shape)
    # print("out_vals: ", out_vals.shape)
    L = 4
    vals = torch.rand(L, L)
    out = torch.tril(torch.ones(L, L), diagonal=-1)
    vals = vals.masked_fill_(out == 0, -1e20)
    print(vals)
    vals = vals.softmax(dim=-1)
    print(vals)


if __name__ == "__main__":
    # single_seq_attn()
    # batch_attn()
    # batch_head_attn()
    debug()
