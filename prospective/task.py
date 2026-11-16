"""MQAR with revision: the endpoint the six-arm ladder actually measured.

Each sequence writes key-value pairs, REWRITES some of them, then queries:

    [k v] x n_pairs   [k v'] x n_revisions   [k QUERY] x n_queries

The model reads the answer at the QUERY token; the target value is never an
input token at or after that position (`y` is a separate tensor, not a shifted
copy of `x`; see `test_task.py`).

Read-after-update convention. The mixer is the official MDN layer, which
writes EVERY token into memory before reading it out at the same step. There
is no query write-mask, so the `[k QUERY]` pairs are themselves written: each
query step stores (k, QUERY-embedding) in memory before the readout, and a
later query of the same key reads a memory that already contains that write.
This is the official architecture's behaviour, kept on purpose for an
architecture comparison; it is not a pure "read-only query" probe.

Metrics, all reported with their sample counts `n_<metric>`:

    overall           all queries                        (ladder: recall)
    revised           queried key was overwritten         (ladder: revision)
    untouched         written once, never rewritten       (ladder: retention)
    rev_lag_le8       revised, lag <= 8 tokens
    rev_lag_gt8       revised, lag  > 8 tokens

`lag` is measured in tokens from the rewrite's VALUE token to the query's
readout (QUERY) token, i.e. the number of recurrence steps between the write
of the current value and its retrieval. Pairs are two tokens, so lags are
even; the minimum possible is 2 (a rewrite immediately followed by its query).

Legacy metrics, kept ONLY to reproduce the earlier (v1, reduced-mixer)
numbers, which were reported under the names `immediate_revised` /
`later_revised`:

    legacy_immediate_revised   revised AND query index < round(n_queries/2)
                               AND (query key pos - rewrite key pos) <= 16
    legacy_later_revised       revised AND NOT legacy_immediate

The v1 report described the first as "rewrite <= 8 tokens back"; the code
checked 16 tokens and also restricted to the first half of the queries. The
v1 numbers are reproduced by the legacy metrics, not by `rev_lag_le8`.
"""

import torch

#: the reported lag threshold, in tokens
LAG_TOKENS = 8
#: the v1 code's constant; the v1 mask used 2 * this, plus a query-index cut
LEGACY_IMMEDIATE_GAP = 8
IGNORE = -100

#: the configuration recorded for the production runs (v1 report section 7.1).
#: The CLI defaults in train.py are set from this, not the other way around.
PRODUCTION_TASK = dict(n_pairs=8, n_revisions=6, n_queries=12, n_keys=24,
                       n_values=32)

#: revised accuracy by ACTUAL token lag (readout token - rewrite value token);
#: lags are even, so (1, 8] holds lags 2..8
LAG_BINS = ((0, 8), (8, 16), (16, 24), (24, 32), (32, None))


def lag_bin_name(lo, hi):
    return f"rev_lag_{lo + 1:02d}_{hi:02d}" if hi is not None else f"rev_lag_{lo + 1:02d}_up"


LAG_BIN_METRICS = tuple(lag_bin_name(lo, hi) for lo, hi in LAG_BINS)

METRICS = ("overall", "revised", "untouched", "rev_lag_le8", "rev_lag_gt8",
           "legacy_immediate_revised", "legacy_later_revised") + LAG_BIN_METRICS

#: v3 primary score, declared before any v3 run: fast revision AND retention
PRIMARY_COMPONENTS = ("rev_lag_le8", "untouched")
#: untouched-accuracy non-inferiority margin for the primary comparison (pp)
NONINFERIORITY_MARGIN_PP = 1.0


def primary_score(m):
    """0.5 * (rev_lag_le8 + untouched), both in percent."""
    return 0.5 * (m["rev_lag_le8"] + m["untouched"])


def make_batch(bsz, n_pairs, n_revisions, n_queries, n_keys, n_values,
               generator, immediate_frac=0.5):
    """Token layout: [k v] pairs, then queries as [k QUERY]. Vocabulary is
    keys [0, n_keys), values [n_keys, n_keys + n_values), then QUERY, PAD.

    The generator is consumed in exactly the same order as the v1 code, so a
    given seed produces the same token sequences as before."""
    query_tok = n_keys + n_values
    pad_tok = query_tok + 1
    vocab = pad_tok + 1
    seq_len = 2 * (n_pairs + n_revisions + n_queries)

    x = torch.full((bsz, seq_len), pad_tok, dtype=torch.long)
    y = torch.full((bsz, seq_len), IGNORE, dtype=torch.long)
    group = torch.zeros((bsz, seq_len), dtype=torch.long)   # 1 rev, 2 untouched
    lag = torch.full((bsz, seq_len), -1, dtype=torch.long)  # revised only
    legacy_immediate = torch.zeros((bsz, seq_len), dtype=torch.bool)

    for b in range(bsz):
        keys = torch.randperm(n_keys, generator=generator)[:n_pairs]
        vals = torch.randint(n_values, (n_pairs,), generator=generator)
        latest = {int(k): int(v) for k, v in zip(keys, vals)}
        pos = 0
        for k, v in zip(keys, vals):
            x[b, pos], x[b, pos + 1] = k, n_keys + v
            pos += 2
        revised = keys[torch.randperm(n_pairs, generator=generator)[:n_revisions]]
        rewrite_at = {}
        for k in revised:
            v = int(torch.randint(n_values, (1,), generator=generator))
            x[b, pos], x[b, pos + 1] = k, n_keys + v
            latest[int(k)] = v
            rewrite_at[int(k)] = pos
            pos += 2
        # queries: even indices on revised keys, odd on untouched ones
        untouched = [int(k) for k in keys if int(k) not in rewrite_at]
        revised_l = [int(k) for k in revised]
        order = []
        for i in range(n_queries):
            pool = revised_l if (i % 2 == 0 and revised_l) else (untouched or revised_l)
            order.append(pool[int(torch.randint(len(pool), (1,), generator=generator))])
        n_imm = int(round(n_queries * immediate_frac))
        for i, k in enumerate(order):
            x[b, pos], x[b, pos + 1] = k, query_tok
            y[b, pos + 1] = n_keys + latest[k]
            if k in rewrite_at:
                group[b, pos + 1] = 1
                # readout token (pos+1) minus rewrite value token (r+1)
                lag[b, pos + 1] = pos - rewrite_at[k]
                legacy_immediate[b, pos + 1] = (
                    (i < n_imm)
                    and (pos - rewrite_at[k]) <= 2 * LEGACY_IMMEDIATE_GAP)
            else:
                group[b, pos + 1] = 2
            pos += 2
    return dict(x=x, y=y, group=group, lag=lag,
                legacy_immediate=legacy_immediate, vocab=vocab,
                query_tok=query_tok, pad_tok=pad_tok)


def masks(batch):
    """name -> boolean mask of scored positions, one per metric."""
    y, g, lag = batch["y"], batch["group"], batch["lag"]
    scored = y != IGNORE
    rev = scored & (g == 1)
    leg = batch["legacy_immediate"]
    out = dict(overall=scored, revised=rev, untouched=scored & (g == 2),
               rev_lag_le8=rev & (lag <= LAG_TOKENS),
               rev_lag_gt8=rev & (lag > LAG_TOKENS),
               legacy_immediate_revised=rev & leg,
               legacy_later_revised=rev & ~leg)
    for lo, hi in LAG_BINS:
        m = rev & (lag > lo)
        out[lag_bin_name(lo, hi)] = m & (lag <= hi) if hi is not None else m
    return out


def hits(logits, batch):
    """name -> (correct, count). Summed across batches by the caller, so a
    metric is pooled over all its samples rather than averaged per batch."""
    hit = logits.argmax(-1) == batch["y"]
    return {k: (int((hit & m).sum().item()), int(m.sum().item()))
            for k, m in masks(batch).items()}


def accuracy(logits, batch):
    """Percent per metric plus `n_<metric>` sample counts, for one batch."""
    out = {}
    for k, (c, n) in hits(logits, batch).items():
        out[k] = 100.0 * c / n if n else float("nan")
        out[f"n_{k}"] = n
    return out


# ------------------------------------------------------- held-out evaluation
#: Separately identified held-out suites, evaluated at the end of training on
#: their own fixed sets. They never feed training or model selection. Vocab
#: is unchanged (24 keys, 32 values), so a trained model can be evaluated.
#:   delayN  N distractor [k v] writes, of keys never queried, inserted
#:           between the revisions and the queries (longer delays)
#:   loadN   N initial pairs instead of 8 (higher memory load)
HELDOUT = {
    "delay8": dict(PRODUCTION_TASK, n_distractors=8),
    "delay16": dict(PRODUCTION_TASK, n_distractors=16),
    "load12": dict(PRODUCTION_TASK, n_pairs=12, n_distractors=0),
    "load16": dict(PRODUCTION_TASK, n_pairs=16, n_distractors=0),
}


def make_batch_heldout(bsz, n_pairs, n_revisions, n_queries, n_keys,
                       n_values, n_distractors, generator, immediate_frac=0.5):
    """`make_batch` (left unchanged above) plus distractor writes between the
    revisions and the queries. Distractor keys are drawn from keys NOT among
    the initial pairs and are never queried; the query and metric rules are
    those of `make_batch`. Lags therefore grow by 2 per distractor."""
    if n_pairs + n_distractors > n_keys:
        raise ValueError("not enough distinct keys for the distractors")
    query_tok = n_keys + n_values
    pad_tok = query_tok + 1
    seq_len = 2 * (n_pairs + n_revisions + n_distractors + n_queries)
    x = torch.full((bsz, seq_len), pad_tok, dtype=torch.long)
    y = torch.full((bsz, seq_len), IGNORE, dtype=torch.long)
    group = torch.zeros((bsz, seq_len), dtype=torch.long)
    lag = torch.full((bsz, seq_len), -1, dtype=torch.long)
    legacy_immediate = torch.zeros((bsz, seq_len), dtype=torch.bool)
    for b in range(bsz):
        perm = torch.randperm(n_keys, generator=generator)
        keys = perm[:n_pairs]
        spare = perm[n_pairs:n_pairs + n_distractors]
        vals = torch.randint(n_values, (n_pairs,), generator=generator)
        latest = {int(k): int(v) for k, v in zip(keys, vals)}
        pos = 0
        for k, v in zip(keys, vals):
            x[b, pos], x[b, pos + 1] = k, n_keys + v
            pos += 2
        revised = keys[torch.randperm(n_pairs, generator=generator)[:n_revisions]]
        rewrite_at = {}
        for k in revised:
            v = int(torch.randint(n_values, (1,), generator=generator))
            x[b, pos], x[b, pos + 1] = k, n_keys + v
            latest[int(k)] = v
            rewrite_at[int(k)] = pos
            pos += 2
        for k in spare:
            x[b, pos] = k
            x[b, pos + 1] = n_keys + int(torch.randint(n_values, (1,), generator=generator))
            pos += 2
        untouched = [int(k) for k in keys if int(k) not in rewrite_at]
        revised_l = [int(k) for k in revised]
        order = []
        for i in range(n_queries):
            pool = revised_l if (i % 2 == 0 and revised_l) else (untouched or revised_l)
            order.append(pool[int(torch.randint(len(pool), (1,), generator=generator))])
        n_imm = int(round(n_queries * immediate_frac))
        for i, k in enumerate(order):
            x[b, pos], x[b, pos + 1] = k, query_tok
            y[b, pos + 1] = n_keys + latest[k]
            if k in rewrite_at:
                group[b, pos + 1] = 1
                lag[b, pos + 1] = pos - rewrite_at[k]
                legacy_immediate[b, pos + 1] = (
                    (i < n_imm)
                    and (pos - rewrite_at[k]) <= 2 * LEGACY_IMMEDIATE_GAP)
            else:
                group[b, pos + 1] = 2
            pos += 2
    return dict(x=x, y=y, group=group, lag=lag,
                legacy_immediate=legacy_immediate, vocab=pad_tok + 1,
                query_tok=query_tok, pad_tok=pad_tok)
