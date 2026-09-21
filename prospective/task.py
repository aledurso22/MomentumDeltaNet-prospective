"""MQAR with revision: the endpoint the six-arm ladder actually measured.

Section 6 of the bridge plan: the observed effect was narrow -- immediate
revision +7.53 pp, retention -0.39, recall -0.37 -- so a probe that only tests
retrieval will miss it. Each sequence writes key-value pairs, REWRITES some of
them, then queries. Queries are scored in three groups:

    overall            -- all queries            (the study's `recall`)
    revised            -- the key was overwritten (the study's `revision`)
    untouched          -- written once, never touched  (its `retention`)
    immediate_revised  -- revised, rewrite <= IMMEDIATE_GAP tokens back
    later_revised      -- revised, rewrite further back

These are the five metrics of the six-arm ladder, so the bridge lines up
row-for-row against it. The effect there was +7.53 on immediate_revised and
+2.67 on later_revised, against -0.39 retention and -0.37 recall, which is why
a retrieval-only endpoint would have missed it.
"""

import torch

IMMEDIATE_GAP = 8
IGNORE = -100


def make_batch(bsz, n_pairs, n_revisions, n_queries, n_keys, n_values,
               generator, immediate_frac=0.5):
    """Token layout: [k v] pairs, then queries as [k PAD]. Vocabulary is
    keys [0, n_keys), values [n_keys, n_keys + n_values), then QUERY, PAD."""
    query_tok = n_keys + n_values
    pad_tok = query_tok + 1
    vocab = pad_tok + 1
    seq_len = 2 * (n_pairs + n_revisions + n_queries)

    x = torch.full((bsz, seq_len), pad_tok, dtype=torch.long)
    y = torch.full((bsz, seq_len), IGNORE, dtype=torch.long)
    group = torch.zeros((bsz, seq_len), dtype=torch.long)   # 1 rev, 2 untouched
    immediate = torch.zeros((bsz, seq_len), dtype=torch.bool)

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
        # queries: half on revised keys, half on untouched ones
        untouched = [int(k) for k in keys if int(k) not in rewrite_at]
        revised_l = [int(k) for k in revised]
        order = []
        for i in range(n_queries):
            pool = revised_l if (i % 2 == 0 and revised_l) else (untouched or revised_l)
            order.append(pool[int(torch.randint(len(pool), (1,), generator=generator))])
        # immediate queries first, so their gap to the rewrite is small
        n_imm = int(round(n_queries * immediate_frac))
        for i, k in enumerate(order):
            x[b, pos], x[b, pos + 1] = k, query_tok
            y[b, pos + 1] = n_keys + latest[k]
            if k in rewrite_at:
                group[b, pos + 1] = 1
                immediate[b, pos + 1] = (i < n_imm) and (pos - rewrite_at[k]) <= 2 * IMMEDIATE_GAP
            else:
                group[b, pos + 1] = 2
            pos += 2
    return dict(x=x, y=y, group=group, immediate=immediate, vocab=vocab)


def accuracy(logits, batch):
    """overall / revised / untouched / immediate_revised, in percent."""
    pred = logits.argmax(-1)
    y, g, imm = batch["y"], batch["group"], batch["immediate"]
    scored = y != IGNORE
    hit = (pred == y) & scored
    def pct(mask):
        n = mask.sum().item()
        return 100.0 * (hit & mask).sum().item() / n if n else float("nan")
    return dict(overall=pct(scored), revised=pct(scored & (g == 1)),
                untouched=pct(scored & (g == 2)),
                immediate_revised=pct(scored & (g == 1) & imm),
                later_revised=pct(scored & (g == 1) & ~imm))
