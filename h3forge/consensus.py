"""Use disagreement between H3 video windows to select cleaner context for a second prediction."""
from itertools import combinations, pairwise
from threading import Lock

import torch


CALLS_KEY = "h3forge_consensus_calls"


class ConsensusCalls:
    def __init__(self):
        self.first = 0
        self.second = 0
        self._lock = Lock()

    def admit(self, first, requested, fraction):
        # Model-option copies on sampling workers share this run's counter.
        # Reserve all readers together so a recurring seam cannot starve one side.
        with self._lock:
            self.first += first
            if self.second + requested > fraction * self.first:
                return False
            self.second += requested
            return True


class ConsensusStep:
    def __init__(self, writes):
        # Split at actual write boundaries (including rescued beats). Each shared
        # piece stores one prediction per writer, even when three windows overlap.
        edges = sorted({edge for pieces in writes for piece in pieces for edge in piece})
        self.pieces = [[] for _ in writes]
        self.overlaps = {}
        pairs = {}
        for lo, hi in pairwise(edges):
            owners = [i for i, pieces in enumerate(writes) if any(a <= lo < b for a, b in pieces)]
            for i in owners:
                self.pieces[i].append((lo, hi))
            if len(owners) < 2:
                continue
            self.overlaps[lo, hi] = {}
            for pair in combinations(owners, 2):
                spans = pairs.setdefault(pair, [])
                if not spans or spans[-1][-1][1] != lo:
                    spans.append([])
                spans[-1].append((lo, hi))
        self.pairs = pairs
        self.weights = {}

    def record(self, i, start, prediction, weight):
        self.weights[i] = weight
        for lo, hi in self.pieces[i]:
            if (lo, hi) in self.overlaps:
                self.overlaps[lo, hi][i] = prediction[:, :, lo-start:hi-start].clone()

    def disagreements(self, fused):
        """One norm ratio per window pair and contiguous co-written interval."""
        scores, spans = [], []
        for (i, j), intervals in self.pairs.items():
            for pieces in intervals:
                lo, hi = pieces[0][0], pieces[-1][1]
                squared = []
                for piece in pieces:
                    preds = self.overlaps[piece]
                    squared.append((preds[i].float() - preds[j].float()).square().sum())
                diff = sum(squared).sqrt()
                norm = torch.linalg.vector_norm(fused[:, :, lo:hi])
                # Zero/zero is agreement. Nonzero/zero is exact cancellation.
                scores.append(torch.where(diff == 0, 0.0, diff / norm))
                spans.append((lo, hi))
        return list(zip(spans, torch.stack(scores).tolist())) if scores else []

    def repredict(self, plan, flagged, calls, video, mask, fused, sigma, policy, predict, denominator):
        # The same pass-one consensus conditions every reader. No replay sees
        # another replay's output, including where three windows share rows.
        readers = {}
        for i, (_, start, end, *_) in enumerate(plan):
            shared = [(max(lo, start), min(hi, end)) for lo, hi in flagged if lo < end and hi > start]
            if shared:
                readers[i] = shared
        if not calls.admit(len(plan), len(readers), policy.consensus_max_fraction):
            return

        replaced = set()
        lead = policy.consensus_lead
        for i, shared in readers.items():
            start, end = plan[i][1:3]
            local = video[:, :, start:end].clone()
            local_mask = (torch.ones_like(local[:, :1], dtype=torch.float32) if mask is None
                          else mask[:, :, start:end].float().clone())
            for lo, hi in shared:
                rows = slice(lo-start, hi-start)
                # x' = (1-sigma')*x0_c + sigma'*noise_c = x + (sigma'-sigma)*v.
                # Existing masked inputs carry m*sigma; pinned cells stay pinned.
                m = local_mask[:, :, rows]
                local[:, :, rows] = local[:, :, rows].float() + ((lead - 1) * sigma) * m * fused[:, :, lo:hi]
                m.mul_(lead)
            (prediction, _), *_ = predict(i, local, local_mask)
            for lo, hi in self.pieces[i]:
                if any(a <= lo < b for a, b in flagged):
                    continue
                rows = prediction[:, :, lo-start:hi-start]
                if (lo, hi) in self.overlaps:
                    self.overlaps[lo, hi][i] = rows.clone()
                    replaced.add((lo, hi))
                else:
                    fused[:, :, lo:hi] = rows

        # Rebuild quiet overlaps in pass-one order from the latest prediction
        # of each writer. An unaffected neighbour keeps its original vote.
        for lo, hi in replaced:
            acc = torch.zeros_like(fused[:, :, lo:hi])
            for i, rows in self.overlaps[lo, hi].items():
                start = plan[i][1]
                acc.add_(rows.float() * self.weights[i][:, :, lo-start:hi-start])
            fused[:, :, lo:hi] = acc / denominator[:, :, lo:hi]
