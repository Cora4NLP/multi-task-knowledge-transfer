import logging
from collections import Counter
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from pytorch_ie.core import PyTorchIEModel
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertModel
from typing_extensions import TypeAlias

from src.models.components import TransformerMultiModel


def f1_score(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefHoiEvaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe_simplified:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1_score(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def b_cubed_simplified(clusters, mention_to_gold):
    num, dem = 0, 0
    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)
    return num, dem


def muc_simplified(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4_simplified(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe_simplified(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4_simplified(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    matching = np.transpose(np.asarray(matching))
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea_simplified(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1 :]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()

TRAINING = "train"
VALIDATION = "val"
TEST = "test"


def bucket_distance(offsets):
    """offsets: [num spans1, num spans2]"""
    # 10 semi-logscale bin: 0, 1, 2, 3, 4, (5-7)->5, (8-15)->6, (16-31)->7, (32-63)->8, (64+)->9
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def batch_select(tensor, idx, device=torch.device("cpu")):
    """Do selection per row (first axis)."""
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        selected = torch.squeeze(selected, -1)

    return selected


def attended_antecedent(top_span_emb, top_antecedent_emb, top_antecedent_scores, device):
    num_top_spans = top_span_emb.shape[0]
    top_antecedent_weights = torch.cat(
        [torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1
    )
    top_antecedent_weights = nn.functional.softmax(top_antecedent_weights, dim=1)
    top_antecedent_emb = torch.cat([torch.unsqueeze(top_span_emb, 1), top_antecedent_emb], dim=1)
    refined_span_emb = torch.sum(
        torch.unsqueeze(top_antecedent_weights, 2) * top_antecedent_emb, dim=1
    )  # [num top spans, span emb size]
    return refined_span_emb


def max_antecedent(top_span_emb, top_antecedent_emb, top_antecedent_scores, device):
    num_top_spans = top_span_emb.shape[0]
    top_antecedent_weights = torch.cat(
        [torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1
    )
    top_antecedent_emb = torch.cat([torch.unsqueeze(top_span_emb, 1), top_antecedent_emb], dim=1)
    max_antecedent_idx = torch.argmax(top_antecedent_weights, dim=1, keepdim=True)
    refined_span_emb = batch_select(top_antecedent_emb, max_antecedent_idx, device=device).squeeze(
        1
    )  # [num top spans, span emb size]
    return refined_span_emb


def entity_equalization(
    top_span_emb, top_antecedent_emb, top_antecedent_idx, top_antecedent_scores, device
):
    # Use TF implementation in another repo
    pass


def span_clustering(
    top_span_emb, top_antecedent_idx, top_antecedent_scores, span_attn_ffnn, device
):
    # Get predicted antecedents
    num_top_spans, max_top_antecedents = top_antecedent_idx.shape[0], top_antecedent_idx.shape[1]
    predicted_antecedents = []
    top_antecedent_scores = torch.cat(
        [torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1
    )
    for i, idx in enumerate((torch.argmax(top_antecedent_scores, axis=1) - 1).tolist()):
        if idx < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(top_antecedent_idx[i, idx].item())
    # Get predicted clusters
    predicted_clusters = []
    span_to_cluster_id = [-1] * num_top_spans
    for i, predicted_idx in enumerate(predicted_antecedents):
        if predicted_idx < 0:
            continue
        assert i > predicted_idx, f"span idx: {i}; antecedent idx: {predicted_idx}"
        # Check antecedent's cluster
        antecedent_cluster_id = span_to_cluster_id[predicted_idx]
        if antecedent_cluster_id == -1:
            antecedent_cluster_id = len(predicted_clusters)
            predicted_clusters.append([predicted_idx])
            span_to_cluster_id[predicted_idx] = antecedent_cluster_id
        # Add mention to cluster
        predicted_clusters[antecedent_cluster_id].append(i)
        span_to_cluster_id[i] = antecedent_cluster_id
    if len(predicted_clusters) == 0:
        return top_span_emb

    # Pad clusters
    max_cluster_size = max([len(c) for c in predicted_clusters])
    cluster_sizes = []
    for cluster in predicted_clusters:
        cluster_sizes.append(len(cluster))
        cluster += [0] * (max_cluster_size - len(cluster))
    predicted_clusters_mask = torch.arange(0, max_cluster_size, device=device).repeat(
        len(predicted_clusters), 1
    )
    predicted_clusters_mask = predicted_clusters_mask < torch.tensor(
        cluster_sizes, device=device
    ).unsqueeze(
        1
    )  # [num clusters, max cluster size]
    # Get cluster repr
    predicted_clusters = torch.tensor(predicted_clusters, device=device)
    cluster_emb = top_span_emb[predicted_clusters]  # [num clusters, max cluster size, emb size]
    span_attn = torch.squeeze(span_attn_ffnn(cluster_emb), 2)
    span_attn += torch.log(predicted_clusters_mask.to(torch.float))
    span_attn = nn.functional.softmax(span_attn, dim=1)
    cluster_emb = torch.sum(
        cluster_emb * torch.unsqueeze(span_attn, 2), dim=1
    )  # [num clusters, emb size]
    # Get refined span
    refined_span_emb = []
    for i, cluster_idx in enumerate(span_to_cluster_id):
        if cluster_idx < 0:
            refined_span_emb.append(top_span_emb[i])
        else:
            refined_span_emb.append(cluster_emb[cluster_idx])
    refined_span_emb = torch.stack(refined_span_emb, dim=0)
    return refined_span_emb


def cluster_merging(
    top_span_emb,
    top_antecedent_idx,
    top_antecedent_scores,
    emb_cluster_size,
    cluster_score_ffnn,
    cluster_transform,
    dropout,
    device,
    reduce="mean",
    easy_cluster_first=False,
):
    num_top_spans, max_top_antecedents = top_antecedent_idx.shape[0], top_antecedent_idx.shape[1]
    span_emb_size = top_span_emb.shape[-1]
    max_num_clusters = num_top_spans

    span_to_cluster_id = torch.zeros(
        num_top_spans, dtype=torch.long, device=device
    )  # id 0 as dummy cluster
    cluster_emb = torch.zeros(
        max_num_clusters, span_emb_size, dtype=torch.float, device=device
    )  # [max num clusters, emb size]
    num_clusters = 1  # dummy cluster
    cluster_sizes = torch.ones(max_num_clusters, dtype=torch.long, device=device)

    merge_order = torch.arange(0, num_top_spans)
    if easy_cluster_first:
        max_antecedent_scores, _ = torch.max(top_antecedent_scores, dim=1)
        merge_order = torch.argsort(max_antecedent_scores, descending=True)
    cluster_merging_scores = [None] * num_top_spans

    for i in merge_order.tolist():
        # Get cluster scores
        antecedent_cluster_idx = span_to_cluster_id[top_antecedent_idx[i]]
        antecedent_cluster_emb = cluster_emb[antecedent_cluster_idx]
        # antecedent_cluster_emb = dropout(cluster_transform(antecedent_cluster_emb))

        antecedent_cluster_size = cluster_sizes[antecedent_cluster_idx]
        antecedent_cluster_size = bucket_distance(antecedent_cluster_size)
        cluster_size_emb = dropout(emb_cluster_size(antecedent_cluster_size))

        span_emb = top_span_emb[i].unsqueeze(0).repeat(max_top_antecedents, 1)
        similarity_emb = span_emb * antecedent_cluster_emb
        pair_emb = torch.cat(
            [span_emb, antecedent_cluster_emb, similarity_emb, cluster_size_emb], dim=1
        )  # [max top antecedents, pair emb size]
        cluster_scores = torch.squeeze(cluster_score_ffnn(pair_emb), 1)
        cluster_scores_mask = (antecedent_cluster_idx > 0).to(torch.float)
        cluster_scores *= cluster_scores_mask
        cluster_merging_scores[i] = cluster_scores

        # Get predicted antecedent
        antecedent_scores = top_antecedent_scores[i] + cluster_scores
        max_score, max_score_idx = torch.max(antecedent_scores, dim=0)
        if max_score < 0:
            continue  # Dummy antecedent
        max_antecedent_idx = top_antecedent_idx[i, max_score_idx]

        if not easy_cluster_first:  # Always add span to antecedent's cluster
            # Create antecedent cluster if needed
            antecedent_cluster_id = span_to_cluster_id[max_antecedent_idx]
            if antecedent_cluster_id == 0:
                antecedent_cluster_id = num_clusters
                span_to_cluster_id[max_antecedent_idx] = antecedent_cluster_id
                cluster_emb[antecedent_cluster_id] = top_span_emb[max_antecedent_idx]
                num_clusters += 1
            # Add span to cluster
            span_to_cluster_id[i] = antecedent_cluster_id
            _merge_span_to_cluster(
                cluster_emb, cluster_sizes, antecedent_cluster_id, top_span_emb[i], reduce=reduce
            )
        else:  # current span can be in cluster already
            antecedent_cluster_id = span_to_cluster_id[max_antecedent_idx]
            curr_span_cluster_id = span_to_cluster_id[i]
            if antecedent_cluster_id > 0 and curr_span_cluster_id > 0:
                # Merge two clusters
                span_to_cluster_id[max_antecedent_idx] = curr_span_cluster_id
                _merge_clusters(
                    cluster_emb,
                    cluster_sizes,
                    antecedent_cluster_id,
                    curr_span_cluster_id,
                    reduce=reduce,
                )
            elif curr_span_cluster_id > 0:
                # Merge antecedent to span's cluster
                span_to_cluster_id[max_antecedent_idx] = curr_span_cluster_id
                _merge_span_to_cluster(
                    cluster_emb,
                    cluster_sizes,
                    curr_span_cluster_id,
                    top_span_emb[max_antecedent_idx],
                    reduce=reduce,
                )
            else:
                # Create antecedent cluster if needed
                if antecedent_cluster_id == 0:
                    antecedent_cluster_id = num_clusters
                    span_to_cluster_id[max_antecedent_idx] = antecedent_cluster_id
                    cluster_emb[antecedent_cluster_id] = top_span_emb[max_antecedent_idx]
                    num_clusters += 1
                # Add span to cluster
                span_to_cluster_id[i] = antecedent_cluster_id
                _merge_span_to_cluster(
                    cluster_emb,
                    cluster_sizes,
                    antecedent_cluster_id,
                    top_span_emb[i],
                    reduce=reduce,
                )

    cluster_merging_scores = torch.stack(cluster_merging_scores, dim=0)
    return cluster_merging_scores


def _merge_span_to_cluster(cluster_emb, cluster_sizes, cluster_to_merge_id, span_emb, reduce):
    cluster_size = cluster_sizes[cluster_to_merge_id].item()
    if reduce == "mean":
        cluster_emb[cluster_to_merge_id] = (
            cluster_emb[cluster_to_merge_id] * cluster_size + span_emb
        ) / (cluster_size + 1)
    elif reduce == "max":
        cluster_emb[cluster_to_merge_id], _ = torch.max(
            torch.stack([cluster_emb[cluster_to_merge_id], span_emb]), dim=0
        )
    else:
        raise ValueError("reduce value is invalid: %s" % reduce)
    cluster_sizes[cluster_to_merge_id] += 1


def _merge_clusters(cluster_emb, cluster_sizes, cluster1_id, cluster2_id, reduce):
    """Merge cluster1 to cluster2."""
    cluster1_size, cluster2_size = (
        cluster_sizes[cluster1_id].item(),
        cluster_sizes[cluster2_id].item(),
    )
    if reduce == "mean":
        cluster_emb[cluster2_id] = (
            cluster_emb[cluster1_id] * cluster1_size + cluster_emb[cluster2_id] * cluster2_size
        ) / (cluster1_size + cluster2_size)
    elif reduce == "max":
        cluster_emb[cluster2_id] = torch.max(cluster_emb[cluster1_id], cluster_emb[cluster2_id])
    else:
        raise ValueError("reduce value is invalid: %s" % reduce)
    cluster_sizes[cluster2_id] += cluster_sizes[cluster1_id]


class CorefHoiModelInputs(TypedDict):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    speaker_ids: torch.Tensor
    sentence_len: torch.Tensor
    genre: torch.Tensor
    sentence_map: torch.Tensor


class CorefHoiModelTargets(TypedDict):
    gold_starts: torch.Tensor
    gold_ends: torch.Tensor
    gold_mention_cluster_map: torch.Tensor


CorefHoiModelStepBatchEncoding: TypeAlias = Tuple[
    CorefHoiModelInputs, Optional[CorefHoiModelTargets]
]


class CorefHoiModelBatchOutput(TypedDict):
    candidate_starts: torch.Tensor
    candidate_ends: torch.Tensor
    candidate_mention_scores: torch.Tensor
    top_span_starts: torch.Tensor
    top_span_ends: torch.Tensor
    top_antecedent_idx: torch.Tensor
    top_antecedent_scores: torch.Tensor


class CorefHoiModelPrediction(TypedDict):
    spans: List[Tuple[int, int]]
    antecedents: List[int]
    clusters: List[Tuple[Tuple[int, int], ...]]


class MultiModelCorefHoiInputs(TypedDict):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    speaker_ids: torch.Tensor
    sentence_len: torch.Tensor
    genre: torch.Tensor
    sentence_map: torch.Tensor


class MultiModelCorefHoiTargets(TypedDict):
    gold_starts: torch.Tensor
    gold_ends: torch.Tensor
    gold_mention_cluster_map: torch.Tensor


MultiModelCorefHoiStepBatchEncoding: TypeAlias = Tuple[
    MultiModelCorefHoiInputs, Optional[MultiModelCorefHoiTargets]
]


class MultiModelCorefHoiBatchOutput(TypedDict):
    candidate_starts: torch.Tensor
    candidate_ends: torch.Tensor
    candidate_mention_scores: torch.Tensor
    top_span_starts: torch.Tensor
    top_span_ends: torch.Tensor
    top_antecedent_idx: torch.Tensor
    top_antecedent_scores: torch.Tensor


class MultiModelCorefHoiPrediction(TypedDict):
    spans: List[Tuple[int, int]]
    antecedents: List[int]
    clusters: List[Tuple[Tuple[int, int], ...]]


class CorefHoiF1(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.evaluators = [
            CorefHoiEvaluator(m) for m in (muc_simplified, b_cubed_simplified, ceafe_simplified)
        ]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def compute(self, reset=True):
        result = self.get_f1()
        if reset:
            self.reset()
        return result

    def forward(self, predictions, targets):
        predicted_clusters, mention_to_predicted = predictions
        gold_clusters, mention_to_gold = targets
        self.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return self.compute(reset=False)
