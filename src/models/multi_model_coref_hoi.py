import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from pytorch_ie.core import PyTorchIEModel
from torch.optim.lr_scheduler import LambdaLR

from src.metrics.coref import CorefHoiF1
from src.models.components import TransformerMultiModel
from src.models.components.coref import (
    CorefHoiModelBatchOutput,
    CorefHoiModelInputs,
    CorefHoiModelPrediction,
    CorefHoiModelStepBatchEncoding,
    attended_antecedent,
    batch_select,
    bucket_distance,
    cluster_merging,
    entity_equalization,
    max_antecedent,
    span_clustering,
)

TRAINING = "train"
VALIDATION = "val"
TEST = "test"


logger = logging.getLogger()


@PyTorchIEModel.register()
class MultiModelCorefHoiModel(PyTorchIEModel):
    def __init__(
        self,
        pretrained_models: Dict[str, str],
        genres: List[str],
        max_segment_len: int,
        max_span_width: int,
        loss_type: str,
        coref_depth: int,
        higher_order: str,
        fine_grained: bool,
        dropout_rate: float,
        use_features: bool,
        feature_emb_size: int,
        use_metadata: bool,
        use_segment_distance: bool,
        use_width_prior: bool,
        use_distance_prior: bool,
        max_training_sentences: int,
        model_heads,
        ffnn_size: int,
        ffnn_depth: int,
        cluster_ffnn_size: int,
        cluster_dloss,
        cluster_reduce,
        easy_cluster_first,
        false_new_delta,
        max_num_extracted_spans,
        max_top_antecedents,
        mention_loss_coef,
        top_span_ratio,
        bert_learning_rate,
        adam_weight_decay,
        task_learning_rate,
        adam_eps,
        warmup_ratio,
        num_genres=None,
        aggregate: str = "mean",
        freeze_models: Optional[List[str]] = None,
        truncate_models: Optional[Dict[str, int]] = None,
        pretrained_default_config: Optional[str] = None,
        pretrained_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        normalize_embeddings: bool = False,
        cossim_target_embed_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if model_name is not None:
            logger.warning(
                "The `model_name` argument is deprecated and will be removed in a future version. "
                "Please use `pretrained_default_config` instead."
            )
            pretrained_default_config = model_name
        self.save_hyperparameters(ignore=["model_name"])

        # need to be disabled because we have multiple optimizers
        self.automatic_optimization = False

        self.base_models = TransformerMultiModel(
            pretrained_models=pretrained_models,
            pretrained_default_config=pretrained_default_config,
            pretrained_configs=pretrained_configs,
            load_model_weights=not self.is_from_pretrained,
            aggregate=aggregate,
            freeze_models=freeze_models,
            truncate_models=truncate_models,
            normalize_embeddings=normalize_embeddings,
            cossim_target_embed_key=cossim_target_embed_key,
        )

        self.num_genres = num_genres if num_genres else len(genres)
        self.max_seg_len = max_segment_len
        self.max_span_width = max_span_width
        assert loss_type in ["marginalized", "hinge"]
        if coref_depth > 1 or higher_order == "cluster_merging":
            assert fine_grained  # Higher-order is in slow fine-grained scoring

        self.max_training_sentences = max_training_sentences
        self.cluster_dloss = cluster_dloss
        self.cluster_reduce = cluster_reduce
        self.coref_depth = coref_depth
        self.easy_cluster_first = easy_cluster_first
        self.false_new_delta = false_new_delta
        self.fine_grained = fine_grained
        self.higher_order = higher_order
        self.loss_type = loss_type
        self.max_num_extracted_spans = max_num_extracted_spans
        self.max_top_antecedents = max_top_antecedents
        self.max_training_sentences = max_training_sentences
        self.mention_loss_coef = mention_loss_coef
        self.model_heads = model_heads
        self.top_span_ratio = top_span_ratio
        self.use_distance_prior = use_distance_prior
        self.use_features = use_features
        self.use_metadata = use_metadata
        self.use_segment_distance = use_segment_distance
        self.use_width_prior = use_width_prior

        self.bert_learning_rate = bert_learning_rate
        self.adam_weight_decay = adam_weight_decay
        self.task_learning_rate = task_learning_rate
        self.adam_eps = adam_eps
        self.warmup_ratio = warmup_ratio
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # Model
        self.dropout = nn.Dropout(p=dropout_rate)

        self.feature_emb_size = feature_emb_size
        self.bert_emb_size = self.base_models.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if use_features:
            self.span_emb_size += feature_emb_size
        self.pair_emb_size = self.span_emb_size * 3
        if use_metadata:
            self.pair_emb_size += 2 * feature_emb_size
        if use_features:
            self.pair_emb_size += feature_emb_size
        if use_segment_distance:
            self.pair_emb_size += feature_emb_size

        self.emb_span_width = self.make_embedding(self.max_span_width) if use_features else None
        self.emb_span_width_prior = (
            self.make_embedding(self.max_span_width) if use_width_prior else None
        )
        self.emb_antecedent_distance_prior = (
            self.make_embedding(10) if use_distance_prior else None
        )
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if use_metadata else None
        self.emb_segment_distance = (
            self.make_embedding(max_training_sentences) if use_segment_distance else None
        )
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = (
            self.make_embedding(10) if higher_order == "cluster_merging" else None
        )

        self.mention_token_attn = (
            self.make_ffnn(self.bert_emb_size, 0, output_size=1) if model_heads else None
        )
        self.span_emb_score_ffnn = self.make_ffnn(
            self.span_emb_size, [ffnn_size] * ffnn_depth, output_size=1
        )
        self.span_width_score_ffnn = (
            self.make_ffnn(
                feature_emb_size,
                [ffnn_size] * ffnn_depth,
                output_size=1,
            )
            if use_width_prior
            else None
        )
        self.coarse_bilinear = self.make_ffnn(
            self.span_emb_size, 0, output_size=self.span_emb_size
        )
        self.antecedent_distance_score_ffnn = (
            self.make_ffnn(feature_emb_size, 0, output_size=1) if use_distance_prior else None
        )
        self.coref_score_ffnn = (
            self.make_ffnn(self.pair_emb_size, [ffnn_size] * ffnn_depth, output_size=1)
            if fine_grained
            else None
        )

        self.gate_ffnn = (
            self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size)
            if coref_depth > 1
            else None
        )
        self.span_attn_ffnn = (
            self.make_ffnn(self.span_emb_size, 0, output_size=1)
            if higher_order == "span_clustering"
            else None
        )
        self.cluster_score_ffnn = (
            self.make_ffnn(
                3 * self.span_emb_size + feature_emb_size,
                [cluster_ffnn_size] * ffnn_depth,
                output_size=1,
            )
            if higher_order == "cluster_merging"
            else None
        )

        self.update_steps = 0  # Internal use for debug
        self.debug = True

        self.f1 = nn.ModuleDict(
            {f"stage_{stage}": CorefHoiF1() for stage in [TRAINING, VALIDATION, TEST]}
        )

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.feature_emb_size)
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith("base_models"):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, **batch):
        return self.get_predictions_and_loss(**batch)

    def get_predictions_and_loss(
        self,
        input_ids,
        input_mask,
        speaker_ids,
        sentence_len,
        genre,
        sentence_map,
        gold_starts=None,
        gold_ends=None,
        gold_mention_cluster_map=None,
    ) -> Tuple[CorefHoiModelBatchOutput, Optional[torch.Tensor], Optional[Dict]]:
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Only support batch size 1 for now"
        # use just first example in batch
        input_ids = input_ids[0]
        input_mask = input_mask[0]
        speaker_ids = speaker_ids[0]
        sentence_len = sentence_len[0]
        genre = genre[0]
        sentence_map = sentence_map[0]
        """Model and input are already on the device."""
        device = self.device

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True
            # use just first example in batch
            gold_starts = gold_starts[0]
            gold_ends = gold_ends[0]
            gold_mention_cluster_map = gold_mention_cluster_map[0]

        # get token emb

        # get the sequence logits aggregated over all models
        model_inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        embedded_inputs, embed_cossim_dict = self.base_models(
            **model_inputs
        )  # [seg length, num tokens, emb size]

        input_mask = input_mask.to(torch.bool)  # [seg length, num tokens]
        mention_doc = embedded_inputs[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]

        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(
            1, self.max_span_width
        )
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[
            torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))
        ]
        candidate_mask = (candidate_ends < num_words) & (
            candidate_start_sent_idx == candidate_end_sent_idx
        )
        candidate_starts, candidate_ends = (
            candidate_starts[candidate_mask],
            candidate_ends[candidate_mask],
        )  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            same_start = torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0)
            same_end = torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0)
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(
                torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                same_span.to(torch.float),
            )
            candidate_labels = torch.squeeze(
                candidate_labels.to(torch.long), 0
            )  # [num candidates]; non-gold span has label 0

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if self.use_features:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(
            num_candidates, 1
        )
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
            candidate_tokens <= torch.unsqueeze(candidate_ends, 1)
        )
        if self.model_heads:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(
                num_words, dtype=torch.float, device=device
            )  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(
            candidate_tokens_mask.to(torch.float)
        ) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if self.use_width_prior:
            width_score = torch.squeeze(
                self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1
            )
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(
            candidate_mention_scores, descending=True
        ).tolist()
        candidate_starts_cpu, candidate_ends_cpu = (
            candidate_starts.tolist(),
            candidate_ends.tolist(),
        )
        num_top_spans = int(min(self.max_num_extracted_spans, self.top_span_ratio * num_words))
        selected_idx_cpu = self._extract_top_spans(
            candidate_idx_sorted_by_score, candidate_starts_cpu, candidate_ends_cpu, num_top_spans
        )
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        top_span_starts, top_span_ends = (
            candidate_starts[selected_idx],
            candidate_ends[selected_idx],
        )
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, self.max_top_antecedents)
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(
            top_span_range, 0
        )
        antecedent_mask = antecedent_offsets >= 1
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(
            top_span_mention_scores, 0
        )
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if self.use_distance_prior:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(
                    self.dropout(self.emb_antecedent_distance_prior.weight)
                ),
                1,
            )
            bucketed_distance = bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(
            pairwise_fast_scores, k=max_top_antecedents
        )
        top_antecedent_mask = batch_select(
            antecedent_mask, top_antecedent_idx, device
        )  # [num top spans, max top antecedents]
        top_antecedent_offsets = batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if self.fine_grained:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = (
                None,
                None,
                None,
                None,
            )
            if self.use_metadata:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = (
                    torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                )
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(
                    num_top_spans, max_top_antecedents, 1
                )
            if self.use_segment_distance:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = (
                    torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                )
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = (
                    torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                )
                top_antecedent_seg_distance = torch.clamp(
                    top_antecedent_seg_distance, 0, self.max_training_sentences - 1
                )
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if self.use_features:  # Antecedent distance
                top_antecedent_distance = bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(
                    top_antecedent_distance
                )

            for depth in range(self.coref_depth):
                top_antecedent_emb = top_span_emb[
                    top_antecedent_idx
                ]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if self.use_metadata:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if self.use_segment_distance:
                    feature_list.append(seg_distance_emb)
                if self.use_features:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat(
                    [target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2
                )
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if self.higher_order == "cluster_merging":
                    cluster_merging_scores = cluster_merging(
                        top_span_emb,
                        top_antecedent_idx,
                        top_pairwise_scores,
                        self.emb_cluster_size,
                        self.cluster_score_ffnn,
                        None,
                        self.dropout,
                        device=device,
                        reduce=self.cluster_reduce,
                        easy_cluster_first=self.easy_cluster_first,
                    )
                    break
                elif depth != self.coref_depth - 1:
                    if self.higher_order == "attended_antecedent":
                        refined_span_emb = attended_antecedent(
                            top_span_emb, top_antecedent_emb, top_pairwise_scores, device
                        )
                    elif self.higher_order == "max_antecedent":
                        refined_span_emb = max_antecedent(
                            top_span_emb, top_antecedent_emb, top_pairwise_scores, device
                        )
                    elif self.higher_order == "entity_equalization":
                        refined_span_emb = entity_equalization(
                            top_span_emb,
                            top_antecedent_emb,
                            top_antecedent_idx,
                            top_pairwise_scores,
                            device,
                        )
                    elif self.higher_order == "span_clustering":
                        refined_span_emb = span_clustering(
                            top_span_emb,
                            top_antecedent_idx,
                            top_pairwise_scores,
                            self.span_attn_ffnn,
                            device,
                        )

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = (
                        gate * refined_span_emb + (1 - gate) * top_span_emb
                    )  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if self.fine_grained and self.higher_order == "cluster_merging":
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat(
                [torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1
            )  # [num top spans, max top antecedents + 1]
            return (
                CorefHoiModelBatchOutput(
                    candidate_starts=candidate_starts,
                    candidate_ends=candidate_ends,
                    candidate_mention_scores=candidate_mention_scores,
                    top_span_starts=top_span_starts,
                    top_span_ends=top_span_ends,
                    top_antecedent_idx=top_antecedent_idx,
                    top_antecedent_scores=top_antecedent_scores,
                ),
                None,
                embed_cossim_dict,
            )

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (
            top_antecedent_mask.to(torch.long) - 1
        ) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = top_antecedent_cluster_ids == torch.unsqueeze(
            top_span_cluster_ids, 1
        )
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat(
            [torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1
        )
        if self.loss_type == "marginalized":
            log_marginalized_antecedent_scores = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)),
                dim=1,
            )
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif self.loss_type == "hinge":
            top_antecedent_mask = torch.cat(
                [
                    torch.ones(num_top_spans, 1, dtype=torch.bool, device=device),
                    top_antecedent_mask,
                ],
                dim=1,
            )
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(
                top_antecedent_scores, dim=1
            )
            gold_antecedent_scores = top_antecedent_scores + torch.log(
                top_antecedent_gold_labels.to(torch.float)
            )
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(
                gold_antecedent_scores, dim=1
            )
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = highest_antecedent_idx == highest_gold_antecedent_idx
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(
                dummy_antecedent_labels.squeeze()
            )
            delta = ((3 - self.false_new_delta) / 2) * torch.ones(
                num_top_spans, dtype=torch.float, device=device
            )
            delta -= (1 - self.false_new_delta) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        if self.mention_loss_coef:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = (
                -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * self.mention_loss_coef
            )
            loss_mention += (
                -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores)))
                * self.mention_loss_coef
            )
            loss += loss_mention

        if self.higher_order == "cluster_merging":
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat(
                [torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1
            )
            log_marginalized_antecedent_scores2 = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)),
                dim=1,
            )
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if self.cluster_dloss:
                loss += loss_cm
            else:
                loss = loss_cm

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info("---------debug step: %d---------" % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info(
                    "spans/gold: %d/%d; ratio: %.2f"
                    % (
                        num_top_spans,
                        (top_span_cluster_ids > 0).sum(),
                        (top_span_cluster_ids > 0).sum() / num_top_spans,
                    )
                )
                if self.mention_loss_coef:
                    logger.info("mention loss: %.4f" % loss_mention)
                if self.loss_type == "marginalized":
                    logger.info(
                        "norm/gold: %.4f/%.4f"
                        % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores))
                    )
                else:
                    logger.info("loss: %.4f" % loss)
        self.update_steps += 1
        return (
            CorefHoiModelBatchOutput(
                candidate_starts=candidate_starts,
                candidate_ends=candidate_ends,
                candidate_mention_scores=candidate_mention_scores,
                top_span_starts=top_span_starts,
                top_span_ends=top_span_ends,
                top_antecedent_idx=top_antecedent_idx,
                top_antecedent_scores=top_antecedent_scores,
            ),
            loss,
            embed_cossim_dict,
        )

    def _extract_top_spans(
        self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans
    ):
        """Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of
        loop."""
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(
            selected_candidate_idx, key=lambda idx: (candidate_starts[idx], candidate_ends[idx])
        )
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += [selected_candidate_idx[0]] * (
                num_top_spans - len(selected_candidate_idx)
            )
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """CPU list input."""
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """CPU list input."""
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f"span idx: {i}; antecedent idx: {predicted_idx}"
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(
        self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator
    ):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(
            span_starts, span_ends, antecedent_idx, antecedent_scores
        )
        mention_to_predicted = {
            m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()
        }
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def predict(
        self,
        inputs: CorefHoiModelInputs,
        **kwargs,
    ) -> CorefHoiModelPrediction:
        predictions: CorefHoiModelBatchOutput = self(**inputs)[0]
        span_starts = predictions["top_span_starts"].cpu().tolist()
        span_ends = predictions["top_span_ends"].cpu().tolist()
        clusters, mention_to_cluster_id, antecedents = self.get_predicted_clusters(
            span_starts=span_starts,
            span_ends=span_ends,
            antecedent_idx=predictions["top_antecedent_idx"].cpu().tolist(),
            antecedent_scores=predictions["top_antecedent_scores"].cpu().tolist(),
        )
        spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
        return CorefHoiModelPrediction(spans=spans, antecedents=antecedents, clusters=clusters)

    def step(
        self,
        stage: str,
        batch: CorefHoiModelStepBatchEncoding,
        batch_idx: int,
    ):
        inputs, targets = batch
        # check the target content
        assert targets is not None, "target has to be available for training"
        assert set(targets) == {"gold_starts", "gold_ends", "gold_mention_cluster_map"}

        batch_predictions, loss, embed_cossim_dict = self(**inputs, **targets)
        cluster2mentions: Dict[int, List[List[int]]] = dict()
        for mention_idx, m_cluster in enumerate(
            targets["gold_mention_cluster_map"][0].cpu().detach().numpy().tolist()
        ):
            if not (m_cluster in cluster2mentions):
                cluster2mentions[m_cluster] = []
            cluster2mentions[m_cluster].append(
                [
                    targets["gold_starts"][0].cpu().detach().numpy().tolist()[mention_idx],
                    targets["gold_ends"][0].cpu().detach().numpy().tolist()[mention_idx],
                ]
            )
        gold_cluster_list = []
        for cluster in cluster2mentions:
            gold_cluster_list.append(cluster2mentions[cluster])
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_cluster_list]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}

        span_starts = batch_predictions["top_span_starts"].cpu().detach().numpy()
        span_ends = batch_predictions["top_span_ends"].cpu().detach().numpy()
        antecedent_idx = batch_predictions["top_antecedent_idx"].cpu().detach().numpy()
        antecedent_scores = batch_predictions["top_antecedent_scores"].cpu().detach().numpy()
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(
            span_starts, span_ends, antecedent_idx, antecedent_scores
        )
        mention_to_predicted = {
            m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()
        }

        predictions_tuple = (predicted_clusters, mention_to_predicted)
        targets_tuple = (gold_clusters, mention_to_gold)
        f1_value = self.f1[f"stage_{stage}"](predictions_tuple, targets_tuple)

        # show loss on each step only during training
        self.log(f"{stage}/f1", f1_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)
        per_batch_embed_cossim_dict = dict()
        for cossim_model_name in embed_cossim_dict:
            # add batch_idx
            per_batch_key = str(batch_idx) + "/" + cossim_model_name
            per_batch_embed_cossim_dict[per_batch_key] = embed_cossim_dict[cossim_model_name]

            self.log(
                f"{stage}/cossim: {per_batch_key}",
                per_batch_embed_cossim_dict[per_batch_key],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def training_step(self, batch: CorefHoiModelStepBatchEncoding, batch_idx: int):
        # we need to implement the optimization by ourself because automatic optimization is not
        # possible with multiple optimizers

        for opt in self.optimizers():
            opt.zero_grad()

        loss = self.step(stage=TRAINING, batch=batch, batch_idx=batch_idx)
        self.manual_backward(loss)

        for opt in self.optimizers():
            # clip gradients
            if self.gradient_clip_val is not None:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm=self.gradient_clip_algorithm,
                )
            # optimizer step
            opt.step()

        for schedulers in self.lr_schedulers():
            schedulers.step()

        return loss

    def validation_step(self, batch: CorefHoiModelStepBatchEncoding, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch, batch_idx=batch_idx)

    def test_step(self, batch: CorefHoiModelStepBatchEncoding, batch_idx: int):
        return self.step(stage=TEST, batch=batch, batch_idx=batch_idx)

    def on_train_epoch_end(self):
        self.epoch_end(stage=TRAINING)

    def on_validation_epoch_end(self):
        self.epoch_end(stage=VALIDATION)

    def on_test_epoch_end(self):
        self.epoch_end(stage=TEST)

    def epoch_end(self, stage):
        f1_value = self.f1[f"stage_{stage}"].compute(reset=True)
        self.log(f"{stage}/f1", f1_value, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param, task_param = self.get_params(named=True)
        grouped_bert_param = [
            {
                "params": [
                    p
                    for n, p in bert_param
                    if not any(no_decay_infix in n for no_decay_infix in no_decay)
                ],
                "lr": self.bert_learning_rate,
                "weight_decay": self.adam_weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in bert_param
                    if any(no_decay_infix in n for no_decay_infix in no_decay)
                ],
                "lr": self.bert_learning_rate,
                "weight_decay": 0.0,
            },
        ]
        optimizers = [
            torch.optim.AdamW(grouped_bert_param, lr=self.bert_learning_rate, eps=self.adam_eps),
            torch.optim.Adam(
                self.get_params()[1], lr=self.task_learning_rate, eps=self.adam_eps, weight_decay=0
            ),
        ]

        # Only warm up bert lr
        total_update_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_update_steps * self.warmup_ratio)

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_update_steps - current_step)
                / float(max(1, total_update_steps - warmup_steps)),
            )

        def lr_lambda_task(current_step):
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps))
            )

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task),
        ]
        return optimizers, schedulers
