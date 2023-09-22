import logging
from copy import copy
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module, ModuleDict
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class ConcatAggregator(Module):
    def __init__(self, input_size: int, num_models: int):
        super().__init__()
        self.linear = torch.nn.Linear(num_models * input_size, input_size)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.cat(list(x.values()), dim=-1)
        return self.linear(stacked)


def aggregate_mean(x: Dict[str, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(x.values()), dim=-1)
    aggregated = torch.mean(stacked, dim=-1)
    return aggregated


def aggregate_sum(x: Dict[str, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(x.values()), dim=-1)
    aggregated = torch.sum(stacked, dim=-1)
    return aggregated


class AggregatorAttentionLogger(Module):
    def __init__(self, model_ids: List[str]):
        super().__init__()
        self.model_ids = model_ids

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_flat = x.view(-1, len(self.model_ids))
        return {model_id: x_flat[:, i] for i, model_id in enumerate(self.model_ids)}


class AttentionBasedAggregator(Module):
    """This module implements an attention-based aggregation mechanism for multiple transformer
    model outputs.

    The module expects a dictionary of model outputs as input. The keys of the dictionary are the model ids and the
    values are the model outputs. The model outputs are expected to be tensors of shape (batch_size, num_tokens,
    hidden_size). The module then calculates attention weights for each model output and aggregates the outputs
    using these weights. The attention weights are calculated using a query vector that comes from one of the model
    outputs (the target model) and key vectors that come from all model outputs. The query and key vectors are
    calculated using linear layers. The query vector is multiplied with the transpose of the key vectors to get
    scores that are then normalized using a softmax function to get the attention weights. The attention weights are
    then used to weight the projected model outputs (values) and the weighted values are summed up to get the final
    output. The module can be configured to use different parts of the model outputs as query and key vectors. The
    following query and key modes are supported (they can be set individually via the `mode_query` and `mode_keys`
    parameters or together via the `mode` parameter in the format `mode_query2mode_keys`):

    - token: use the token embeddings as input for the linear layer to get the query or key vectors.
    - cls: broadcast the cls embedding along the token dimension and use it as input for the linear layer to get
        the query or key vectors.
    - constant: broadcast a constant tensor along the batch and token dimension and use it as query or key vectors.

    Args:
        input_size: The size of the input embeddings.
        n_models: The number of model outputs to aggregate.
        hidden_size: The size of the query and key vectors.
        output_size: The size of the output embeddings. If None (default), the input_size is used.
        query_idx: The index of the model to use as the query, i.e. the target model. If a string is provided,
            it is the key of the model in the input dictionary. If an int is provided, it is the index of the model
            *values* in the input dictionary. Defaults to 0.
        mode: The mode to use for the query and key vectors. Should be a string in the format `mode_query2mode_keys`.
            Defaults to `token2token`.
        mode_query: The mode to use for the query vector. If defined, overrides the respective part of the `mode`
            parameter.
        mode_keys: The mode to use for the key vectors. If defined, overrides the respective part of the `mode`
            parameter.
        project_target_query: Whether to project the target model embeddings with a linear query layer to get the
            query vector. If disabled, the target model embeddings are used directly as the query vector. Defaults to
            True.
        project_target_key: Whether to project the target model embeddings with a linear key layer. If disabled, the
            target model embeddings are used directly as the key vectors. Defaults to True.
        project_target_value: Whether to project the target model embeddings with a linear value layer. If disabled,
            the target model embeddings are used directly as the value vectors. Defaults to True.
        reuse_target_query_as_key: Whether to reuse the target model query vector as the key vector. Defaults to True.

    Returns:
        The aggregated model output of shape (batch_size, num_tokens, output_size).
    """

    def __init__(
        self,
        input_size: int,
        model_ids: List[str],
        hidden_size: int = 128,
        output_size: Optional[int] = None,
        query_idx: Union[int, str] = 0,
        mode: str = "token2token",
        mode_query: Optional[str] = None,
        mode_keys: Optional[str] = None,
        project_target_query: bool = True,
        project_target_key: bool = True,
        project_target_value: bool = True,
        reuse_target_query_as_key: bool = True,
    ):
        super().__init__()
        # the index of the model to use as the query. If a string is provided, it is the key of the model in
        # the input dictionary. If an int is provided, it is the index of the model *values* in the input dictionary.
        self.query_idx = query_idx
        self.mode_query = mode_query or mode.split("2")[0]
        self.mode_keys = mode_keys or mode.split("2")[1]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.n_models = len(model_ids)
        self.attention_logger = AggregatorAttentionLogger(model_ids)

        if not (project_target_query and project_target_key) and self.hidden_size != input_size:
            logger.warning(
                f"We do not project the target embeddings with a query or key layer, "
                f"so hidden_size [{self.hidden_size}] is overwritten with input_size [{input_size}]."
            )
            self.hidden_size = input_size

        if not project_target_value and self.output_size != self.input_size:
            logger.warning(
                f"We do not project the target values, so output_size [{self.output_size}] is overwritten "
                f"with input_size [{self.input_size}]."
            )
            self.output_size = self.input_size

        # we need only one query projection (just for the target model embeddings)
        if project_target_query:
            self.query = torch.nn.Linear(self.input_size, self.hidden_size)
        else:
            self.query = torch.nn.Identity()

        # we need individual key projections for all model embeddings
        if reuse_target_query_as_key:
            target_key = self.query
        else:
            if project_target_key:
                target_key = torch.nn.Linear(self.input_size, self.hidden_size)
            else:
                target_key = torch.nn.Identity()
        self.keys = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.input_size, self.hidden_size)
                if i != query_idx
                else target_key
                for i in range(self.n_models)
            ]
        )

        # we need individual value projections for all model embeddings
        self.values = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.input_size, self.output_size)
                if i != query_idx or project_target_value
                else torch.nn.Identity()
                for i in range(self.n_models)
            ]
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_key = (
            self.query_idx if isinstance(self.query_idx, str) else list(x.keys())[self.query_idx]
        )
        # (batch_size, num_tokens, output_size, num_models)
        values = torch.stack([k(v) for k, v in zip(self.values, x.values())], dim=-1)
        batch_size, num_tokens = values.shape[:2]

        if self.mode_query == "token":
            # (batch_size, num_tokens, hidden_size)
            query = self.query(x[query_key])
        elif self.mode_query == "cls":
            # (batch_size, hidden_size)
            query_cls = self.query(x[query_key][:, 0, :])
            # (batch_size, num_tokens, hidden_size)
            query = query_cls.unsqueeze(dim=1).expand(-1, values.shape[1], -1)
        elif self.mode_query == "constant":
            # passing a tensor of zeros is fine because we still have the bias of the linear layer
            # (hidden_size,)
            query_constant = self.query(torch.zeros(self.query.in_features, device=values.device))
            # (batch_size, num_tokens, hidden_size)
            query = (
                query_constant.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, num_tokens, -1)
            )
        else:
            raise ValueError(f"Unknown query mode: {self.mode_query}")

        if self.mode_keys == "token":
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = torch.stack([k(v) for k, v in zip(self.keys, x.values())], dim=-1)
        elif self.mode_keys == "cls":
            # (batch_size, hidden_size, num_models)
            keys_cls = torch.stack([k(v[:, 0, :]) for k, v in zip(self.keys, x.values())], dim=-1)
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = keys_cls.unsqueeze(dim=1).expand(-1, values.shape[1], -1, -1)
        elif self.mode_keys == "constant":
            # passing a tensor of zeros is fine because we still have the bias of the linear layer
            # (hidden_size, num_models)
            keys_constant = torch.stack(
                [key(torch.zeros(key.in_features, device=values.device)) for key in self.keys],
                dim=-1,
            )
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = (
                keys_constant.unsqueeze(dim=0)
                .unsqueeze(dim=0)
                .expand(batch_size, num_tokens, -1, -1)
            )
        else:
            raise ValueError(f"Unknown keys mode: {self.mode_keys}")

        attn_weight = torch.softmax((query.unsqueeze(-2) @ keys / self.hidden_size**0.5), dim=-1)
        self.attention_logger(attn_weight)
        aggregated = attn_weight @ values.transpose(-1, -2)

        return aggregated.squeeze(-2)


class TransformerMultiModel(Module):
    def __init__(
        self,
        # The shared model type e.g. bert-base-cased. This should work with AutoConfig.from_pretrained.
        model_name: str,
        # A mapping from model ids to actual model names or paths to load the model weights from.
        pretrained_models: Dict[str, Union[str, Dict[str, Any]]],
        # if False, do not load the weights of the pretrained_models. Useful to save bandwidth when the model
        # is already trained because we load the weights from the checkpoint after initialisation.
        load_model_weights: bool = True,
        # can be either a string indicating an aggregation type or a dictionary with a "type" key and additional
        # parameters for the aggregation function. Currently, the following aggregation types are supported:
        # - "mean": average the logits of all models
        # - "sum": sum the logits of all models
        # - "attention": use an attention mechanism to aggregate the logits of all models
        aggregate: Union[str, Dict[str, Any]] = "mean",
        # A list of model ids to freeze during training.
        freeze_models: Optional[List[str]] = None,
        # A dictionary of config overrides to pass to AutoConfig.from_pretrained.
        config_overrides: Optional[Dict[str, Any]] = None,
        # The size of the vocabulary of the tokenizer. If provided, the token embeddings of all models are resized
        # to this size.
        tokenizer_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        if len(pretrained_models) < 1:
            raise ValueError("At least one model path must be provided")

        self.config = AutoConfig.from_pretrained(model_name, **(config_overrides or {}))
        pretrained_models_and_configs = {}
        for model_id, path_or_kwargs in pretrained_models.items():
            if isinstance(path_or_kwargs, str):
                pretrained_models_and_configs[model_id] = (path_or_kwargs, copy(self.config))
            elif isinstance(path_or_kwargs, dict):
                # copy to not modify the original
                config = copy(self.config)
                kwargs = copy(path_or_kwargs)

                name_or_path = kwargs.pop("name_or_path")
                for k, v in kwargs.items():
                    setattr(config, k, v)
                pretrained_models_and_configs[model_id] = (name_or_path, config)
            else:
                raise ValueError(
                    f"Expected a string or a dictionary for model id {model_id}, but got "
                    f"{path_or_kwargs} of type {type(path_or_kwargs)}"
                )
        if load_model_weights:
            self.models = ModuleDict(
                {
                    model_id: AutoModel.from_pretrained(
                        name_or_path,
                        config=config,
                    )
                    for model_id, (name_or_path, config) in pretrained_models_and_configs.items()
                }
            )
        else:
            self.models = ModuleDict(
                {
                    model_id: AutoModel.from_config(config=config)
                    for model_id, (name_or_path, config) in pretrained_models_and_configs.items()
                }
            )

        if tokenizer_vocab_size is not None:
            self.resize_token_embeddings(tokenizer_vocab_size)

        for model_id in freeze_models or []:
            self.models[model_id].requires_grad_(False)

        aggregate_config: Dict[str, Any]
        if isinstance(aggregate, str):
            aggregate_config = {"type": aggregate}
        else:
            # copy to not modify the original
            aggregate_config = copy(aggregate)
        aggregate_type = aggregate_config.pop("type")
        if aggregate_type == "mean":
            self.aggregate = aggregate_mean
        elif aggregate_type == "sum":
            self.aggregate = aggregate_sum
        elif aggregate_type == "concat":
            self.aggregate = ConcatAggregator(
                input_size=self.config.hidden_size, num_models=len(self.models)
            )
        elif aggregate_type == "attention":
            self.aggregate = AttentionBasedAggregator(
                input_size=self.config.hidden_size,
                model_ids=list(self.models),
                **aggregate_config,
            )
        else:
            raise NotImplementedError(f"Aggregate method '{aggregate_type}' is not implemented")

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        for model in self.models.values():
            model.resize_token_embeddings(new_num_tokens)

    def forward(self, **inputs):
        results_per_model = {
            model_name: model(**inputs) for model_name, model in self.models.items()
        }
        # get the logits from each model output
        logits_per_model = {k: v[0] for k, v in results_per_model.items()}
        return self.aggregate(logits_per_model)
