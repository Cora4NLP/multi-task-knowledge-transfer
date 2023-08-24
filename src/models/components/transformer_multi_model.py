from copy import copy
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module, ModuleDict
from transformers import AutoConfig, AutoModel


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


class AttentionBasedAggregator(Module):
    def __init__(
        self,
        input_size: int,
        n_models: int,
        hidden_size: int = 128,
        query_idx: Union[int, str] = 0,
        mode: str = "token2token",
    ):
        super().__init__()
        # the index of the model to use as the query. If a string is provided, it is the key of the model in
        # the input dictionary. If an int is provided, it is the index of the model *values* in the input dictionary.
        self.query_idx = query_idx
        self.mode = mode
        self.hidden_size = hidden_size
        self.n_models = n_models
        # we need only one query (just for the target model embeddings)
        self.query = torch.nn.Linear(input_size, hidden_size)
        # we need keys and values for all model embeddings
        self.keys = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, hidden_size) for _ in range(n_models)]
        )
        self.values = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, input_size) for _ in range(n_models)]
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_key = (
            self.query_idx if isinstance(self.query_idx, str) else list(x.keys())[self.query_idx]
        )
        # (batch_size, num_tokens, output_size, num_models)
        values = torch.stack([k(v) for k, v in zip(self.values, x.values())], dim=-1)
        batch_size, num_tokens = values.shape[:2]

        # use token embeddings of the target model as query and the token embeddings of all models as keys
        if self.mode == "token2token":
            # (batch_size, num_tokens, hidden_size)
            query = self.query(x[query_key])
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = torch.stack([k(v) for k, v in zip(self.keys, x.values())], dim=-1)

        # use token embeddings of the target model as query and the cls embeddings of all models as keys
        elif self.mode == "token2cls":
            # (batch_size, num_tokens, hidden_size)
            query = self.query(x[query_key])
            # (batch_size, hidden_size, num_models)
            keys_cls = torch.stack([k(v[:, 0, :]) for k, v in zip(self.keys, x.values())], dim=-1)
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = keys_cls.unsqueeze(dim=1).expand(-1, values.shape[1], -1, -1)

        # use the cls embedding of the target model as query and the token embeddings of all models as keys
        elif self.mode == "cls2token":
            # (batch_size, hidden_size)
            query_cls = self.query(x[query_key][:, 0, :])
            # (batch_size, num_tokens, hidden_size)
            query = query_cls.unsqueeze(dim=1).expand(-1, values.shape[1], -1)
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = torch.stack([k(v) for k, v in zip(self.keys, x.values())], dim=-1)

        # use the cls embedding of the target model as query and the cls embeddings of all models as keys
        elif self.mode == "cls2cls":
            # (batch_size, hidden_size)
            query_cls = self.query(x[query_key][:, 0, :])
            # (batch_size, num_tokens, hidden_size)
            query = query_cls.unsqueeze(dim=1).expand(-1, values.shape[1], -1)
            # (batch_size, hidden_size, num_models)
            keys_cls = torch.stack([k(v[:, 0, :]) for k, v in zip(self.keys, x.values())], dim=-1)
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = keys_cls.unsqueeze(dim=1).expand(-1, values.shape[1], -1, -1)

        # use a (learned) constant query and the token embeddings of all models as keys
        elif self.mode == "constant2token":
            # passing a tensor of zeros is fine because we still have the bias of the linear layer
            # (hidden_size,)
            query_constant = self.query(torch.zeros(self.query.in_features, device=values.device))
            # (batch_size, num_tokens, hidden_size)
            query = (
                query_constant.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, num_tokens, -1)
            )
            # (batch_size, num_tokens, hidden_size, num_models)
            keys = torch.stack([k(v) for k, v in zip(self.keys, x.values())], dim=-1)

        # use token embeddings of the target model as query and (learned) constant keys
        elif self.mode == "token2constant":
            # (batch_size, num_tokens, hidden_size)
            query = self.query(x[query_key])
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

        # use a (learned) constant query and (learned) constant keys
        elif self.mode == "constant2constant":
            # passing a tensor of zeros is fine because we still have the bias of the linear layer
            # (hidden_size,)
            query_constant = self.query(torch.zeros(self.query.in_features, device=values.device))
            # (batch_size, num_tokens, hidden_size)
            query = (
                query_constant.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, num_tokens, -1)
            )
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
            raise ValueError(f"Unknown attention mode: {self.mode}")

        # flatten (combine batch dim and token dim) to calculate attention weights
        # (batch_size x num_tokens, 1, hidden_size)
        query_flat = query.reshape(-1, 1, self.hidden_size)
        # (batch_size x num_tokens, hidden_size, num_models)
        keys_flat = keys.reshape(-1, self.hidden_size, self.n_models)
        # (batch_size x num_tokens, 1, num_models)
        scores_flat = torch.bmm(query_flat, keys_flat)
        attention_flat = torch.softmax(scores_flat, dim=-1)

        # unflatten
        # (batch_size, num_tokens, 1, num_models)
        attention = attention_flat.view(batch_size, num_tokens, 1, self.n_models)

        # (batch_size, num_tokens, output_size, num_models)
        weighted = values * attention
        # (batch_size, num_tokens, output_size)
        aggregated = weighted.sum(dim=-1)

        return aggregated


class TransformerMultiModel(Module):
    def __init__(
        self,
        # The shared model type e.g. bert-base-cased. This should work with AutoConfig.from_pretrained.
        model_name: str,
        # A mapping from model ids to actual model names or paths to load the model weights from.
        pretrained_models: Dict[str, str],
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
    ):
        super().__init__()
        if len(pretrained_models) < 1:
            raise ValueError("At least one model path must be provided")

        self.config = AutoConfig.from_pretrained(model_name, **(config_overrides or {}))
        if load_model_weights:
            self.models = ModuleDict(
                {
                    model_id: AutoModel.from_pretrained(
                        path, config=self.config, ignore_mismatched_sizes=True
                    )
                    for model_id, path in pretrained_models.items()
                }
            )
        else:
            self.models = ModuleDict(
                {
                    model_id: AutoModel.from_config(config=self.config)
                    for model_id in pretrained_models
                }
            )

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
                n_models=len(self.models),
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
