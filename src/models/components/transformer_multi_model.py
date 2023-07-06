from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module, ModuleDict
from transformers import AutoConfig, AutoModel


def aggregate_mean(x: Dict[str, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(x.values()), dim=-1)
    aggregated = torch.mean(stacked, dim=-1)
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
        aggregate: str = "mean",
        # A list of model ids to freeze during training.
        freeze_models: Optional[List[str]] = None,
        # A dictionary of config overrides to pass to AutoConfig.from_pretrained.
        config_overrides: Optional[Dict[str, Any]] = None,
        # you may want to resize the tokenizer embeddings to match the size of the actual tokenizer vocab.
        tokenizer_vocab_size: Optional[int] = None,
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
            if tokenizer_vocab_size is not None:
                for model in self.models.values():
                    model.resize_token_embeddings(tokenizer_vocab_size)
        else:
            self.models = ModuleDict(
                {
                    model_id: AutoModel.from_config(config=self.config)
                    for model_id in pretrained_models
                }
            )

        for model_id in freeze_models or []:
            self.models[model_id].requires_grad_(False)

        if aggregate == "mean":
            self.aggregate = aggregate_mean
        else:
            raise NotImplementedError(f"Aggregate method '{aggregate}' is not implemented")

    def forward(self, **inputs):
        results_per_model = {
            model_name: model(**inputs) for model_name, model in self.models.items()
        }
        # get the logits from each model output
        logits_per_model = {k: v[0] for k, v in results_per_model.items()}
        return self.aggregate(logits_per_model)
