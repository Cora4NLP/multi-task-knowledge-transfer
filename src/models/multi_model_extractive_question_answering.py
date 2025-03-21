import logging
from typing import Any, Dict, List, Optional, Tuple

import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, functional
from torch.optim import Adam
from transformers import BatchEncoding, get_linear_schedule_with_warmup
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from src.models.components import TransformerMultiModel

BatchOutput = Dict[str, Any]

StepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Dict[str, Tensor]],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class MultiModelExtractiveQuestionAnsweringModel(PyTorchIEModel):
    def __init__(
        self,
        pretrained_models: Dict[str, str],
        max_input_length: int,
        pretrained_default_config: Optional[str] = None,
        pretrained_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.0,
        aggregate: str = "mean",
        freeze_models: Optional[List[str]] = None,
        truncate_models: Optional[Dict[str, int]] = None,
        model_name: Optional[str] = None,
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
        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion

        self.base_models = TransformerMultiModel(
            pretrained_models=pretrained_models,
            pretrained_default_config=pretrained_default_config,
            pretrained_configs=pretrained_configs,
            load_model_weights=not self.is_from_pretrained,
            aggregate=aggregate,
            freeze_models=freeze_models,
            truncate_models=truncate_models,
        )
        self.max_input_length = max_input_length

        self.qa_outputs = nn.Linear(self.base_models.config.hidden_size, 2)

        self.f1_start = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.max_input_length
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )
        self.f1_end = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.max_input_length
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

        # Initialize weights and apply final processing (taken from self.post_init())
        # Initialize weights
        self.apply(self._init_weights)

    # taken from BertPreTrainedModel, see
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.base_models.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.base_models.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs: BatchEncoding) -> QuestionAnsweringModelOutput:
        sequence_output = self.base_models(**inputs)

        # taken from BertForQuestionAnswering, see
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
        )

    def step(
        self,
        stage: str,
        batch: StepBatchEncoding,
    ):
        inputs, targets = batch
        assert targets is not None, "targets has to be available for training"

        output = self(inputs)

        start_logits = output.start_logits
        end_logits = output.end_logits

        start_positions = targets["start_positions"]
        end_positions = targets["end_positions"]

        # taken from BertForQuestionAnswering, see
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        sequence_length = inputs["input_ids"].size(1)

        f1_start = self.f1_end[f"stage_{stage}"]
        # We need to pad the logits to the max_input_length, otherwise the F1 metric complains
        # that the shape does not match the num_classes.
        start_logits_padded = functional.pad(
            start_logits, (0, self.max_input_length - sequence_length), value=float("-inf")
        )
        f1_start(start_logits_padded, start_positions)
        self.log(
            f"{stage}/f1_start",
            f1_start,
            on_step=(stage == TRAINING),
            on_epoch=True,
            prog_bar=True,
        )
        f1_end = self.f1_end[f"stage_{stage}"]
        # We need to pad the logits to the max_input_length, otherwise the F1 metric complains
        # that the shape does not match the num_classes.
        end_logits_padded = functional.pad(
            end_logits, (0, self.max_input_length - sequence_length), value=float("-inf")
        )
        f1_end(end_logits_padded, end_positions)
        self.log(
            f"{stage}/f1_end", f1_end, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True
        )
        # log f1 as simple average of start and end f1. we need to call compute() on the metric to get
        # the actual value, otherwise lightning complains that there is no model attribute with name f"{stage}/f1".
        f1_value = (f1_start.compute() + f1_end.compute()) / 2
        self.log(f"{stage}/f1", f1_value, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: StepBatchEncoding, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepBatchEncoding, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: StepBatchEncoding, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            all_params = dict(self.named_parameters())
            base_model_params = dict(self.base_models.named_parameters(prefix="base_models"))
            task_params = {k: v for k, v in all_params.items() if k not in base_model_params}
            optimizer = Adam(
                [
                    {"params": base_model_params.values(), "lr": self.learning_rate},
                    {"params": task_params.values(), "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = Adam(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
