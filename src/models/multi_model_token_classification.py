import logging
from typing import Any, Dict, List, Optional, Tuple

import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from torch import Tensor, nn
from torch.optim import Adam
from transformers import BatchEncoding, get_linear_schedule_with_warmup
from typing_extensions import TypeAlias

from src.models.components import TransformerMultiModel

MultiModelTokenClassificationModelBatchEncoding: TypeAlias = BatchEncoding
MultiModelTokenClassificationModelBatchOutput = Dict[str, Any]

MultiModelTokenClassificationModelStepBatchEncoding = Tuple[
    MultiModelTokenClassificationModelBatchEncoding,
    Optional[Tensor],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"


logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class MultiModelTokenClassificationModel(PyTorchIEModel):
    def __init__(
        self,
        num_classes: int,
        pretrained_models: Dict[str, str],
        pretrained_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        pretrained_default_config: Optional[str] = None,
        aggregate: str = "mean",
        freeze_models: Optional[List[str]] = None,
        classifier_dropout: float = 0.1,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.0,
        label_pad_token_id: int = -100,
        ignore_index: int = 0,
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

        self.label_pad_token_id = label_pad_token_id
        self.num_classes = num_classes

        self.base_models = TransformerMultiModel(
            pretrained_models=pretrained_models,
            pretrained_default_config=pretrained_default_config,
            pretrained_configs=pretrained_configs,
            load_model_weights=not self.is_from_pretrained,
            aggregate=aggregate,
            freeze_models=freeze_models,
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.base_models.config.hidden_size, self.num_classes)

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(
        self, inputs: MultiModelTokenClassificationModelBatchEncoding
    ) -> MultiModelTokenClassificationModelBatchOutput:
        # get the sequence logits aggregated over all models
        logits = self.base_models(**inputs)

        sequence_output = self.dropout(logits)
        logits = self.classifier(sequence_output)

        # Return a dict with "logits" as key to confirm to the interface of the taskmodule,
        # i.e. pytorch_ie.taskmodules.TransformerTokenClassificationTaskModule
        return {"logits": logits}

    def step(
        self,
        stage: str,
        batch: MultiModelTokenClassificationModelStepBatchEncoding,
    ):
        input_, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(input_)["logits"]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), target.view(-1))

        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        target_flat = target.view(-1)

        valid_indices = target_flat != self.label_pad_token_id
        valid_logits = logits.view(-1, self.num_classes)[valid_indices]
        valid_target = target_flat[valid_indices]

        f1 = self.f1[f"stage_{stage}"]
        f1(valid_logits, valid_target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(
        self, batch: MultiModelTokenClassificationModelStepBatchEncoding, batch_idx: int
    ):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(
        self, batch: MultiModelTokenClassificationModelStepBatchEncoding, batch_idx: int
    ):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(
        self, batch: MultiModelTokenClassificationModelStepBatchEncoding, batch_idx: int
    ):
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
