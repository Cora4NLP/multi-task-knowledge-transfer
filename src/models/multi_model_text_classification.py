import logging
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.models.components import TransformerMultiModel

TransformerTextClassificationModelBatchEncoding = MutableMapping[str, Any]
TransformerTextClassificationModelBatchOutput = Dict[str, Any]

TransformerTextClassificationModelStepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Tensor],
]

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class MultiModelTextClassificationModel(PyTorchIEModel):
    def __init__(
        self,
        model_name: str,
        pretrained_models: Dict[str, Union[str, Dict[str, Any]]],
        num_classes: int,
        tokenizer_vocab_size: Optional[int] = None,
        aggregate: str = "mean",
        freeze_models: Optional[List[str]] = None,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        multi_label: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion

        self.base_models = TransformerMultiModel(
            model_name=model_name,
            pretrained_models=pretrained_models,
            load_model_weights=not self.is_from_pretrained,
            aggregate=aggregate,
            freeze_models=freeze_models,
            config_overrides={"num_labels": num_classes},
            # this is important because we may have added new special tokens to the tokenizer
            tokenizer_vocab_size=tokenizer_vocab_size,
        )

        classifier_dropout = (
            self.base_models.config.classifier_dropout
            if self.base_models.config.classifier_dropout is not None
            else self.base_models.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(self.base_models.config.hidden_size, num_classes)

        self.loss_fct = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(
        self, inputs: TransformerTextClassificationModelBatchEncoding
    ) -> TransformerTextClassificationModelBatchOutput:
        last_hidden_states = self.base_models(**inputs)

        cls_embeddings = last_hidden_states[:, 0, :]
        logits = self.classifier(cls_embeddings)

        return {"logits": logits}

    def step(self, stage: str, batch: TransformerTextClassificationModelStepBatchEncoding):
        input_, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(input_)["logits"]

        loss = self.loss_fct(logits, target)

        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: TransformerTextClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            all_params = dict(self.named_parameters())
            base_model_params = dict(self.base_models.named_parameters(prefix="base_models"))
            task_params = {k: v for k, v in all_params.items() if k not in base_model_params}
            optimizer = AdamW(
                [
                    {"params": base_model_params.values(), "lr": self.learning_rate},
                    {"params": task_params.values(), "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
