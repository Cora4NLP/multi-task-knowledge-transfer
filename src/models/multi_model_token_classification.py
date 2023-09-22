from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import BatchEncoding
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


@PyTorchIEModel.register()
class MultiModelTokenClassificationModel(PyTorchIEModel):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained_models: Dict[str, Union[str, Dict[str, Any]]],
        aggregate: str = "mean",
        freeze_models: Optional[List[str]] = None,
        classifier_dropout: float = 0.1,
        learning_rate: float = 1e-5,
        label_pad_token_id: int = -100,
        ignore_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.label_pad_token_id = label_pad_token_id
        self.num_classes = num_classes

        self.base_models = TransformerMultiModel(
            model_name=model_name,
            pretrained_models=pretrained_models,
            load_model_weights=not self.is_from_pretrained,
            aggregate=aggregate,
            freeze_models=freeze_models,
            config_overrides={"num_labels": num_classes},
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(
            self.base_models.config.hidden_size, self.base_models.config.num_labels
        )

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index
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

        loss_fct = CrossEntropyLoss()
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

    def training_step(self, batch: MultiModelTokenClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: MultiModelTokenClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: MultiModelTokenClassificationModelStepBatchEncoding, batch_idx: int):  # type: ignore
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
