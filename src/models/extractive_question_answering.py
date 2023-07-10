from typing import Any, Dict, Optional, Tuple

from pytorch_ie.core import PyTorchIEModel
from torch import Tensor
from torch.optim import Adam
from transformers import AutoConfig, AutoModelForQuestionAnswering, BatchEncoding
from transformers.modeling_outputs import QuestionAnsweringModelOutput

BatchOutput = Dict[str, Any]

StepBatchEncoding = Tuple[
    Dict[str, Tensor],
    Optional[Dict[str, Tensor]],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"


@PyTorchIEModel.register()
class ExtractiveQuestionAnsweringModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModelForQuestionAnswering.from_config(config=config)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                model_name_or_path, config=config
            )

        # self.f1 = nn.ModuleDict(
        #    {
        #        f"stage_{stage}": torchmetrics.F1Score(
        #            num_classes=num_classes, ignore_index=ignore_index
        #        )
        #        for stage in [TRAINING, VALIDATION, TEST]
        #    }
        # )

    def forward(self, inputs: BatchEncoding) -> QuestionAnsweringModelOutput:
        return self.model(**inputs)

    def step(
        self,
        stage: str,
        batch: StepBatchEncoding,
    ):
        inputs, targets = batch
        assert targets is not None, "targets has to be available for training"

        output = self({**inputs, **targets})

        loss = output.loss
        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        # target_flat = target.view(-1)

        # valid_indices = target_flat != self.label_pad_token_id
        # valid_logits = output.logits.view(-1, self.num_classes)[valid_indices]
        # valid_target = target_flat[valid_indices]

        # f1 = self.f1[f"stage_{stage}"]
        # f1(valid_logits, valid_target)
        # self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: StepBatchEncoding, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepBatchEncoding, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: StepBatchEncoding, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
