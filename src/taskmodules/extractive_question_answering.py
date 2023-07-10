import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from pytorch_ie.annotations import Span
from pytorch_ie.core import (
    Annotation,
    AnnotationList,
    Document,
    TaskEncoding,
    TaskModule,
    annotation_field,
)
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=True, frozen=True)
class Question(Annotation):
    """A question about a context."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(eq=True, frozen=True)
class ExtractiveAnswer(Span):
    """An answer to a question."""

    question: Question

    def __str__(self) -> str:
        if self.question.target is None:
            return ""
        return str(self.question.target[self.start : self.end])


@dataclasses.dataclass
class ExtractiveQADocument(Document):
    """A PIE document with annotations for extractive question answering."""

    context: str
    questions: AnnotationList[Question] = annotation_field(target="context")
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(target="questions")
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


DocumentType: TypeAlias = ExtractiveQADocument
InputEncoding: TypeAlias = Union[Dict[str, Any], BatchEncoding]


@dataclasses.dataclass
class TargetEncoding:
    start_position: int
    end_position: int


_TaskEncoding: TypeAlias = TaskEncoding[
    ExtractiveQADocument,
    InputEncoding,
    TargetEncoding,
]

TaskBatchEncoding: TypeAlias = Tuple[BatchEncoding, Optional[Dict[str, Any]]]
ModelBatchOutput: TypeAlias = QuestionAnsweringModelOutput

TaskOutput: TypeAlias = Dict[str, Any]


@TaskModule.register()
class ExtractiveQuestionAnsweringTaskModule(TaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        max_length: int,
        answer_annotation: str = "answers",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.answer_annotation = answer_annotation
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length

    def get_answer_layer(self, document: DocumentType) -> AnnotationList[ExtractiveAnswer]:
        return document[self.answer_annotation]

    def get_question_layer(self, document: DocumentType) -> AnnotationList[Question]:
        answers = self.get_answer_layer(document)
        if len(answers._targets) != 1:
            raise Exception(
                f"the answers layer is expected to target exactly one questions layer, but it has "
                f"the following targets: {answers._targets}"
            )
        question_layer_name = answers._targets[0]
        return document[question_layer_name]

    def get_context(self, document: DocumentType) -> str:
        questions = self.get_question_layer(document)
        if len(questions._targets) != 1:
            raise Exception(
                f"the questions layer is expected to target exactly one context layer, but it has "
                f"the following targets: {questions._targets}"
            )
        context_field_name = questions._targets[0]
        return getattr(document, context_field_name)

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[
        Union[
            TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
            Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        ]
    ]:
        context = self.get_context(document)
        questions = self.get_question_layer(document)
        task_encodings: List[_TaskEncoding] = []
        for question in questions:
            # TODO: is this the right way to encode the context and question?
            context_and_question = f"context: {context} question: {question.text}"

            inputs: BatchEncoding = self.tokenizer(
                context_and_question,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=inputs,
                    metadata=dict(
                        context_start=len("context: "),
                        context_end=len(f"context: {context}"),
                        question=question,
                    ),
                )
            )
        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
    ) -> Optional[TargetEncoding]:
        answers = self.get_answer_layer(task_encoding.document)
        if len(answers) > 1:
            logger.warning(
                f"The answers layer is expected to have not more than one answer, but it has "
                f"{len(answers)} answers. We take just the first one."
            )
        if len(answers) == 0:
            return TargetEncoding(0, 0)
        answer = answers[0]
        if not answer.question == task_encoding.metadata["question"]:
            raise Exception(
                f"the answer {answer} does not match the question {task_encoding.metadata['question']}"
            )
        start_char = answer.start + task_encoding.metadata["context_start"]
        start_token = task_encoding.inputs.char_to_token(start_char)
        end_char = answer.end + task_encoding.metadata["context_start"]
        # the end token is inclusive
        end_token = task_encoding.inputs.char_to_token(end_char - 1)
        return TargetEncoding(start_token, end_token)

    def collate(
        self, task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ) -> TaskBatchEncoding:
        input_features = [task_encoding.inputs for task_encoding in task_encodings]

        inputs: BatchEncoding = self.tokenizer.pad(
            input_features, padding="longest", max_length=self.max_length, return_tensors="pt"
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        start_positions = torch.tensor(
            [task_encoding.targets.start_position for task_encoding in task_encodings],
            dtype=torch.int64,
        )
        end_positions = torch.tensor(
            [task_encoding.targets.end_position for task_encoding in task_encodings],
            dtype=torch.int64,
        )
        targets = {"start_positions": start_positions, "end_positions": end_positions}

        return inputs, targets

    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutput]:
        raise NotImplementedError

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        raise NotImplementedError
