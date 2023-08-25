import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pytorch_ie import tokenize_document
from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation, AnnotationList, TaskEncoding, TaskModule, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from tokenizers import Encoding
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
    score: Optional[float] = None

    def __str__(self) -> str:
        if self.targets is None:
            return ""
        context = self.targets[0]
        return str(context[self.start : self.end])


@dataclasses.dataclass
class ExtractiveQADocument(TextBasedDocument):
    """A PIE document with annotations for SQuAD v2.0."""

    title: Optional[str] = None
    questions: AnnotationList[Question] = annotation_field()
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(targets=["text", "questions"])


@dataclasses.dataclass
class TokenizedExtractiveQADocument(TokenBasedDocument):
    """A tokenized PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(targets=["tokens", "questions"])


DocumentType: TypeAlias = ExtractiveQADocument
InputEncoding: TypeAlias = Union[Dict[str, Any], BatchEncoding]


@dataclasses.dataclass
class TargetEncoding:
    start_position: int
    end_position: int


TaskEncodingType: TypeAlias = TaskEncoding[
    ExtractiveQADocument,
    InputEncoding,
    TargetEncoding,
]

TaskBatchEncoding: TypeAlias = Tuple[BatchEncoding, Optional[Dict[str, Any]]]
ModelBatchOutput: TypeAlias = QuestionAnsweringModelOutput


@dataclasses.dataclass
class TaskOutput:
    start: int
    end: int
    # the logits are not yet used
    start_probability: float
    end_probability: float


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
        if len(answers._targets) != 2:
            raise Exception(
                f"the answers layer is expected to target exactly two layers: text and questions, but it has "
                f"the following targets: {answers._targets}"
            )
        question_layer_name = answers._targets[1]
        return document[question_layer_name]

    def get_context(self, document: DocumentType) -> str:
        answers = self.get_answer_layer(document)
        if len(answers._targets) != 2:
            raise Exception(
                f"the answers layer is expected to target exactly two layers: text and questions, but it has "
                f"the following targets: {answers._targets}"
            )
        context_field_name = answers._targets[0]
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

        questions = self.get_question_layer(document)
        task_encodings: List[TaskEncodingType] = []
        for question in questions:
            tokenized_docs = tokenize_document(
                document,
                tokenizer=self.tokenizer,
                text=question.text.strip(),
                truncation="only_second",
                max_length=self.max_length,
                return_overflowing_tokens=True,
                result_document_type=TokenizedExtractiveQADocument,
                strict_span_conversion=False,
                verbose=False,
            )
            for doc in tokenized_docs:
                inputs = self.tokenizer.convert_tokens_to_ids(list(doc.tokens))
                task_encodings.append(
                    TaskEncodingType(
                        document=document,
                        inputs=inputs,
                        metadata=dict(question=question, tokenized_document=doc),
                    )
                )
        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> Optional[TargetEncoding]:
        answers = self.get_answer_layer(task_encoding.metadata["tokenized_document"])
        if len(answers) > 1:
            logger.warning(
                f"The answers layer is expected to have not more than one answer, but it has "
                f"{len(answers)} answers. We take just the first one."
            )
        # if there are no answers, we return a dummy target encoding
        if len(answers) == 0:
            return TargetEncoding(0, 0)
        answer = answers[0]
        if not answer.question == task_encoding.metadata["question"]:
            raise Exception(
                f"the answer {answer} does not match the question {task_encoding.metadata['question']}"
            )
        return TargetEncoding(answer.start, answer.end - 1)

    def collate(
        self, task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ) -> TaskBatchEncoding:
        def task_encoding2input_features(task_encoding: TaskEncodingType) -> Dict[str, Any]:
            encoding = task_encoding.metadata["tokenized_document"].metadata["tokenizer_encoding"]
            return {"input_ids": encoding.ids, "token_type_ids": encoding.type_ids}

        input_features = [
            task_encoding2input_features(task_encoding) for task_encoding in task_encodings
        ]

        # will contain: input_ids, token_type_ids, attention_mask
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
        batch_size = len(model_output.start_logits)
        start_probs = torch.softmax(model_output.start_logits, dim=-1).detach().cpu().numpy()
        end_probs = torch.softmax(model_output.end_logits, dim=-1).detach().cpu().numpy()
        best_start = np.argmax(start_probs, axis=1)
        best_end = np.argmax(end_probs, axis=1)
        return [
            TaskOutput(
                start=best_start[i],
                end=best_end[i],
                start_probability=start_probs[i, best_start[i]],
                end_probability=end_probs[i, best_end[i]],
            )
            for i in range(batch_size)
        ]

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:

        tokenizer_encoding: Encoding = task_encoding.metadata["tokenized_document"].metadata[
            "tokenizer_encoding"
        ]
        start_chars = tokenizer_encoding.token_to_chars(task_output.start)
        end_chars = tokenizer_encoding.token_to_chars(task_output.end)
        if start_chars is not None and end_chars is not None:
            start_sequence_index = tokenizer_encoding.token_to_sequence(task_output.start)
            end_sequence_index = tokenizer_encoding.token_to_sequence(task_output.end)
            # the indices need to point into the context which is the second sequence
            if start_sequence_index == 1 and end_sequence_index == 1:
                start_char = start_chars[0]
                end_char = end_chars[-1]
                context = self.get_context(task_encoding.document)
                if 0 <= start_char < end_char <= len(context):
                    yield self.answer_annotation, ExtractiveAnswer(
                        start=start_char,
                        end=end_char,
                        question=task_encoding.metadata["question"],
                        score=float(task_output.start_probability * task_output.end_probability),
                    )
