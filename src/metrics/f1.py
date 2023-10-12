import logging
from collections import defaultdict
from typing import Callable, Optional, Set, Tuple

from pytorch_ie.core import Annotation, Document
from pytorch_ie.metrics import F1Metric

from src.taskmodules.extractive_question_answering import ExtractiveAnswer, ExtractiveQADocument

logger = logging.getLogger(__name__)


class F1BestVsCandidatesMetric(F1Metric):
    """This metric calculates the F1 score over documents by comparing the best predicted
    annotation to multiple gold candidate annotations.

    This is useful for tasks like extractive question answering where there are multiple gold
    candidate answers for a single question and, on the other hand, the model may produce multiple
    answers for one question because of windowing (induced by max input length restrictions of the
    model).
    """

    def __init__(self, score_field: str = "score", **kwargs) -> None:
        super().__init__(**kwargs)
        self.score_field = score_field

    def calculate_counts(
        self,
        document: Document,
        annotation_filter: Optional[Callable[[Annotation], bool]] = None,
    ) -> Tuple[int, int, int]:
        annotation_filter = annotation_filter or (lambda ann: True)
        predicted_annotations = {
            ann for ann in document[self.layer].predictions if annotation_filter(ann)
        }
        gold_annotations = {ann for ann in document[self.layer] if annotation_filter(ann)}

        if len(gold_annotations) == 0 and len(predicted_annotations) == 0:
            # tp, fp, fn
            return 0, 0, 0
        if len(gold_annotations) > 0 and len(predicted_annotations) == 0:
            # tp, fp, fn
            return 0, 0, 1
        if len(gold_annotations) == 0 and len(predicted_annotations) > 0:
            # tp, fp, fn
            return 0, 1, 0

        if any(getattr(ann, self.score_field) is None for ann in predicted_annotations):
            raise ValueError(
                f"All predicted annotations must have a {self.score_field} value to calculate "
                f"{self.__class__.__name__}."
            )
        best_predicted_annotation = max(
            predicted_annotations, key=lambda ann: getattr(ann, self.score_field)
        )

        if best_predicted_annotation in gold_annotations:
            # tp, fp, fn
            return 1, 0, 0
        else:
            # tp, fp, fn
            return 0, 1, 1


class F1ForExtractiveQuestionAnswering(F1Metric):
    def __init__(self, score_field: str = "score", **kwargs) -> None:
        super().__init__(layer="answers", **kwargs)
        self.score_field = score_field
        self.reset()

    def calculate_counts(
        self,
        document: ExtractiveQADocument,
        annotation_filter: Optional[Callable[[Annotation], bool]] = None,
    ) -> Tuple[int, int, int]:
        if annotation_filter is not None:
            raise ValueError(f"{self.__class__.__name__} does not support annotation filters.")
        predicted_annotations_per_question = defaultdict(set)
        ann: ExtractiveAnswer
        for ann in document[self.layer].predictions:
            predicted_annotations_per_question[ann.question].add(ann)
        gold_annotations_per_question = defaultdict(set)
        for ann in document[self.layer]:
            gold_annotations_per_question[ann.question].add(ann)

        tp, fp, fn = 0, 0, 0
        questions = set(gold_annotations_per_question) | set(predicted_annotations_per_question)
        for question in questions:
            gold_annotations: Set[ExtractiveAnswer] = gold_annotations_per_question.get(
                question, set()
            )
            predicted_annotations: Set[ExtractiveAnswer] = predicted_annotations_per_question.get(
                question, set()
            )

            if len(gold_annotations) == 0 and len(predicted_annotations) == 0:
                continue

            if len(gold_annotations) > 0 and len(predicted_annotations) == 0:
                fn += 1
                continue

            if len(gold_annotations) == 0 and len(predicted_annotations) > 0:
                fp += 1
                continue

            if any(getattr(ann, self.score_field) is None for ann in predicted_annotations):
                raise ValueError(
                    f"All predicted annotations must have a {self.score_field} value to calculate "
                    f"{self.__class__.__name__}."
                )
            best_predicted_annotation: ExtractiveAnswer = max(
                predicted_annotations, key=lambda ann: getattr(ann, self.score_field)
            )

            # TODO: calculate the actual tp, fp, fn with respect to all gold questions

        return tp, fp, fn
