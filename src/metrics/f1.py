import logging
from typing import Callable, Optional, Tuple

from pytorch_ie.core import Annotation, Document
from pytorch_ie.metrics import F1Metric

logger = logging.getLogger(__name__)


class F1BestVsCandidatesMetric(F1Metric):
    """This metric calculates the F1 score over documents by comparing the best predicted
    annotation to multiple gold candidate annotations.

    This is useful for tasks like extractive question answering where there are multiple possible
    answers for a question, but only one is correct.
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
