import logging
from collections import defaultdict
from typing import Callable, Optional, Set, Tuple

from pytorch_ie.core import Annotation, Document
from pytorch_ie.metrics import F1Metric

from src.taskmodules.extractive_question_answering import ExtractiveAnswer, ExtractiveQADocument
import string
import re
import collections
from typing import Callable, Collection, Dict, Hashable, Optional, Tuple
from functools import partial

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
            ) # take best phrase from different strides

            # TODO: calculate the actual tp, fp, fn with respect to all gold questions

        return tp, fp, fn


#class SQuADF1ForExtractiveQuestionAnswering(F1Metric):
class SQuADF1ForExtractiveQuestionAnswering(F1Metric):
    def __init__(self, score_field: str = "score", **kwargs) -> None:
        super().__init__(layer="answers", **kwargs)
        self.score_field = score_field
        self.reset()
        self.labels = ["overall"]

    def has_this_label(self, ann: Annotation, label_field: str, label: str) -> bool:
        return getattr(ann, label_field) == label

    def reset(self):
        self.counts = defaultdict(lambda: (0, 0))

    def add_counts(self, counts: Tuple[float, float], label: str):
        self.counts[label] = (
            self.counts[label][0] + counts[0],
            self.counts[label][1] + counts[1],
        )
    def _update(self, document: Document):

        for label in self.labels:
            new_counts = self.calculate_counts(
                document=document,
                #annotation_filter=partial(
                #    self.has_this_label, label_field=self.label_field, label=label
                #),
                annotation_filter=None,
            )
            self.add_counts(new_counts, label=label)
    def _compute(self) -> Dict[str, Dict[str, float]]:
        res = dict()

        if self.per_label:
            res["overall"] = {"f1": 0.0, "em": 0.0}

        print(self.counts.items())
        for label, counts in self.counts.items():

            f1, em = counts
            res[label] = {"f1": f1, "em": em}

            #
            #if label in self.labels:
            #    res[label]["f1"] += f1 / len(self.labels)
            #    res[label]["em"] += em / len(self.labels)

        return res

    def normalize_answer(self,s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    #def __init__(self, score_field: str = "score", **kwargs) -> None:
    def get_tokens(self,s):
         if not s:
           return []
         return self.normalize_answer(s).split()


    def compute_exact(self,a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))


    def compute_f1(self,a_gold, a_pred):
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_exact(self, a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def calculate_counts(
        self,
        document: ExtractiveQADocument,
        annotation_filter: Optional[Callable[[Annotation], bool]] = None,
    ) -> Tuple[int, int, int]:
        if annotation_filter is not None:
            raise ValueError(f"{self.__class__.__name__} does not support annotation filters.")
        predicted_annotations_per_question = defaultdict(set)
        passage_per_question = defaultdict(set)
        ann: ExtractiveAnswer
        #print(document[self.layer])
        for ann in document[self.layer].predictions:
            predicted_annotations_per_question[ann.question].add(ann)
        gold_annotations_per_question = defaultdict(set)
        for ann in document[self.layer]:
            gold_annotations_per_question[ann.question].add(ann)

        for ann in document[self.layer]:
            print("Ann:", ann, type(ann))
            #print("question:",ann.question)
            passage_per_question[ann.question].add(ann)
        tp, fp, fn = 0, 0, 0
        questions = set(gold_annotations_per_question) | set(predicted_annotations_per_question)
        for question in questions:
            gold_annotations: Set[ExtractiveAnswer] = gold_annotations_per_question.get(
                question, set()
            )
            predicted_annotations: Set[ExtractiveAnswer] = predicted_annotations_per_question.get(
                question, set()
            )

            #passage=
            #print("--question--:", question)
            #print("--passage--:",[ p for p in passage_per_question[question]])
            #print("gold_annotations:", [ g.__str__() for g in gold_annotations])
            #print("predicted_annotations:", [ p.__str__() for p in predicted_annotations])

            best_f1=0
            best_em=0
            for g in gold_annotations:
                for p in predicted_annotations:
                    f1=self.compute_f1(g.__str__(), p.__str__())
                    em = self.compute_exact(g.__str__(), p.__str__())

                    if f1 > best_f1:
                        best_f1=f1
                        best_em=em

                    #if em > best_em:
                    #    best_em=em

            #print("F1:",best_f1)
            #print("EM:", best_em)

        #return tp, fp, fn
        return best_f1, best_em
