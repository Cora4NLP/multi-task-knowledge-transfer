import collections
import logging
import re
import string
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from pytorch_ie.core import Annotation, Document, DocumentMetric
from pytorch_ie.metrics import F1Metric

from src.taskmodules.extractive_question_answering import ExtractiveQADocument

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


class SQuADF1ForExtractiveQuestionAnswering(DocumentMetric):
    def __init__(
        self,
        no_answer_probs: Optional[Dict[str, float]] = None,
        no_answer_probability_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.no_answer_probs = no_answer_probs
        self.no_answer_probability_threshold = no_answer_probability_threshold
        self.default_na_prob = 0.0

    def reset(self):
        self.exact_scores = {}
        self.f1_scores = {}
        # qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
        self.qas_id_to_has_answer = {}
        # has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
        self.has_answer_qids = []
        # no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
        self.no_answer_qids = []

    def _update(self, document: ExtractiveQADocument):
        gold_answers_for_questions = defaultdict(list)
        predicted_answers_for_questions = defaultdict(list)
        for ann in document.answers:
            gold_answers_for_questions[ann.question].append(ann)
        for ann in document.answers.predictions:
            predicted_answers_for_questions[ann.question].append(ann)

        for idx, question in enumerate(document.questions):
            # qas_id = example.qas_id
            if document.id is None:
                qas_id = f"text={document.text},question={question}"
            else:
                qas_id = document.id + f"_{idx}"

            # qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
            self.qas_id_to_has_answer[qas_id] = bool(gold_answers_for_questions[question])
            # has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
            # no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
            if self.qas_id_to_has_answer[qas_id]:
                self.has_answer_qids.append(qas_id)
            else:
                self.no_answer_qids.append(qas_id)

            # gold_answers = [answer["text"] for answer in example.answers if self.normalize_answer(answer["text"])]
            gold_answers = [
                str(answer)
                for answer in gold_answers_for_questions[question]
                if self.normalize_answer(str(answer))
            ]

            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]

            predicted_answers = predicted_answers_for_questions[question]
            # if qas_id not in preds:
            if len(predicted_answers) == 0:
                # logger.warning(f"Missing prediction for {qas_id}")
                self.exact_scores[qas_id] = 0
                self.f1_scores[qas_id] = 0.0
            else:
                # prediction = preds[qas_id]
                best_predicted_answer = max(predicted_answers, key=lambda ann: ann.score)
                prediction = str(best_predicted_answer)
                self.exact_scores[qas_id] = max(
                    self.compute_exact(a, prediction) for a in gold_answers
                )
                self.f1_scores[qas_id] = max(self.compute_f1(a, prediction) for a in gold_answers)

    def apply_no_ans_threshold(self, scores: Dict[str, float]) -> Dict[str, float]:
        new_scores = {}
        for qid, s in scores.items():
            # pred_na = na_probs[qid] > na_prob_thresh
            no_prob = (
                self.no_answer_probs[qid]
                if self.no_answer_probs is not None
                else self.default_na_prob
            )
            pred_na = no_prob > self.no_answer_probability_threshold
            if pred_na:
                new_scores[qid] = float(not self.qas_id_to_has_answer[qid])
            else:
                new_scores[qid] = s
        return new_scores

    def make_eval_dict(
        self, exact_scores: Dict[str, float], f1_scores: Dict[str, float], qid_list=None
    ) -> collections.OrderedDict:
        if not qid_list:
            total = len(exact_scores)
            return collections.OrderedDict(
                [
                    ("exact", 100.0 * sum(exact_scores.values()) / total),
                    ("f1", 100.0 * sum(f1_scores.values()) / total),
                    ("total", total),
                ]
            )
        else:
            total = len(qid_list)
            return collections.OrderedDict(
                [
                    ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                    ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                    ("total", total),
                ]
            )

    def merge_eval(
        self, main_eval: Dict[str, float], new_eval: Dict[str, float], prefix: str
    ) -> None:
        for k in new_eval:
            main_eval[f"{prefix}_{k}"] = new_eval[k]

    def _compute(self) -> Dict[str, Dict[str, float]]:

        exact_threshold = self.apply_no_ans_threshold(self.exact_scores)
        f1_threshold = self.apply_no_ans_threshold(self.f1_scores)

        evaluation = self.make_eval_dict(exact_threshold, f1_threshold)

        if self.has_answer_qids:
            has_ans_eval = self.make_eval_dict(
                exact_threshold, f1_threshold, qid_list=self.has_answer_qids
            )
            self.merge_eval(evaluation, has_ans_eval, "HasAns")

        if self.no_answer_qids:
            no_ans_eval = self.make_eval_dict(
                exact_threshold, f1_threshold, qid_list=self.no_answer_qids
            )
            self.merge_eval(evaluation, no_ans_eval, "NoAns")

        if self.no_answer_probs:
            raise NotImplementedError
            # find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

        # return evaluation
        return dict(evaluation)

    def normalize_answer(self, s: str) -> str:
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

    def get_tokens(self, s: str) -> List[str]:
        if not s:
            return []
        return self.normalize_answer(s).split()

    def compute_exact(self, a_gold: str, a_pred: str) -> int:
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold: str, a_pred: str) -> float:
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
