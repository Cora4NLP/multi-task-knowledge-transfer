import logging
from typing import Any, Dict, List

import pandas as pd
from pytorch_ie.core import Document

from src.metrics.interface import DocumentMetric
from src.utils.coval.eval.evaluator import (
    b_cubed,
    blancc,
    blancn,
    ceafe,
    ceafm,
    evaluate_documents,
    evaluate_non_referrings,
    lea,
    muc,
)
from src.utils.coval.ua.reader import get_coref_infos

logger = logging.getLogger(__name__)


def get_clusters(
    mention_predictions: List[dict], cluster_predictions: List[dict]
) -> List[List[List[int]]]:
    mention_ids = {}
    for mention in mention_predictions:
        mention_ids[mention["_id"]] = [mention["start"], mention["end"] - 1]
    clusters = []
    for el in cluster_predictions:
        cluster_mentions = []
        for span_id in el["spans"]:
            cluster_mentions.append(mention_ids[span_id])
        clusters.append(cluster_mentions)

    return clusters


def convert_doc_to_conllua_lines(document: Document, use_predictions: bool) -> List[str]:
    predictions_or_annotations = "predictions" if use_predictions else "annotations"
    orig_doc = document.asdict()
    clusters = get_clusters(
        orig_doc["mentions"][predictions_or_annotations],
        orig_doc["clusters"][predictions_or_annotations],
    )
    doc_as_dict = {
        "doc_key": orig_doc["id"],
        "tokens": orig_doc["tokens"],
        "sentences": orig_doc["sentences"],
        "speakers": [],
        "constituents": [],
        "ner": [],
        "clusters": clusters,
        "sentence_map": orig_doc["sentence_map"],
        "subtoken_map": orig_doc["subtoken_map"],
        "pronouns": [],
    }

    pred_clusters = [tuple(tuple(m) for m in cluster) for cluster in doc_as_dict["clusters"]]

    lines = []
    lines.append(
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC IDENTITY BRIDGING DISCOURSE_DEIXIS REFERENCE NOM_SEM"
    )
    lines.append("# newdoc id = " + doc_as_dict["doc_key"])
    markable_id = 1
    entity_id = 1

    coref_strs = [""] * len(doc_as_dict["tokens"])

    for clus in pred_clusters:
        for (start, end) in clus:
            start = doc_as_dict["subtoken_map"][start]
            end = doc_as_dict["subtoken_map"][end]

            coref_strs[start] += "(EntityID={}|MarkableID=markable_{}".format(
                entity_id, markable_id
            )
            markable_id += 1
            if start == end:
                coref_strs[end] += ")"
            else:
                coref_strs[end] = ")" + coref_strs[end]

        entity_id += 1

    for _id, token in enumerate(doc_as_dict["tokens"]):
        if coref_strs[_id] == "":
            coref_strs[_id] = "_"
        sentence = "{}  {}  _  _  _  _  _  _  _  _  {}  _  _  _  _".format(
            _id, token, coref_strs[_id]
        )
        lines.append(sentence)

    return lines


class CorefMetrics(DocumentMetric):
    def __init__(
        self,
        keep_singletons: bool = True,
        keep_split_antecedent: bool = True,
        use_MIN: bool = False,
        keep_non_referring: bool = False,
        keep_bridging: bool = False,
        only_split_antecedent: bool = False,
        evaluate_discourse_deixis: bool = False,
        show_as_markdown: bool = False,
    ):
        metric_dict = {
            "lea": lea,
            "muc": muc,
            "bcub": b_cubed,
            "ceafe": ceafe,
            "ceafm": ceafm,
            "blanc": [blancc, blancn],
        }
        self.metrics = [(k, metric_dict[k]) for k in metric_dict]
        # TA: pass these as parameters (too many?)
        # or create a new config for coreference evaluation
        self.keep_singletons = keep_singletons
        self.keep_split_antecedent = keep_split_antecedent
        self.use_MIN = use_MIN
        self.keep_non_referring = keep_non_referring
        self.keep_bridging = keep_bridging
        self.only_split_antecedent = only_split_antecedent
        self.evaluate_discourse_deixis = evaluate_discourse_deixis

        self.show_as_markdown = show_as_markdown

        self.reset()

    def reset(self):
        self.gold_lines = []
        self.sys_lines = []

    def _update(self, document: Document) -> None:
        self.gold_lines.extend(convert_doc_to_conllua_lines(document, use_predictions=False))
        self.sys_lines.extend(convert_doc_to_conllua_lines(document, use_predictions=True))

    def _values(self) -> Any:

        doc_coref_infos, doc_non_referring_infos, doc_bridging_infos = get_coref_infos(
            self.gold_lines,
            self.sys_lines,
            self.keep_singletons,
            self.keep_split_antecedent,
            self.keep_bridging,
            self.keep_non_referring,
            self.evaluate_discourse_deixis,
            self.use_MIN,
            print_debug=True,
        )

        metrics_dict = {}
        for name, metric in self.metrics:
            recall, precision, f1 = evaluate_documents(
                doc_coref_infos, metric, beta=1, only_split_antecedent=self.only_split_antecedent
            )
            metrics_dict[name] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }
        if self.show_as_markdown:
            logger.info(f"evaluation:\n{pd.DataFrame(metrics_dict).T.to_markdown()}")

        return metrics_dict
