import logging
from typing import Any, List

import pandas as pd
from pytorch_ie.core import Document

from src.metrics.interface import DocumentMetric
from src.taskmodules.coref_hoi_preprocessed import Conll2012OntonotesV5PreprocessedDocument
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

AVAILABLE_METRICS = {
    "lea": lea,
    "muc": muc,
    "bcub": b_cubed,
    "ceafe": ceafe,
    "ceafm": ceafm,
    "blanc": [blancc, blancn],
}

logger = logging.getLogger(__name__)


def convert_doc_to_conllua_lines(document: Document, use_predictions: bool) -> List[str]:
    if not isinstance(document, Conll2012OntonotesV5PreprocessedDocument):
        raise ValueError(
            f"document must be of type Conll2012OntonotesV5PreprocessedDocument to convert to conllua, "
            f"but it is of type {type(document)}"
        )

    cluster_annotations = document.clusters.predictions if use_predictions else document.clusters

    lines = []
    lines.append(
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC IDENTITY BRIDGING DISCOURSE_DEIXIS REFERENCE NOM_SEM"
    )
    lines.append("# newdoc id = " + str(document.id))
    markable_id = 1
    entity_id = 1

    coref_strs = [""] * len(document.tokens)
    if document.subtoken_map is None:
        raise ValueError("subtoken_map is None")

    for clus in cluster_annotations:
        for m in clus.spans:
            subtoken_start = document.subtoken_map[m.start]
            subtoken_end = document.subtoken_map[m.end - 1]

            coref_strs[subtoken_start] += "(EntityID={}|MarkableID=markable_{}".format(
                entity_id, markable_id
            )
            markable_id += 1
            if subtoken_start == subtoken_end:
                coref_strs[subtoken_end] += ")"
            else:
                coref_strs[subtoken_end] = ")" + coref_strs[subtoken_end]

        entity_id += 1

    for _id, token in enumerate(document.tokens):
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
        metrics: List[str] = ["lea", "muc", "bcub", "ceafe", "ceafm", "blanc"],
    ):
        self.keep_singletons = keep_singletons
        self.keep_split_antecedent = keep_split_antecedent
        self.use_MIN = use_MIN
        self.keep_non_referring = keep_non_referring
        self.keep_bridging = keep_bridging
        self.only_split_antecedent = only_split_antecedent
        self.evaluate_discourse_deixis = evaluate_discourse_deixis

        self.show_as_markdown = show_as_markdown

        self.metrics = {k: AVAILABLE_METRICS[k] for k in metrics}

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
        for name, metric in self.metrics.items():
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
