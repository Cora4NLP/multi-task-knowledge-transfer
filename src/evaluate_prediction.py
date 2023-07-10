import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import logging
from typing import Callable, Type, Union

import pandas as pd
from hydra._internal.instantiate._instantiate2 import _resolve_target

from src.document.types import DocumentWithEntitiesRelationsAndLabeledPartitions
from src.serializer import JsonSerializer
from src.utils.metrics import evaluate_document_layer
from src.utils.ua_scorer import eval_coref_ua_scorer

logger = logging.getLogger(__name__)


def get_type_or_callable(type_str: str) -> Union[Type, Callable]:
    return _resolve_target(type_str, full_key="")


def get_document_converter(document_converter: str) -> Callable:
    raise NotImplementedError(f"unknown document converter: {document_converter}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serialized_documents",
        type=str,
        nargs="+",
        required=True,
        help="file name of serialized documents in jsonl format",
    )
    parser.add_argument("--layer", type=str, required=True, help="annotation layer to evaluate")
    parser.add_argument(
        "--label_field",
        type=str,
        default="label",
        help="Compute metrics per label. This requires the layer to contain annotations with that field.",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Do not compute metrics per label. Enable this flag if the layer does not contain annotations "
        "with a label field.",
    )
    parser.add_argument(
        "--document_type",
        type=get_type_or_callable,
        default=DocumentWithEntitiesRelationsAndLabeledPartitions,
        help="document type to load serialized documents",
    )
    parser.add_argument(
        "--exclude_labels",
        type=str,
        nargs="+",
        default=["no_relation"],
        help="labels to exclude from evaluation",
    )
    parser.add_argument(
        "--exclude_annotation_fields",
        type=str,
        nargs="+",
        default=["score"],
        help="annotation fields to exclude from evaluation",
    )
    parser.add_argument(
        "--preprocess_documents",
        type=get_document_converter,
        default=None,
        help="document converter function to preprocess documents",
    )
    parser.add_argument(
        "--ua_scorer",
        type=str,
        default="False",
        help="whether to use the ua-scorer for evaluation",
    )
    parser.add_argument(
        "--coref_gold",
        type=str,
        nargs="+",
        default="None",
        help="file with the golden annotations in the CONLLUA format",
    )

    args = parser.parse_args()

    # show info messages
    logging.basicConfig(level=logging.INFO)

    all_metric_values = []

    for f_i, file_name in enumerate(args.serialized_documents):
        logger.info(f"evaluating {file_name} ...")
        if args.ua_scorer == "True":
            # file_name is the system output
            metric_values = eval_coref_ua_scorer(args.coref_gold[f_i], file_name)
        else:
            documents = JsonSerializer.read(
                file_name=file_name,
                document_type=args.document_type,
            )
            if args.preprocess_documents is not None:
                documents = [args.preprocess_documents(document=document) for document in documents]

            metric_values = evaluate_document_layer(
                path_or_documents=documents,
                layer=args.layer,
                label_field=args.label_field if not args.no_labels else None,
                exclude_labels=args.exclude_labels,
                exclude_annotation_fields=args.exclude_annotation_fields,
            )
        all_metric_values.append(pd.DataFrame(metric_values).T)

    if len(all_metric_values) > 1:
        # mean and stddev over all metric results
        grouped_metric_values = pd.concat(all_metric_values).groupby(level=0)
        logger.info(f"aggregated results (n={len(all_metric_values)}):")
        logger.info(f"\nmean:\n{grouped_metric_values.mean().round(3).to_markdown()}")
        logger.info(f"\nstddev:\n{grouped_metric_values.std().round(3).to_markdown()}")
