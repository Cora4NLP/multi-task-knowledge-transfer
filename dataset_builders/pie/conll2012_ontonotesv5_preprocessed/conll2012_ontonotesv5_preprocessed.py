import dataclasses
import logging
from typing import Any, Dict, List, Optional

import datasets
import pytorch_ie.data.builder
from pytorch_ie.core import Document

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Conll2012OntonotesV5PreprocessedDocument(Document):
    tokens: List[str]
    sentences: List[List[str]]
    speakers: List[List[str]]
    clusters: List[List[List[int]]]
    sentence_map: List[int]
    subtoken_map: Optional[List[int]] = None
    id: Optional[str] = None  # doc_key
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # these are not needed:
    # constituents: List
    # ner: List
    # pronouns: List


def example_to_document(
    example: Dict[str, Any],
) -> Conll2012OntonotesV5PreprocessedDocument:
    document = Conll2012OntonotesV5PreprocessedDocument(
        id=example["doc_key"],
        tokens=example["tokens"],
        sentences=example["sentences"],
        speakers=example["speakers"],
        clusters=example["clusters"],
        sentence_map=example["sentence_map"],
        subtoken_map=example["subtoken_map"],
    )
    return document


def document_to_example(
    document: Conll2012OntonotesV5PreprocessedDocument,
) -> Dict[str, Any]:
    example = {
        "doc_key": document.id,
        "tokens": document.tokens,
        "sentences": document.sentences,
        "speakers": document.speakers,
        "clusters": document.clusters,
        "sentence_map": document.sentence_map,
        "subtoken_map": document.subtoken_map,
    }
    return example


class Conll2012OntonotesV5PreprocessedConfig(datasets.BuilderConfig):
    """BuilderConfig for CDCP."""

    def __init__(self, **kwargs):
        """BuilderConfig for CDCP.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class Conll2012Ontonotesv5Preprocessed(pytorch_ie.data.builder.ArrowBasedBuilder):
    DOCUMENT_TYPE = Conll2012OntonotesV5PreprocessedDocument

    BASE_DATASET_PATH = "json"

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]

    DEFAULT_CONFIG_NAME = "default"  # type: ignore

    def _generate_document(self, example):
        return example_to_document(example)
