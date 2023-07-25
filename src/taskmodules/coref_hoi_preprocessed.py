"""
workflow:
    document
        -> (input_encoding, target_encoding) -> task_encoding
            -> model_encoding -> model_output
        -> task_output
    -> document
"""
import dataclasses
import logging
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
from pytorch_ie.annotations import Label, Span
from pytorch_ie.core import (
    Annotation,
    AnnotationList,
    Document,
    TaskEncoding,
    TaskModule,
    annotation_field,
)
from transformers import AutoTokenizer
from typing_extensions import TypeAlias

from src.models.coref_hoi import (
    CorefHoiModelInputs,
    CorefHoiModelPrediction,
    CorefHoiModelStepBatchEncoding,
    CorefHoiModelTargets,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(eq=True, frozen=True)
class SpanSet(Annotation):
    spans: Tuple[Span, ...]
    score: float = 1.0

    def __post_init__(self) -> None:
        # make the referenced spans unique, sort them and convert to tuples to make everything hashable
        object.__setattr__(
            self,
            "spans",
            tuple(sorted(set(s for s in self.spans), key=lambda s: (s.start, s.end))),
        )


@dataclasses.dataclass
class Conll2012OntonotesV5PreprocessedDocument(Document):
    tokens: List[str]
    sentences: List[List[str]]
    speakers: List[List[str]]
    sentence_map: List[int]
    subtoken_map: Optional[List[int]] = None
    mentions: AnnotationList[Span] = annotation_field(target="tokens")
    clusters: AnnotationList[SpanSet] = annotation_field(target="mentions")
    id: Optional[str] = None  # doc_key
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class TaskInputs(TypedDict, total=False):
    input_ids: np.ndarray
    input_mask: np.ndarray
    speaker_ids: np.ndarray
    sentence_len: np.ndarray
    genre: int
    sentence_map: List[int]


class TaskTargets(TypedDict, total=False):
    gold_starts: np.ndarray
    gold_ends: np.ndarray
    gold_mention_cluster_map: np.ndarray


TaskEncodingType: TypeAlias = TaskEncoding[
    Conll2012OntonotesV5PreprocessedDocument, TaskInputs, TaskTargets
]
TaskModuleType: TypeAlias = TaskModule[
    Conll2012OntonotesV5PreprocessedDocument,
    TaskInputs,
    TaskTargets,
    CorefHoiModelStepBatchEncoding,
    CorefHoiModelPrediction,
    # this is the same because the model does not allow batching
    CorefHoiModelPrediction,
]


def flatten(lst: List):
    return [item for sublist in lst for item in sublist]


def list_of_dicts_to_dict_of_lists(
    list_of_dicts: List[Dict[str, Any]], keys: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    if keys is None:
        keys = list(list_of_dicts[0].keys())
    dict_of_lists = {}
    for key in keys:
        dict_of_lists[key] = [dic[key] for dic in list_of_dicts]
    return dict_of_lists


def construct_inputs_from_list_of_dicts(list_of_dicts):
    as_dict = list_of_dicts_to_dict_of_lists(
        list_of_dicts=list_of_dicts,
        keys=[
            "input_ids",
            "input_mask",
            "speaker_ids",
            "sentence_len",
            "genre",
            "sentence_map",
        ],
    )
    return CorefHoiModelInputs(
        input_ids=torch.tensor(np.stack(as_dict["input_ids"]), dtype=torch.long),
        input_mask=torch.tensor(np.stack(as_dict["input_mask"]), dtype=torch.long),
        speaker_ids=torch.tensor(np.stack(as_dict["speaker_ids"]), dtype=torch.long),
        sentence_len=torch.tensor(np.stack(as_dict["sentence_len"]), dtype=torch.long),
        genre=torch.tensor(as_dict["genre"], dtype=torch.long),
        sentence_map=torch.tensor(as_dict["sentence_map"], dtype=torch.long),
    )


def construct_targets_from_list_of_dicts(list_of_dicts):
    as_dict = list_of_dicts_to_dict_of_lists(
        list_of_dicts, keys=["gold_starts", "gold_ends", "gold_mention_cluster_map"]
    )
    return CorefHoiModelTargets(
        gold_starts=torch.tensor(np.stack(as_dict["gold_starts"]), dtype=torch.long),
        gold_ends=torch.tensor(np.stack(as_dict["gold_ends"]), dtype=torch.long),
        gold_mention_cluster_map=torch.tensor(
            np.stack(as_dict["gold_mention_cluster_map"]), dtype=torch.long
        ),
    )


@TaskModule.register()
class CorefHoiPreprocessedTaskModule(TaskModuleType):
    # If these attributes are set, the taskmodule is considered as prepared. They should be calculated
    # within _prepare() and are dumped automatically when saving the taskmodule with save_pretrained().
    # PREPARED_ATTRIBUTES = ["label_to_id"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        genres: List[str],
        max_num_speakers: int,
        max_segment_len: int,
        max_training_sentences: int,
        **kwargs,
    ) -> None:
        # Important: Remaining keyword arguments need to be passed to super.
        super().__init__(**kwargs)
        # Save all passed arguments. They will be available via self._config().
        self.save_hyperparameters()
        self.max_training_sentences = max_training_sentences
        self.genres = genres
        self.max_num_speakers = max_num_speakers
        self.max_segment_len = max_segment_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # Will be used in evaluation
        self.stored_info: Dict[str, Any] = {}
        self.stored_info["tokens"] = {}  # {doc_key: ...}
        self.stored_info["subtoken_maps"] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info["gold"] = {}  # {doc_key: ...}
        self.stored_info["genre_dict"] = {genre: idx for idx, genre in enumerate(genres)}

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {"UNK": 0, "[SPL]": 1}
        for speaker in speakers:
            if len(speaker_dict) > self.max_num_speakers:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example_inputs(
        self,
        speakers,
        sentences,
        sentence_map,
        doc_key,
        tokens,
        subtoken_map=None,
    ) -> TaskInputs:
        # Speakers
        speaker_dict = self._get_speaker_dict(flatten(speakers))

        # Sentences/segments
        # sentences = example["sentences"]  # Segments
        # sentence_map = example["sentence_map"]
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.max_segment_len
        sentence_len = np.array([len(s) for s in sentences])

        # Bert input
        input_ids, input_mask, speaker_ids = [], [], []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Keep info to store
        self.stored_info["subtoken_maps"][doc_key] = subtoken_map
        self.stored_info["tokens"][doc_key] = tokens

        # Construct example
        genre = self.stored_info["genre_dict"].get(doc_key[:2], 0)

        return TaskInputs(
            input_ids=input_ids,
            input_mask=input_mask,
            speaker_ids=speaker_ids,
            sentence_len=sentence_len,
            genre=genre,
            sentence_map=sentence_map,
        )

    def encode_input(
        self,
        document: Conll2012OntonotesV5PreprocessedDocument,
        is_training: bool = False,
    ) -> TaskEncodingType:
        """Create one or multiple task encodings for the given document."""

        inputs = self.tensorize_example_inputs(
            speakers=document.speakers,
            sentences=document.sentences,
            sentence_map=document.sentence_map,
            doc_key=document.id,
            subtoken_map=document.subtoken_map,
            tokens=document.tokens,
        )

        return TaskEncoding(
            document=document,
            inputs=inputs,
        )

    def tensorize_example_targets(self, doc_key, clusters: Iterable[SpanSet]) -> TaskTargets:
        # Mentions and clusters
        cluster_list = [
            [(span.start, span.end - 1) for span in cluster.spans] for cluster in clusters
        ]
        self.stored_info["gold"][doc_key] = cluster_list
        gold_mentions = sorted(tuple(mention) for mention in flatten(cluster_list))
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        for cluster_id, cluster in enumerate(cluster_list):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1

        gold_starts, gold_ends = self._tensorize_spans(gold_mentions)

        return TaskTargets(
            gold_starts=gold_starts,
            gold_ends=gold_ends,
            gold_mention_cluster_map=gold_mention_cluster_map,
        )

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TaskTargets:
        """Create a target for a task encoding.

        This may use any annotations of the underlying document.
        """

        targets = self.tensorize_example_targets(
            doc_key=task_encoding.document.id,
            clusters=task_encoding.document.clusters,
        )

        return targets

    def get_truncation_offsets(self, input_ids, sentence_len, sentence_offset=None):
        max_sentences = self.max_training_sentences
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences

        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - max_sentences)
        word_offset = sentence_len[:sent_offset].sum()
        return sent_offset, word_offset

    def truncate_example_inputs(
        self,
        sent_offset,
        word_offset,
        input_ids,
        input_mask,
        speaker_ids,
        sentence_len,
        genre,
        sentence_map,
    ) -> TaskInputs:
        max_sentences = self.max_training_sentences
        num_words = sentence_len[sent_offset : sent_offset + max_sentences].sum()

        input_ids = input_ids[sent_offset : sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset : sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset : sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset : sent_offset + max_sentences]

        sentence_map = sentence_map[word_offset : word_offset + num_words]

        return TaskInputs(
            input_ids=input_ids,
            input_mask=input_mask,
            speaker_ids=speaker_ids,
            sentence_len=sentence_len,
            genre=genre,
            sentence_map=sentence_map,
        )

    def truncate_example_targets(
        self,
        word_offset,
        sent_offset,
        gold_starts,
        gold_ends,
        gold_mention_cluster_map,
    ) -> TaskTargets:
        max_sentences = self.max_training_sentences

        gold_starts = gold_starts[sent_offset : sent_offset + max_sentences] - word_offset
        gold_ends = gold_ends[sent_offset : sent_offset + max_sentences] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[
            sent_offset : sent_offset + max_sentences
        ]

        return TaskTargets(
            gold_starts=gold_starts,
            gold_ends=gold_ends,
            gold_mention_cluster_map=gold_mention_cluster_map,
        )

    def collate(
        self, task_encodings: Sequence[TaskEncodingType]
    ) -> CorefHoiModelStepBatchEncoding:
        """Convert a list of task encodings to a batch that will be passed to the model."""

        is_training = task_encodings[0].has_targets

        if not is_training:
            model_inputs = construct_inputs_from_list_of_dicts(
                [task_encoding.inputs for task_encoding in task_encodings]
            )
            return model_inputs, None

        all_inputs = []
        all_targets = []
        for task_encoding in task_encodings:
            inputs = task_encoding.inputs
            targets = task_encoding.targets
            if is_training and len(task_encoding.document.sentences) > self.max_training_sentences:
                sent_offset, word_offset = self.get_truncation_offsets(
                    task_encoding.inputs["input_ids"], task_encoding.inputs["sentence_len"]
                )
                inputs = self.truncate_example_inputs(sent_offset, word_offset, **inputs)
                targets = self.truncate_example_targets(word_offset, sent_offset, **targets)
            all_inputs.append(inputs)
            all_targets.append(targets)

        model_inputs = construct_inputs_from_list_of_dicts(all_inputs)
        model_targets = construct_targets_from_list_of_dicts(all_targets)
        return model_inputs, model_targets

    def unbatch_output(
        self, model_output: CorefHoiModelPrediction
    ) -> Sequence[CorefHoiModelPrediction]:
        """Convert one model output batch to a sequence of taskmodule outputs."""
        # the model works only with batch_size 1, so we can just return the whole elements
        return [model_output]

    def create_annotations_from_output(
        self,
        task_encodings: TaskEncodingType,
        task_outputs: CorefHoiModelPrediction,
    ) -> Iterator[Tuple[str, Label]]:
        """Convert a task output to annotations.

        The method has to yield tuples (annotation_layer_name, annotation).
        """

        for raw_spans in task_outputs["clusters"]:
            spans = tuple(Span(start=start, end=end + 1) for start, end in raw_spans)
            for span in spans:
                yield "mentions", span
            yield "clusters", SpanSet(spans=spans, score=1.0)
