# based on https://github.com/juntaoy/codi-crac2022_scripts/blob/main/helper.py
import json
import os
from typing import Dict, List, Sequence, Type, TypeVar

from pytorch_ie.core import Document

from src.serializer.preprocess_conllua import get_document, get_all_docs, get_tokenizer

from src.serializer.interface import DocumentSerializer
from src.utils import get_pylogger


log = get_pylogger(__name__)

D = TypeVar("D", bound=Document)


def as_json_lines(file_name: str) -> bool:
    if file_name.lower().endswith(".jsonl"):
        return True
    elif file_name.lower().endswith(".json"):
        return False
    else:
        raise Exception(f"unknown file extension: {file_name}")


class ConllUaSerializer(DocumentSerializer):
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def __call__(self, documents: Sequence[Document]) -> Dict[str, str]:
        return self.dump(documents=documents, path=self.path, **self.kwargs)

    @classmethod
    def convert_json_to_ua(cls, doc_as_dict: Dict) -> List[str]:
        pred_clusters = [tuple(tuple(m) for m in cluster) for cluster in doc_as_dict["clusters"]]

        lines = []
        lines.append("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC IDENTITY BRIDGING DISCOURSE_DEIXIS REFERENCE NOM_SEM")
        lines.append("# newdoc id = " + doc_as_dict['doc_key'])
        markable_id = 1
        entity_id = 1

        coref_strs = [""]*len(doc_as_dict['tokens'])

        for clus in pred_clusters:
            for (start,end) in clus:
                start = doc_as_dict['subtoken_map'][start]
                end = doc_as_dict['subtoken_map'][end]

                coref_strs[start] += "(EntityID={}|MarkableID=markable_{}".format(entity_id, markable_id)
                markable_id += 1
                if start == end:
                    coref_strs[end] += ")"
                else:
                    coref_strs[end] = ")" + coref_strs[end]

            entity_id += 1

        for _id, token in enumerate(doc_as_dict['tokens']):
            if coref_strs[_id] == "":
                coref_strs[_id] = "_"
            sentence = "{}  {}  _  _  _  _  _  _  _  _  {}  _  _  _  _".format(_id, token, coref_strs[_id])
            lines.append(sentence)

        return lines

    @classmethod
    def get_clusters(cls, mention_predictions: List[dict], cluster_predictions: List[dict]) -> List[List[int]]:
        clusters = []
        mention_ids = {}
        for mention in mention_predictions:
            mention_ids[mention["_id"]] = [mention["start"], mention["end"]-1]
        clusters = []
        for el in cluster_predictions:
            cluster_mentions = []
            for span_id in el["spans"]:
                cluster_mentions.append(mention_ids[span_id])
            clusters.append(cluster_mentions)

        return clusters


    @classmethod
    def dump(cls, documents: Sequence[Document], path: str, **kwargs) -> Dict[str, str]:
        realpath = os.path.realpath(path)
        log.info(f'serialize documents to "{realpath}" ...')
        dir_path = os.path.dirname(realpath)
        os.makedirs(dir_path, exist_ok=True)

        conllua_docs = []
        for doc in documents:
            orig_doc = doc.asdict()
            clusters = cls.get_clusters(orig_doc["mentions"]["predictions"], orig_doc["clusters"]["predictions"])
            conll_doc_as_dict = {"doc_key":orig_doc["id"], "tokens":orig_doc["tokens"],\
            "sentences":orig_doc["sentences"], "speakers":[], "constituents":[], "ner":[],\
            "clusters":clusters, "sentence_map":orig_doc["sentence_map"],\
            "subtoken_map":orig_doc["subtoken_map"], "pronouns":[]}
            conllua_docs += cls.convert_json_to_ua(conll_doc_as_dict) + ["\n"]
        with open(realpath, "w") as f:
            for doc in conllua_docs:
                f.write(doc + "\n")

        return {"path": realpath}

    @classmethod # reading jsonlines
    def read(cls, file_name: str, document_type: Type[D]) -> List[D]:
        documents = []
        if as_json_lines(str(file_name)):
            with open(file_name) as f:
                for line in f:
                    json_dict = json.loads(line)
                    documents.append(document_type.fromdict(json_dict))
        else:
            with open(file_name) as f:
                json_list = json.load(f)
            for json_dict in json_list:
                documents.append(document_type.fromdict(json_dict))
        return documents

    @classmethod # reading jsonlines
    def read_conllua2dict(cls, file_name: str, document_type: Type[D], segment_size=384) -> List[D]:
        documents = []
        key_docs, key_doc_sents = get_all_docs(file_name)
        tokenizer = get_tokenizer("bert-base-cased")

        for doc in key_doc_sents:
            document = get_document(doc, key_docs[doc], 'english', segment_size, tokenizer)
            documents.append(document_type.fromdict(document))
        return documents
