<div align="center">

# Prediction for coreference model

##

```
# run prediction
python src/evaluate_prediction.py --serialized_documents predictions/default/test/2023-07-10_10-32-26/documents.conllua --document_type src.taskmodules.coref_hoi_preprocessed.Conll2012OntonotesV5PreprocessedDocument --layer clusters --no_labels --ua_scorer True --coref_gold data/ontonotes_coref/gold_conllua_seg_len_384/documents-gold.conllua cfg.serializer=conllua
```
