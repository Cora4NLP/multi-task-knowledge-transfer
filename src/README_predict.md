
## Prediction for coreference model

```
# run prediction and save to conllua
python src/predict.py dataset=conll2012_ontonotesv5_preprocessed \
dataset.input.base_dataset_kwargs.data_dir=/home/ubuntu/projects/multi-task-knowledge-transfer/data/ontonotes_coref/seg_len_384 \
model_name_or_path=/ext-disk/coref-model-ontonotes/model extras.enforce_tags=false serializer=conllua
```

```
# run prediction and save to jsonlines
python src/predict.py dataset=conll2012_ontonotesv5_preprocessed \
dataset.input.base_dataset_kwargs.data_dir=/home/ubuntu/projects/multi-task-knowledge-transfer/data/ontonotes_coref/seg_len_384 \
model_name_or_path=/ext-disk/coref-model-ontonotes/model extras.enforce_tags=false serializer=json
```


## Evaluation for coreference model

```
# run evaluation with ua-scorer (cfg.serializer=conllua)
python src/evaluate_prediction.py \
--serialized_documents predictions/default/test/2023-07-10_10-32-26/documents.conllua \
--document_type src.taskmodules.coref_hoi_preprocessed.Conll2012OntonotesV5PreprocessedDocument \
--layer clusters --no_labels --ua_scorer True \
--coref_gold data/ontonotes_coref/gold_conllua_seg_len_384/documents-gold.conllua```

```
# run evaluation w/o ua-scorer, F1 based on exact match  (cfg.serializer=json)
python src/evaluate_prediction.py --serialized_documents predictions/default/test/2023-07-10_17-30-57/documents.jsonl --document_type src.taskmodules.coref_hoi_preprocessed.Conll2012OntonotesV5PreprocessedDocument --layer clusters --no_labels --ua_scorer False
```
