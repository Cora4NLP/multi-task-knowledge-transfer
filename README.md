<div align="center">

# Multi-Task Knowledge Transfer

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ChristophAlt/pytorch-ie-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-PyTorch--IE--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## 📌 Description

What it does

## 🚀 Quickstart

### Environment Setup

```bash
# clone project
git clone https://github.com/Cora4NLP/multi-task-knowledge-transfer
cd multi-task-knowledge-transfer

# [OPTIONAL] create conda environment
conda create -n multi-task-knowledge-transfer python=3.9
conda activate multi-task-knowledge-transfer

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# [OPTIONAL] symlink log directories and the default model directory to
# "$HOME/experiments/multi-task-knowledge-transfer" since they can grow a lot
bash setup_symlinks.sh $HOME/experiments/multi-task-knowledge-transfer

# [OPTIONAL] set any environment variables by creating an .env file
# 1. copy the provided example file:
cp .env.example .env
# 2. edit the .env file for your needs!
```

### Data Preparation

#### Relation Extraction

We use the [TACRED dataset](https://nlp.stanford.edu/projects/tacred/).

To use TACRED you have to download it manually. It is available via the LDC
at https://catalog.ldc.upenn.edu/LDC2018T24. Please extract all files in one folder
and set the relevant environment variable: `TACRED_DATA_DIR=[path/to/tacred]/data/json`.

DFKI-internal: On the cluster, use `TACRED_DATA_DIR=/ds/text/tacred/data/json`

#### Named Entity Recognition

We use the [CoNLL 2012 dataset](https://aclanthology.org/W13-3516/), see
[here](https://huggingface.co/datasets/conll2012_ontonotesv5) for the HuggingFace Dataset.
Because of license restrictions you need to download the data manually (e.g. from here
https://data.mendeley.com/public-files/datasets/zmycy7t9h9/files/b078e1c4-f7a4-4427-be7f-9389967831ef/file_downloaded)
and set the environment variable `CONLL2012_ONTONOTESV5_DATA_DIR` to that location (either to the extracted folder
or directly to the zip file).

DFKI-internal: On the cluster, use `CONLL2012_ONTONOTESV5_DATA_DIR=/ds/text/conll-2012`

#### Coreference Resolution

TODO: How to get the data? can we just point to the same as described in the NER section (i.e. https://data.mendeley.com/public-files/datasets/zmycy7t9h9/files/b078e1c4-f7a4-4427-be7f-9389967831ef/file_downloaded)?
TODO: How to preprocess? explanations should use the local version of the preprocessing script in `dataset_builders/pie/conll2012_ontonotesv5_preprocessed`!
TODO: How to set the environment variable `CONLL2012_ONTONOTESV5_PREPROCESSED_DATA_DIR`?

DFKI-internal: On the cluster, use `CONLL2012_ONTONOTESV5_PREPROCESSED_DATA_DIR=/ds/text/cora4nlp/datasets/ontonotes_coref`

##### Extractive Question Answering

We use the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset for extractive question
answering which is fully accessible as a [HuggingFace Dataset](https://huggingface.co/datasets/squad_v2),
so no additional data preparation is required.

### Model Training

**Have a look into the [train.yaml](configs/train.yaml) config to see all available options.**

Train model with default configuration

```bash
# train on CPU
python src/train.py

# train on GPU
python src/train.py trainer=gpu
```

Execute a fast development run (train for two steps)

```bash
python src/train.py +trainer.fast_dev_run=true
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=conll2003
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

Start multiple runs at once (multirun):

```bash
python src/train.py seed=42,43 --multirun
```

Notes:

- this will execute two experiments (one after the other), one for each seed
- the results will be aggregated and stored in `logs/multirun/`, see the last logging output for the exact path

### Model evaluation

This will evaluate the model on the test set of the chosen dataset using the *metrics implemented within the model*.
See [config/dataset/](configs/dataset/) for available datasets.

**Have a look into the [evaluate.yaml](configs/evaluate.yaml) config to see all available options.**

```bash
python src/evaluate.py dataset=conll2003 model_name_or_path=pie/example-ner-spanclf-conll03
```

Notes:

- add the command line parameter `trainer=gpu` to run on GPU

### Inference

This will run inference on the given dataset and split. See [config/dataset/](configs/dataset/) for available datasets.
The result documents including the predicted annotations will be stored in the `predictions/` directory (exact
location will be printed to the console).

**Have a look into the [predict.yaml](configs/predict.yaml) config to see all available options.**

```bash
python src/predict.py dataset=conll2003 model_name_or_path=pie/example-ner-spanclf-conll03
```

Notes:

- add the command line parameter `+pipeline.device=0` to run the inference on GPU 0

### Evaluate Serialized Documents

This will evaluate serialized documents including predicted annotations (see [Inference](#inference)) using a
*document metric*. See [config/metric/](configs/metric/) for available metrics.

**Have a look into the [evaluate_documents.yaml](configs/evaluate_documents.yaml) config to see all available options**

```bash
python src/evaluate_documents.py metric=f1 metric.layer=entities +dataset.data_dir=PATH/TO/DIR/WITH/SPLITS
```

Note: By default, this utilizes the dataset provided by the
[from_serialized_documents](configs/dataset/from_serialized_documents.yaml) configuration. This configuration is
designed to facilitate the loading of serialized documents, as generated during the [Inference](#inference) step. It
requires to set the parameter `data_dir`. If you want to use a different dataset,
you can override the `dataset` parameter as usual with any existing dataset config, e.g `dataset=conll2003`. But
calculating the F1 score on the bare `conll2003` dataset does not make much sense, because it does not contain any
predictions. However, it could be used with statistical metrics such as
[count_text_tokens](configs/metric/count_text_tokens.yaml) or
[count_entity_labels](configs/metric/count_entity_labels.yaml).

## Pre-trained Models

### Coreference

The pre-trained coreference model in `models/pretrained/bert-base-cased-coref-hoi` is trained on the CoNLL2012 [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) data using the official code from the [coref-hoi repository](https://github.com/lxucs/coref-hoi/) as follows:

```bash
python run.py bert_base 0
```

`bert_base` is the configuration defined below and 0 is the gpu id.

```bash
bert_base = ${best}{
  num_docs = 2802
  trn_data_path = /ds/text/ontonotes_coref/seg_len_384/train.english.384.jsonlines
  dev_data_path = /ds/text/ontonotes_coref/seg_len_384/dev.english.384.jsonlines
  tst_data_path = /ds/text/ontonotes_coref/seg_len_384/test.english.384.jsonlines
  bert_learning_rate = 1e-05
  task_learning_rate = 2e-4
  max_segment_len = 384
  ffnn_size = 3000
  cluster_ffnn_size = 3000
  max_training_sentences = 11
  bert_tokenizer_name = bert-base-cased
  bert_pretrained_name_or_path = bert-base-cased
}
```

The `bert_base` configuration refers to the `best` configuration which specifies additional parameters, it is defined in the coref-hoi repository: [https://github.com/lxucs/coref-hoi/blob/master/experiments.conf](https://github.com/lxucs/coref-hoi/blob/master/experiments.conf)

## Development

```bash
# run pre-commit: code formatting, code analysis, static type checking, and more (see .pre-commit-config.yaml)
pre-commit run -a

# run tests
pytest -k "not slow" --cov --cov-report term-missing
```
