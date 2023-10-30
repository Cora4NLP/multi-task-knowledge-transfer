#!/bin/sh

COREF_SCRIPTS_DIR="dataset_builders/pie/conll2012_ontonotesv5_preprocessed"
DATA_DIR="data"
TARGET_DIR="$DATA_DIR/ontonotes_coref"

# download and unpack the data: this will create the directory "$DATA_DIR/conll-2012"
sh $COREF_SCRIPTS_DIR/download_data.sh $DATA_DIR
# combine the annotations and stores them in the conll format
bash $COREF_SCRIPTS_DIR/setup_coref_data.sh $DATA_DIR $TARGET_DIR
# tokenize the input (this requires the Huggingface transformers package!) and convert the files to jsonlines
python $COREF_SCRIPTS_DIR/preprocess.py --input_dir $TARGET_DIR --output_dir $TARGET_DIR/english.384.bert-base-cased --seg_len 384
