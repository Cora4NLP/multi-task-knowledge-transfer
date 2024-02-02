# Experimentation Log

This file is meant to log the development and experimentation process of this project. Please log relevant experiments
including at least a short motivation, the applied commands, and a summary of any relevant outcome or follow-up ideas.

**Usage:** If you want to add content, please create a new section with the current date as heading (if not yet available),
maybe add a subsection with your topic (e.g. NER, CoRef, General, Aggregation, ...) and structure your content in a
meaning full way, e.g.

```markdown
## 2023-07-27

### Relation Extraction

- short experiment to verify that the code for the simple multi-model variant is working
  - preparation: implemented the multi-model variant of the text classification model
  - command: `python src/train.py experiment=tacred +trainer.fast_dev_run=true`
  - wandb (weights & biases) run: https://wandb.ai/arne/pie-example-scidtb/runs/2rsl4z9p (this is just an example!)
  - artefacts, e.g.
    - model location: `path/to/model/dir`, or
    - serialized documents (in the case of inference): `path/to/serialized/documents.jsonl`
  - metric values:
    |          |    f1 |     p |     r |
    | :------- | ----: | ----: | ----: |
    | MACRO    | 0.347 | 0.343 | 0.354 |
    | MICRO    | 0.497 | 0.499 | 0.494 |
  - outcome: the code works
```

IMPORTANT: Execute `pre-commit run -a` before committing to ensure that the markdown is formatted correctly.

## 2023-08-01

### Coreference Resolution: BERT Baseline

- running a single model with bert-base-cased
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi \
        trainer=gpu \
        seed=1,2,3 \
        --multirun
    ```
  - wandb (weights & biases) run:
    - seed1: https://wandb.ai/tanikina/conll2012-coref_hoi-training/runs/in108i6l
    - seed2: https://wandb.ai/tanikina/conll2012-coref_hoi-training/runs/96jm8gzd
    - seed3: https://wandb.ai/tanikina/conll2012-coref_hoi-training/runs/57em3kzx
  - artefacts
    - model location:
      - seed1: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/coref_hoi/2023-08-01_12-39-09`
      - seed2: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/coref_hoi/2023-08-01_15-21-06`
      - seed3: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/coref_hoi/2023-08-01_18-03-12`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 | val/loss |
    | :---- | -------: | ---------------: | ------------------: | ------: | -------: |
    | seed1 |   0.6743 |          108.659 |               56039 | 0.62805 |  168.901 |
    | seed2 |  0.67679 |          106.272 |               56039 | 0.63836 |  163.246 |
    | seed3 |  0.67741 |          108.461 |               56039 | 0.63788 |  177.884 |

### Coreference Resolution: BERT, Re-TACRED and NER (Re-TACRED and NER frozen, sum aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (frozen)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        model.aggregate=sum \
        seed=1 \
        +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes]
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8gurkh4p
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_12-33-45`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | ------: | -------: |
    | run1 |  0.45232 |          4449.93 |               56039 | 0.46102 |  4397.89 |

### Coreference Resolution: BERT, Re-TACRED and NER (all trainable, sum aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (trainable, sum)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        model.aggregate=sum \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pwymsi30
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_12-31-28`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.50 |         1855.579 |               56039 |  0.552 | 1533.741 |

### Coreference Resolution: BERT, Re-TACRED and NER (Re-TACRED and NER frozen, mean aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (frozen, mean)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        model.aggregate=mean \
        seed=1 \
        +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes]
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8a6cvg1c
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_10-22-22`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.47 |          523.931 |               56039 |  0.396 |  698.734 |

### Coreference Resolution: BERT, Re-TACRED and NER (all trainable, mean aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (trainable, mean)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dqb2bpel
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_10-43-45`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |    0.521 |          207.861 |               56039 |  0.555 |  203.024 |

### Coreference Resolution: BERT and NER (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and NER (trainable, mean)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        seed=1
    ```
    NB: manually removed re-tacred from pretrained_models
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yj825oq3
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_11-10-27`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 |  val/loss |
    | ---: | -------: | ---------------: | ------------------: | ------: | --------: |
    | run1 |    0.538 |        503.12421 |               56039 | 0.59018 | 481.17889 |

### Coreference Resolution: BERT and NER (NER frozen, mean aggregation)

- running a multi-model with bert-base-cased and NER (frozen, mean)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        seed=1 \
        +model.freeze_models=[bert-base-cased-ner-ontonotes]
    ```
    NB: manually removed re-tacred from pretrained_models
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/2f63dfmx
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_11-12-31`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 |  val/loss |
    | ---: | -------: | ---------------: | ------------------: | ------: | --------: |
    | run1 |  0.52739 |        583.81372 |               56039 | 0.59065 | 574.26263 |

### Coreference Resolution: BERT and Re-TACRED (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and Re-TACRED (trainable, mean)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        seed=1
    ```
    NB: manually removed ner from pretrained_models
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/njay9nzj
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_12-26-13`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | ------: | -------: |
    | run1 |  0.59548 |         88.57339 |               56039 | 0.62288 | 88.99255 |

### Coreference Resolution: BERT and Re-TACRED (Re-TACRED frozen, mean aggregation)

- running a multi-model with bert-base-cased and Re-TACRED (frozen, mean)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        seed=1 \
        +model.freeze_models=[bert-base-cased-re-tacred]
    ```
    NB: manually removed ner from pretrained_models
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/fthubusx
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-02_19-39-02`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 |  val/loss |
    | ---: | -------: | ---------------: | ------------------: | ------: | --------: |
    | run1 |  0.47011 |        515.68866 |               56039 | 0.56734 | 433.13028 |

## 2023-08-24

### Coreference Resolution: triple BERT (3 times bert-base-cased, mean aggregation)

- running a multi-model with bert-base-cased (3x)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_bert \
        trainer=gpu \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/44daf336
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_14-02-23`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.71 |          106.787 |               56039 |  0.632 |  213.828 |

### Coreference Resolution: BERT, Re-TACRED, NER and SQUAD (all trainable, mean aggregation)

- running a multi-model with bert-base-cased, Re-TACRED, NER and SQUAD
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        model.aggregate=mean \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/kk817tye
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_13-29-52`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.56 |          199.767 |               56039 |  0.613 |  205.018 |

### Coreference Resolution: BERT and SQUAD (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-qa-squad2
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_bert_qa \
        trainer=gpu \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/81089nj3
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_16-08-08`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.57 |          479.938 |               56039 |  0.607 |   496.05 |

### Coreference Resolution: BERT and SQUAD (SQUAD frozen, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-qa-squad2
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_bert_qa \
        trainer=gpu \
        seed=1 \
        +model.freeze_models=[bert-base-cased-qa-squad2]
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/n561w8zl
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_16-03-48`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.57 |          497.308 |               56039 |  0.558 |   608.34 |

### Coreference Resolution: BERT and MRPC (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-mrpc
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_mrpc \
        trainer=gpu \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dv7stxmf
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_09-40-18`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.67 |          116.573 |               56039 |  0.638 |   169.64 |

### Coreference Resolution: BERT and MRPC (MRPC frozen, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-mrpc (frozen)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_mrpc \
        trainer=gpu \
        seed=1 \
        +model.freeze_models=[bert-base-cased-mrpc]
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/tq346u1c
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_09-44-17`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.64 |          255.440 |               56039 |  0.630 |   342.60 |

### Coreference Resolution: BERT pre-trained on coreference (all trainable, mean aggregation)

- running a multi-model with 3 pre-trained coreference BERT models, no bert-base-cased (commented out in the configuration file)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_coref \
        trainer=gpu \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ynkldn9e
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_09-50-48`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.73 |           53.306 |               56039 |  0.655 |   105.85 |

### Coreference Resolution: BERT pre-trained on coreference + bert-base-cased (all trainable, mean aggregation)

- running a multi-model with 3 pre-trained coreference BERT models + bert-base-cased
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_coref \
        trainer=gpu \
        seed=1
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3dwloytn
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_18-30-25`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.73 |           58.478 |               56039 |  0.651 |   121.52 |

### Coreference Resolution: BERT pre-trained on coreference + bert-base-cased (3 models are frozen, mean aggregation)

- running a multi-model with 3 pre-trained coreference BERT models (frozen) + bert-base-cased
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_coref \
        trainer=gpu \
        seed=1 \
        +model.freeze_models=[bert-base-cased-coref1,bert-base-cased-coref2,bert-base-cased-coref3]
    ```
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zx3ytipi
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_18-31-40`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.73 |           88.203 |               56039 |  0.636 |   202.80 |

## 2023-08-31

### Coreference Resolution: BERT, Re-TACRED, NER and SQUAD (all trainable, attention aggregation)

- running a multi-model with 3 pre-trained models + bert-base-cased, attention
  - command:

    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.mode_keys=token,cls,constant \
        seed=1
    ```

  - wandb (weights & biases) run:

    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/wp7vh69x
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qiamto3a
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8wwo57vu

  - artefacts

    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_14-08-43`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_00-19-33`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_10-57-56`

  - metric values:

    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.63 |           58.520 |               56039 |   0.64 |    72.87 | attention |     token |
    | seed1 |     0.62 |           59.893 |               56039 |   0.63 |    69.88 | attention |       cls |
    | seed1 |     0.58 |           69.269 |               56039 |   0.62 |    76.88 | attention |  constant |

### Coreference Resolution: BERT, Re-TACRED, NER and SQUAD (Re-TACRED, NER and SQUAD frozen, attention aggregation)

- running a multi-model with 3 pre-trained frozen models + bert-base-cased, attention

  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.mode_keys=token,cls,constant \
        +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
        seed=1
    ```
  - wandb (weights & biases) run:
    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/u7zwuuhc
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/2932bvxz
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3swxyqt9
  - artefacts
    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_15-20-52`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_22-20-39`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_05-23-14`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.58 |           87.707 |               56039 |   0.61 |    84.60 | attention |     token |
    | seed1 |     0.57 |           90.813 |               56039 |   0.59 |    93.08 | attention |       cls |
    | seed1 |     0.51 |           117.10 |               56039 |   0.56 |   105.37 | attention |  constant |

### Coreference Resolution: BERT, Re-TACRED (all trainable, attention aggregation)

- running a multi-model with Re-TACRED pre-trained model + bert-base-cased, attention

  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_simplified \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token,cls,constant \
        +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred} \
        seed=1
    ```
  - wandb (weights & biases) run:
    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/fj59k57q
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/umu1dygb
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cmfti2es
  - artefacts
    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_19-52-49`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_02-51-40`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_11-26-30`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.57 |           87.958 |               56039 |   0.61 |    87.01 | attention |     token |
    | seed1 |     0.56 |           85.311 |               56039 |   0.60 |    81.57 | attention |       cls |
    | seed1 |     0.41 |           61.539 |               56039 |   0.41 |    61.81 | attention |  constant |

### Coreference Resolution: BERT, Re-TACRED (Re-TACRED frozen, attention aggregation)

-running a multi-model with Re-TACRED frozen model + bert-base-cased, attention

- command:
  ```bash
    python src/train.py \
      experiment=conll2012_coref_hoi_multimodel_simplified \
      trainer=gpu \
      +model.aggregate.type=attention \
      +model.aggregate.mode_query=token \
      +model.aggregate.mode_keys=token,cls,constant \
      +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred} \
      +model.freeze_models=[bert-base-cased-re-tacred] \
      seed=1
  ```
- wandb (weights & biases) run:
  - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/kspzlvxh
  - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/1t1bzvh4
  - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/t5gta8rm
- artefacts
  - model location:
    - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_19-53-49`
    - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_01-52-16`
    - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_07-54-29`
- metric values:
  |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
  | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
  | seed1 |     0.55 |           139.04 |               56039 |   0.58 |   130.97 | attention |     token |
  | seed1 |     0.54 |           134.97 |               56039 |   0.60 |   121.08 | attention |       cls |
  | seed1 |     0.46 |           183.37 |               56039 |   0.54 |   145.39 | attention |  constant |

### Coreference Resolution: BERT, NER (all trainable, attention aggregation)

- running a multi-model with NER pre-trained model + bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_simplified \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token,cls,constant \
        +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
        seed=1
    ```
  - wandb (weights & biases) run:
    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/libl1ly6
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8s9ednug
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zoseqhdt
  - artefacts
    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_19-44-37`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_11-42-49`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_11-43-18`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.58 |           122.32 |               56039 |   0.61 |   139.68 | attention |     token |
    | seed1 |     0.56 |           127.20 |               56039 |   0.57 |   135.40 | attention |       cls |
    | seed1 |     0.51 |           161.07 |               56039 |   0.56 |   148.09 | attention |  constant |

### Coreference Resolution: BERT, NER (NER frozen, attention aggregation)

- running a multi-model with NER frozen model + bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_simplified \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token,cls,constant \
        +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
        +model.freeze_models=[bert-base-cased-ner-ontonotes] \
        seed=1
    ```
  - wandb (weights & biases) run:
    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/gz5448hs
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/vt8c819z
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/npnkewuq
  - artefacts
    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_19-41-49`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_01-15-17`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_06-47-51`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.57 |           139.48 |               56039 |   0.60 |   135.96 | attention |     token |
    | seed1 |     0.55 |           142.61 |               56039 |   0.57 |   154.27 | attention |       cls |
    | seed1 |     0.50 |           198.56 |               56039 |   0.54 |   199.55 | attention |  constant |

### Coreference Resolution: BERT, SQUAD (all trainable, attention aggregation)

- running a multi-model with SQUAD pre-trained model + bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_simplified \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token,cls,constant \
        +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
        seed=1
    ```
  - wandb (weights & biases) run:
    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/tcvdqcai
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zjgtau3f
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/6tmrh85u
  - artefacts
    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_19-50-52`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_02-49-04`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_11-58-00`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.58 |           119.76 |               56039 |   0.63 |   123.72 | attention |     token |
    | seed1 |     0.58 |           121.30 |               56039 |   0.62 |   125.96 | attention |       cls |
    | seed1 |     0.56 |           140.19 |               56039 |   0.61 |   139.06 | attention |  constant |

### Coreference Resolution: BERT, SQUAD (SQUAD frozen, attention aggregation)

- running a multi-model with SQUAD frozen model + bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_simplified \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token,cls,constant \
        +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
        model.freeze_models=[bert-base-cased-qa-squad2] seed=1
    ```
  - wandb (weights & biases) run:
    - token: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ip3qos86
    - cls: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/1tbhl9iz
    - constant: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/td9o66j4
  - artefacts
    - model location:
      - token: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-30_19-54-19`
      - cls: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_01-37-04`
      - constant: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_07-21-01`
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.57 |           133.03 |               56039 |   0.56 |   170.24 | attention |     token |
    | seed1 |     0.57 |           134.90 |               56039 |   0.61 |   146.54 | attention |       cls |
    | seed1 |     0.53 |           176.92 |               56039 |   0.60 |   159.60 | attention |  constant |

## 2023-09-01

### Coreference Resolution: triple BERT (all trainable, attention aggregation)

- running a multi-model with 3 bert-base-cased models and attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_bert \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token \
        seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pyyusdyp
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_22-23-22
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.44 |            41.53 |               56039 |   0.42 |    54.74 | attention |     token |

### Coreference Resolution: triple coreference (all trainable, attention aggregation)

- running a multi-model with 3 pre-trained coreference models and bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_coref \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token \
        seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/7mg5ron8
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-31_22-23-22
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.63 |            51.63 |               56039 |   0.61 |    57.91 | attention |     token |

### Coreference Resolution: triple coreference (frozen pre-trained models, attention aggregation)

- running a multi-model with 3 pre-trained coreference models (frozen) and bert-base-cased, attention

  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_triple_coref \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token \
        +model.freeze_models=[bert-base-cased-coref1,bert-base-cased-coref2,bert-base-cased-coref3] \
        seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/mow07kik
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-01_01-40-01
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.61 |            58.79 |               56039 |   0.61 |    64.57 | attention |     token |

### Coreference Resolution: frozen pre-trained Coref, ReTACRED, NER and SQUAD models, attention aggregation)

- running a multi-model with 4 pre-trained models including coreference (frozen) w/o bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_pretrained_target \
        trainer=gpu \
        +model.aggregate.type=attention \
        +model.aggregate.mode_query=token \
        +model.aggregate.mode_keys=token \
        +model.freeze_models=[bert-base-cased-coref1,bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
        seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/xjxngvln
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-01_08-44-21
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate | mode_keys |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: | --------: |
    | seed1 |     0.50 |           103.13 |               56039 |   0.57 |    80.92 | attention |     token |

## 2023-09-28

### Coreference Resolution: target-only model

- running a target-only model (trained from scratch, using bert-base-cased)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_train_target \
        trainer=gpu \
        model.aggregate=mean
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3dal5i30
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_11-24-47
  - metric values:
    |       | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | ----: | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    | seed1 |    0.949 |            7.926 |               81257 |  0.736 |   139.71 |      mean |

### Coreference Resolution: frozen target (pre-trained) coreference model and bert-base-cased

- running a multi-model with a pre-trained coreference model and bert-base-cased, attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_with_bert \
        trainer=gpu
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ce4ye393
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-05_16-20-29
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.940 |            7.362 |               36425 |  0.736 |    98.87 | attention |

### Coreference Resolution: frozen target (pre-trained) coreference model and frozen NER

- running a multi-model with 2 pre-trained models including coreference (frozen) and NER (frozen), attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model \
        trainer=gpu \
        +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
        +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes] \
        +model.pretrained_configs.bert-base-cased-ner-ontonotes.name_or_path=bert-base-cased
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/jazixec9
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_16-40-54
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.953 |            5.513 |               78455 |  0.739 |   119.41 | attention |

### Coreference Resolution: frozen target (pre-trained) coreference model and frozen Re-TACRED

- running a multi-model with 2 pre-trained models including coreference (frozen) and Re-TACRED (frozen), attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model \
        trainer=gpu \
        +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
        +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-re-tacred] \
        +model.pretrained_configs.bert-base-cased-re-tacred.name_or_path=bert-base-cased \
        +model.pretrained_configs.bert-base-cased-re-tacred.vocab_size=29034
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/sa0bco64
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_16-50-40
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.954 |            5.311 |               81257 |  0.738 |   119.55 | attention |

### Coreference Resolution: frozen target (pre-trained) coreference model and frozen SQUAD

- running a multi-model with 2 pre-trained models including coreference (frozen) and SQUAD (frozen), attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model \
        trainer=gpu \
        +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
        +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-qa-squad2] \
        +model.pretrained_configs.bert-base-cased-qa-squad2.name_or_path=bert-base-cased
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3frydgsh
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_16-54-34
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.940 |            7.049 |               47633 |  0.735 |    93.24 | attention |

### Coreference Resolution: tuned target (pre-trained) coreference model and frozen NER

- running a multi-model with 2 pre-trained models including coreference (tuned) and NER (frozen), attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model \
        trainer=gpu \
        +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
        +model.freeze_models=[bert-base-cased-ner-ontonotes] \
        +model.pretrained_configs.bert-base-cased-ner-ontonotes.name_or_path=bert-base-cased
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dolw92mi
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_17-01-26
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.968 |            4.599 |               72851 |  0.738 |   174.12 | attention |

### Coreference Resolution: tuned target (pre-trained) coreference model and frozen Re-TACRED

- running a multi-model with 2 pre-trained models including coreference (tuned) and Re-TACRED (frozen), attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model \
        trainer=gpu \
        +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
        +model.freeze_models=[bert-base-cased-re-tacred] \
        +model.pretrained_configs.bert-base-cased-re-tacred.name_or_path=bert-base-cased \
        +model.pretrained_configs.bert-base-cased-re-tacred.vocab_size=29034
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/lbzjb6z6
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_18-17-58
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.974 |            3.761 |               92465 |  0.743 |   181.19 | attention |

### Coreference Resolution: tuned target (pre-trained) coreference model and frozen SQUAD

- running a multi-model with 2 pre-trained models including coreference (tuned) and SQUAD (frozen), attention
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model \
        trainer=gpu \
        +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
        +model.freeze_models=[bert-base-cased-qa-squad2] \
        +model.pretrained_configs.bert-base-cased-qa-squad2.name_or_path=bert-base-cased
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/5d1irxwl
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-27_18-56-33
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.972 |            3.924 |               81257 |  0.737 |   153.83 | attention |

### Deleted configuration files:

content of `conll2012_coref_hoi_multimodel_tuned_target_frozen_non-coref_model.yaml`:

```
defaults:
  - conll2012_coref_hoi_multimodel_base.yaml

model:
  pretrained_models:
    # copy from: /ds/text/cora4nlp/models/bert-base-cased-coref-hoi
    bert-base-cased-coref-hoi: "models/pretrained/bert-base-cased-coref-hoi"
  pretrained_configs:
    bert-base-cased-coref-hoi:
      name_or_path: "bert-base-cased"
  freeze_models: ???
```

content of `conll2012_coref_hoi_multimodel_frozen_target_frozen_non-coref_model.yaml`:

```
defaults:
  - conll2012_coref_hoi_multimodel_base.yaml

model:
  pretrained_models:
    # copy from: /ds/text/cora4nlp/models/bert-base-cased-coref-hoi
    bert-base-cased-coref-hoi: "models/pretrained/bert-base-cased-coref-hoi"
  pretrained_configs:
    bert-base-cased-coref-hoi:
      name_or_path: "bert-base-cased"
  freeze_models:
    - bert-base-cased-coref-hoi
```

## 2023-09-29

### Coreference Resolution: target-only model with attention

- running a target-only model (trained from scratch, using bert-base-cased and attention aggregation)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_train_target \
        trainer=gpu \
        model.aggregate=attention
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/miywcbq8
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-29_17-31-34
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.953 |            6.119 |               72851 |  0.736 |   114.95 | attention |

### Coreference Resolution: pre-trained frozen target-only model with attention

- running a pre-trained frozen target-only model (bert-base-cased-coref-hoi with attention aggregation)
  - command:
    ```bash
      python src/train.py \
        experiment=conll2012_coref_hoi_multimodel_frozen_target \
        trainer=gpu
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/7uyjay9c
  - artefacts
    - model location:
      /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-09-29_17-31-16
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.946 |            6.186 |               81257 |  0.735 |   119.19 | attention |

## 2023-10-04

### Relation Extraction - target-only model with attention - DEPRECATED

- running a target-only model (trained from scratch, using bert-base-cased and attention aggregation)
  - command:
    ```bash
      python src/train.py \
        experiment=tacred_multimodel \
        taskmodule.add_type_to_marker=false \
        trainer=gpu
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/8fdr64nu
  - artefacts
    - model location:
      /netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-09-08_14-08-59
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.991 |            0.010 |              106449 |  0.656 |    1.110 | attention |

### Relation Extraction - frozen pre-trained target-model + bert-base-cased with attention - DEPRECATED

- combining a frozen pretrained RE model with bert-base-cased
  - command:
    ```bash
      python src/train.py \
        experiment=tacred_multimodel_bert_frozen_re \
        trainer=gpu \
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/je97qssk
  - artefacts
    - model location:
      /netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-09-26_08-37-53
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.986 |            0.012 |              106449 |  0.709 |    0.955 | attention |

### Relation Extraction - frozen pre-trained target-model + frozen NER model with attention - DEPRECATED

- combining a frozen pretrained RE model with frozen pretrained NER model, using attention aggregation
  - command:
    ```bash
      python src/train.py \
        experiment=tacred_multimodel_frozen_re_ner \
        trainer=gpu \
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/wlcdze53
  - artefacts
    - model location:
      /netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-09-26_08-49-25
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.979 |            0.023 |              106449 |  0.719 |    0.820 | attention |

### Relation Extraction - pre-trained target-model + frozen NER model with attention - DEPRECATED

- combining a pretrained RE model with frozen pretrained NER model, using attention aggregation
  - command:
    ```bash
      python src/train.py \
        experiment=tacred_multimodel_re_frozen_ner \
        trainer=gpu \
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/467lkbxr
  - artefacts
    - model location:
      /netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-09-26_08-49-25
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.991 |            0.010 |              106449 |  0.632 |    1.031 | attention |

### Relation Extraction - pre-trained target-model + frozen NER model with attention, LR = 5e-6 - DEPRECATED

- same as above, but using a much smaller learning rate (combining a pretrained RE model with frozen pretrained NER model, using attention aggregation)
  - command:
    ```bash
      python src/train.py \
        experiment=tacred_multimodel_re_frozen_ner \
        trainer=gpu \
        model.learning_rate=5e-6 \
        "tags=[multimodel-re-frozen-ner,lr5e-6]"
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/by7864hv
  - artefacts
    - model location:
      /netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-09-27_10-48-15
  - metric values:
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.998 |            0.002 |              106449 |  0.681 |    1.224 | attention |

## 2023-10-05

### NER - frozen target model

- running a frozen NER model with mean aggregation
  - command:
    ```bash
       python src/train.py \
         experiment=conll2012_ner-multimodel_frozen_target \
         model.aggregate=mean \
         trainer=gpu \
         seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/dfki-nlp/conll2012-multi_model_token_classification-training/runs/1l35macp
  - artefacts
    - model location:
      /netscratch/harbecke/multi-task-knowledge-transfer/models/conll2012/multi_model_token_classification/2023-10-05_06-30-34
  - metric values (epoch 19):
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.925 |            0.077 |               72399 |  0.897 |    0.100 |      mean |

### NER - frozen finetuned NER target model + bert base cased

- running a frozen NER model with a tunable bert_base_cased and attention aggregation
  - command:
    ```bash
       python src/train.py \
       experiment=conll2012_ner-multimodel_frozen+bert \
       trainer=gpu \
       seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/dfki-nlp/conll2012-multi_model_token_classification-training/runs/12o8g7b3
  - artefacts
    - model location:
      /netscratch/harbecke/multi-task-knowledge-transfer/models/conll2012/multi_model_token_classification/2023-10-05_06-57-29
  - metric values (epoch 4):
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.948 |            0.039 |               72399 |  0.911 |    0.094 | attention |

### NER - frozen finetuned NER target model + frozen coreference model

- running a frozen NER model with a frozen coref model and attention aggregation
  - command:
    ```bash
       python src/train.py \
       experiment=conll2012_ner-multimodel_frozen+frozen_coref \
       trainer=gpu \
       seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/dfki-nlp/conll2012-multi_model_token_classification-training/runs/04jcjyls
  - artefacts
    - model location:
      /netscratch/harbecke/multi-task-knowledge-transfer/models/conll2012/multi_model_token_classification/2023-10-05_06-59-11
  - metric values (epoch 11):
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.930 |            0.070 |               72399 |  0.902 |    0.096 | attention |

### NER - tunable finetuned NER target model + frozen coreference model

- running a tunable NER model with a frozen coref model and attention aggregation
  - command:
    ```bash
       python src/train.py \
       experiment=conll2012_ner-multimodel_tuned+frozen_coref \
       trainer=gpu \
       seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/dfki-nlp/conll2012-multi_model_token_classification-training/runs/dy1928uo
  - artefacts
    - model location:
      /netscratch/harbecke/multi-task-knowledge-transfer/models/conll2012/multi_model_token_classification/2023-10-05_06-59-40
  - metric values (epoch 1):
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.950 |            0.047 |               72399 |  0.911 |    0.091 | attention |

### NER - training bert base cased in multimodel with attention

- training a bert base cased model with an attention head from the multimodel
  - command:
    ```bash
       python src/train.py \
       experiment=conll2012_ner-multimodel_train_target \
       trainer=gpu \
       seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/dfki-nlp/conll2012-multi_model_token_classification-training/runs/mdwun8lz
  - artefacts
    - model location:
      /netscratch/harbecke/multi-task-knowledge-transfer/models/conll2012/multi_model_token_classification/2023-10-05_15-46-51
  - metric values (epoch 7):
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.996 |            0.003 |               72399 |  0.914 |    0.118 | attention |

### Relation Extraction - tune bert-base-cased in multimodel

- tune bert-base-cased in multimodel (with attention)
  - command:
    ```bash
    python src/train.py \
    trainer=gpu \
    experiment=tacred_multimodel \
    tags=[multimodel_bert]
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/5oo1xfcc
  - artefacts
    - model location:
      `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-05_09-30-38`
  - metric values (epoch 3)
    | val/f1 | val/loss |
    | -----: | -------: |
    |  0.764 |   0.4647 |

### Relation Extraction - frozen pre-trained target-model with tuned bert-base-cased

- frozen pre-trained target-model with tuned bert-base-cased
  - original command (from W&B run):
    ```bash
    python src/train.py \
    trainer=gpu \
    experiment=tacred_multimodel_bert_frozen_re
    ```
  - correct command (to reproduce with current configs):
    ```bash
    python src/train.py \
    trainer=gpu \
    experiment=tacred_multimodel_frozen_re_tuned_bert
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/40dmu49m
  - artefacts
    - model location:
      `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-05_14-05-58`
  - metric values (epoch 8)
    | val/f1 | val/loss |
    | -----: | -------: |
    | 0.7305 |   0.7121 |

## 2023-10-06

### Coreference Resolution

- wandb report with the val/f1 and val/loss graphs (experiments from 2023-09-28 and 2023-09-29):
  [https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/reports/Coreference-Experiments--Vmlldzo1NjAwNTMy](https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/reports/Coreference-Experiments--Vmlldzo1NjAwNTMy)

## 2023-10-10

### Relation Extraction - tuned bert

- command:
  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=tacred_multimodel \
  validate=True \
  +logger.wandb.name=multimodel_bert-FINAL
  ```
- wandb run: https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/n17ion9u
- artefacts
  - model location:
    `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-10_12-29-24`
- metric values (epoch_003.ckpt):
  ```
  307 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  308 ┃      Validate metric      ┃       DataLoader 0        ┃
  309 ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  310 │          val/f1           │    0.7639808654785156     │
  311 │         val/loss          │    0.46465492248535156    │
  312 └───────────────────────────┴───────────────────────────┘
  ```

### Relation Extraction - frozen pre-trained RE model + frozen pre-trained NER model

- command:
  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=tacred_multimodel_frozen_re_ner \
  validate=True \
  +logger.wandb.name=multimodel_frozen_re_ner-FINAL
  ```
- wandb run: https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/90amy45n
- artefacts
  - model location:
    `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-10_18-08-49`
- metric values (epoch_012.ckpt):
  ```
  521 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  522 ┃      Validate metric      ┃       DataLoader 0        ┃
  523 ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  524 │          val/f1           │    0.7301324605941772     │
  525 │         val/loss          │    0.7457929253578186     │
  526 └───────────────────────────┴───────────────────────────┘
  ```

### Relation Extraction - frozen pre-trained RE model + tuned bert

- command:
  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=tacred_multimodel_frozen_re_tuned_bert \
  validate=True \
  +logger.wandb.name=multimodel_frozen_re_tuned_bert-FINAL
  ```
- wandb run: https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/bbjpni8a
- artefacts
  - model location:
    `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-10_22-31-41`
- metric values (epoch_008.ckpt):
  ```
  524 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  525 ┃      Validate metric      ┃       DataLoader 0        ┃
  526 ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  527 │          val/f1           │    0.7330757975578308     │
  528 │         val/loss          │     0.779472827911377     │
  529 └───────────────────────────┴───────────────────────────┘
  ```

### Relation Extraction - frozen pre-trained RE model

- command:
  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=tacred_multimodel_frozen_re \
  validate=True \
  +logger.wandb.name=multimodel_frozen_re-FINAL
  ```
- wandb run: https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/u503tnqy
- artefacts
  - model location:
    `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-11_06-11-56`
- metric values (epoch_030.ckpt):
  ```
  303 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  304 ┃      Validate metric      ┃       DataLoader 0        ┃
  305 ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  306 │          val/f1           │    0.7273730635643005     │
  307 │         val/loss          │    0.7825005054473877     │
  308 └───────────────────────────┴───────────────────────────┘
  ```

### Relation Extraction - tuned pre-trained RE model + frozen pre-trained NER model

- command:
  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=tacred_multimodel_re_frozen_ner \
  validate=True \
  +logger.wandb.name=multimodel_re_frozen_ner-FINAL
  ```
- wandb run: https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training/runs/8rh5e4wr
- artefacts
  - model location:
    `/netscratch/hennig/code/multi-task-knowledge-transfer/models/tacred/multi_model_re_text_classification/2023-10-11_14-23-30`
- metric values (epoch_003.ckpt):
  ```
  521 ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  522 ┃      Validate metric      ┃       DataLoader 0        ┃
  523 ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  524 │          val/f1           │    0.7617733478546143     │
  525 │         val/loss          │    0.7384195327758789     │
  526 └───────────────────────────┴───────────────────────────┘
  ```

## 2023-10-17

### Coreference resolution - frozen pre-trained target-model + frozen bert (learning rate optimization)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_frozen_target_with_frozen_bert \
  trainer=gpu \
  model.task_learning_rate=2e-3,1e-3,2e-4,1e-4,2e-5 \
  seed=1,2,3,4,5 \
  --multirun
  ```

- wandb runs for learning_rate=2e-3:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0qj9dypi
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/wt456mvi
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/asa1b45g
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/4kwweri4
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ros5hm3c

- wandb runs for learning_rate=1e-3:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hopchzdz
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ytjw33ox
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/f3wwxqi0
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0pscdw99
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/proghe84

- wandb runs for learning_rate=2e-4:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/di1ocoeb
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cbrgafhc
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/1dttt4mx
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hnrqepd6
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/fbqikzff

- wandb runs for learning_rate=1e-4:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/wuvc53pf
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/vi0yqoyz
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/p8epwiv8
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ah6k4mj4
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/y9kqvz90

- wandb runs for learning_rate=2e-5:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/o8glhamx
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qcbthya0
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/euk6ysrd
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/waymuds1
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3q0ho8l9

- artefacts

  - model location:
    - learning_rate=2e-3:
      - seed1: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-13_11-23-39`
      - seed2: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-13_13-28-47`
      - seed3: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-13_16-02-25`
      - seed4: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-13_18-45-14`
      - seed5: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-13_21-20-12`
    - learning_rate=1e-3:
      - seed1: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-13_23-17-08`
      - seed2: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_02-40-09`
      - seed3: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_04-47-19`
      - seed4: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_06-54-11`
      - seed5: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_09-41-38`
    - learning_rate=2e-4:
      - seed1: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_12-39-28`
      - seed2: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_15-31-46`
      - seed3: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_17-49-12`
      - seed4: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_19-50-07`
      - seed5: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-14_21-31-46`
    - learning_rate=1e-4:
      - seed1: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_00-49-37`
      - seed2: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_02-21-49`
      - seed3: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_04-56-27`
      - seed4: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_06-23-44`
      - seed5: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_08-05-27`
    - learning_rate=2e-5:
      - seed1: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_10-59-18`
      - seed2: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_13-43-54`
      - seed3: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_16-42-53`
      - seed4: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_19-51-49`
      - seed5: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_21-57-53`

- metric values (average over 5 seeds):

  - command for evaluation (shortened, see below):

    ```bash
    python src/evaluate.py \
    dataset=conll2012_ontonotesv5_preprocessed \
    model_name_or_path=/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_00-49-37,/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_02-21-49,/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-15_04-56-27 \
    datamodule.batch_size=1 \
    +datamodule.test_split=validation \
    trainer=gpu \
    --multirun
    ```

  The evaluation was done separately using the best model checkpoint for each setting (in the example above only three models are listed for readability). The best checkpoints correspond to the ones specified under `artefacts - model location`.

  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃      Validate metric      ┃    Learning Rate 2e-3     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │          val/f1           │    0.72661                │
  │         val/loss          │    389.376344             │
  │         best epoch        │    28.4                   │
  └───────────────────────────┴───────────────────────────┘
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃      Validate metric      ┃    Learning Rate 1e-3     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │          val/f1           │    0.730305999            │
  │         val/loss          │    233.429692             │
  │         best epoch        │    30.6                   │
  └───────────────────────────┴───────────────────────────┘
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃      Validate metric      ┃    Learning Rate 2e-4     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │          val/f1           │    0.735008               │
  │         val/loss          │    119.97546              │
  │         best epoch        │    28.6                   │
  └───────────────────────────┴───────────────────────────┘
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃      Validate metric      ┃    Learning Rate 1e-4     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │          val/f1           │    0.736529999999         │
  │         val/loss          │    99.116872              │
  │         best epoch        │    24.2                   │
  └───────────────────────────┴───────────────────────────┘
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃      Validate metric      ┃    Learning Rate 2e-5     ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │          val/f1           │    0.731014               │
  │         val/loss          │    85.900434              │
  │         best epoch        │    32.6                   │
  └───────────────────────────┴───────────────────────────┘
  ```

## 2023-10-23

### Coreference resolution - frozen pre-trained target-model + frozen NER model (learning rate 1e-4)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/y5n287a5
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/sgcjpctm
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/la3n3hhj
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/e4y3c6ra
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/tiksiabq

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss',) | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                          | ('val/f1',) |
  | --: | -------------------: | --------------: | ------------: | ------------: | --------------------: | :----------------------------------------------------------------------------------------------------------- | ----------: |
  |   0 |              1.02181 |         6.29968 |      0.944393 |        89.273 |               6.29968 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_13-35-43 |    0.737421 |
  |   1 |                    0 |         5.29238 |      0.953497 |       104.857 |               5.29238 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_15-26-42 |    0.738567 |
  |   2 |          0.000145435 |         4.58908 |      0.959066 |       112.859 |               4.58908 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_18-02-41 |    0.740512 |
  |   3 |          4.33914e-05 |         5.62474 |      0.949977 |       98.0267 |               5.62474 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_21-13-38 |    0.736869 |
  |   4 |            0.0121821 |          5.3289 |      0.952816 |        104.06 |                5.3289 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_23-25-53 |    0.738005 |

- aggregated values:

  |                  |         25% |         50% |       75% | count |      max |     mean |      min |        std |
  | :--------------- | ----------: | ----------: | --------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |    0.949977 |    0.952816 |  0.953497 |     5 | 0.959066 |  0.95195 | 0.944393 | 0.00535679 |
  | train/loss       |     5.29238 |      5.3289 |   5.62474 |     5 |  6.29968 |  5.42695 |  4.58908 |   0.618571 |
  | train/loss_epoch |     5.29238 |      5.3289 |   5.62474 |     5 |  6.29968 |  5.42695 |  4.58908 |   0.618571 |
  | train/loss_step  | 4.33914e-05 | 0.000145435 | 0.0121821 |     5 |  1.02181 | 0.206836 |        0 |   0.455614 |
  | val/f1           |    0.737421 |    0.738005 |  0.738567 |     5 | 0.740512 | 0.738275 | 0.736869 | 0.00140236 |
  | val/loss         |     98.0267 |      104.06 |   104.857 |     5 |  112.859 |  101.815 |   89.273 |    8.77376 |

### Coreference resolution - frozen pre-trained target-model + frozen RE model (learning rate 1e-4)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-re-tacred] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qfy6d1b0
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/mssfnz3z
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/f50ybjt2
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qe6wkhfr
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/u8cq3f8r

- metric values per seed

  |     | ('train/loss',) | ('model_save_dir',)                                                                                          | ('val/f1',) | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('train/loss_step',) |
  | --: | --------------: | :----------------------------------------------------------------------------------------------------------- | ----------: | ------------: | ------------: | --------------------: | -------------------: |
  |   0 |           5.683 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_13-31-35 |    0.737545 |      0.949149 |       95.7091 |                 5.683 |           7.9748e-05 |
  |   1 |         6.13037 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_15-46-44 |    0.737611 |      0.947831 |       94.5831 |               6.13037 |             0.131756 |
  |   2 |         6.78004 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_17-46-58 |    0.733536 |      0.939424 |       81.6808 |               6.78004 |          0.000452199 |
  |   3 |         5.60846 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_19-17-16 |    0.738129 |      0.950558 |       95.9354 |               5.60846 |              6.01864 |
  |   4 |         6.79369 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_21-28-54 |    0.736191 |      0.941801 |       84.1064 |               6.79369 |              1.76297 |

- aggregated values:

  |                  |         25% |      50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | ----------: | -------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         |    0.941801 | 0.947831 | 0.949149 |     5 | 0.950558 | 0.945752 |   0.939424 | 0.00486375 |
  | train/loss       |       5.683 |  6.13037 |  6.78004 |     5 |  6.79369 |  6.19911 |    5.60846 |   0.572485 |
  | train/loss_epoch |       5.683 |  6.13037 |  6.78004 |     5 |  6.79369 |  6.19911 |    5.60846 |   0.572485 |
  | train/loss_step  | 0.000452199 | 0.131756 |  1.76297 |     5 |  6.01864 |  1.58278 | 7.9748e-05 |    2.58957 |
  | val/f1           |    0.736191 | 0.737545 | 0.737611 |     5 | 0.738129 | 0.736603 |   0.733536 | 0.00185788 |
  | val/loss         |     84.1064 |  94.5831 |  95.7091 |     5 |  95.9354 |   90.403 |    81.6808 |    6.92745 |

### Coreference resolution - frozen pre-trained target-model + frozen SQUAD model (learning rate 1e-4)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zajxvjv4
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/6rrzy3y9
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hu3xo8e9
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/wcwdgwbx
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/geac74ge

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                          | ('train/loss_step',) | ('train/loss',) | ('train/f1',) | ('train/loss_epoch',) |
  | --: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------- | -------------------: | --------------: | ------------: | --------------------: |
  |   0 |    0.738848 |       102.833 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_13-36-00 |              15.2984 |         5.10011 |      0.955167 |               5.10011 |
  |   1 |    0.740119 |       104.696 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_16-11-15 |                    0 |         4.88383 |      0.956375 |               4.88383 |
  |   2 |    0.736739 |       93.0174 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_18-57-25 |            0.0142197 |         6.18833 |      0.947031 |               6.18833 |
  |   3 |    0.738574 |       97.6904 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_20-52-16 |              14.1326 |         5.68418 |      0.951027 |               5.68418 |
  |   4 |    0.738444 |       104.822 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_23-02-37 |           0.00821877 |         5.26755 |      0.954221 |               5.26755 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |   0.951027 |  0.954221 | 0.955167 |     5 | 0.956375 | 0.952764 | 0.947031 | 0.00376912 |
  | train/loss       |    5.10011 |   5.26755 |  5.68418 |     5 |  6.18833 |   5.4248 |  4.88383 |   0.517943 |
  | train/loss_epoch |    5.10011 |   5.26755 |  5.68418 |     5 |  6.18833 |   5.4248 |  4.88383 |   0.517943 |
  | train/loss_step  | 0.00821877 | 0.0142197 |  14.1326 |     5 |  15.2984 |   5.8907 |        0 |    8.06648 |
  | val/f1           |   0.738444 |  0.738574 | 0.738848 |     5 | 0.740119 | 0.738545 | 0.736739 | 0.00120875 |
  | val/loss         |    97.6904 |   102.833 |  104.696 |     5 |  104.822 |  100.612 |  93.0174 |    5.13686 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (learning rate 1e-4)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dz0sxnxa
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cvpf3jp5
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/1rppg5xw
  - seed4: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/epykn14j
  - seed5: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/lua1uxum

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('val/loss',) | ('train/loss',) | ('model_save_dir',)                                                                                          |
  | --: | ----------: | ------------: | --------------------: | -------------------: | ------------: | --------------: | :----------------------------------------------------------------------------------------------------------- |
  |   0 |    0.739796 |      0.957975 |               4.52363 |           3.8147e-05 |       106.637 |         4.52363 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_13-43-05 |
  |   1 |    0.737569 |      0.950618 |               5.47105 |                    0 |       95.4654 |         5.47105 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_16-30-01 |
  |   2 |    0.737099 |      0.950984 |               5.58014 |          7.74858e-06 |        96.361 |         5.58014 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_18-38-06 |
  |   3 |    0.738257 |      0.949871 |               5.67083 |          1.90735e-06 |        94.908 |         5.67083 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_20-41-04 |
  |   4 |    0.738822 |      0.957166 |                4.8782 |            0.0123741 |       103.938 |          4.8782 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-20_22-39-02 |

- aggregated values:

  |                  |         25% |         50% |        75% | count |       max |       mean |      min |        std |
  | :--------------- | ----------: | ----------: | ---------: | ----: | --------: | ---------: | -------: | ---------: |
  | train/f1         |    0.950618 |    0.950984 |   0.957166 |     5 |  0.957975 |   0.953323 | 0.949871 | 0.00390864 |
  | train/loss       |      4.8782 |     5.47105 |    5.58014 |     5 |   5.67083 |    5.22477 |  4.52363 |   0.499404 |
  | train/loss_epoch |      4.8782 |     5.47105 |    5.58014 |     5 |   5.67083 |    5.22477 |  4.52363 |   0.499404 |
  | train/loss_step  | 1.90735e-06 | 7.74858e-06 | 3.8147e-05 |     5 | 0.0123741 | 0.00248438 |        0 | 0.00552853 |
  | val/f1           |    0.737569 |    0.738257 |   0.738822 |     5 |  0.739796 |   0.738308 | 0.737099 | 0.00105923 |
  | val/loss         |     95.4654 |      96.361 |    103.938 |     5 |   106.637 |    99.4617 |   94.908 |    5.42757 |

## 2023-10-25

### LR Tuning for RE (frozen pretrained RE + frozen bert)

- command:

  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=tacred_multimodel_frozen_re_bert \
  seed=1031,1097,1153,1223,1579 \
  model.learning_rate=1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
  validate=True
  ```

- wandb runs

  - Runs with numbers -96 to -139 and -147 at https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training

metric values (averaged over 5 seeds):

| learning rate | mean val/f1 | sd val/f1 | mean val/loss |
| ------------: | ----------: | --------: | ------------: |
|         1e-06 |      0.7243 |    0.0011 |        0.7284 |
|         3e-06 |       0.727 |    0.0012 |        0.7415 |
|         1e-05 |      0.7301 |    0.0021 |         0.754 |
|         3e-05 |       0.732 |     0.003 |         0.798 |
|        0.0001 |      0.7347 |    0.0026 |         0.854 |
|        0.0003 |      0.7361 |    0.0079 |        1.0396 |
|         0.001 |      0.7376 |    0.0085 |        1.7975 |
|         0.003 |      0.7327 |    0.0026 |        2.9013 |
|          0.01 |      0.7283 |     0.004 |         3.387 |

### RE - frozen pre-trained target-model + frozen NER model (learning rate 1e-3)

- command:

  ```bash
  python src/train.py \
  experiment=tacred_multimodel_base \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.pretrained_models.bert-base-cased-ner-ontonotes=/ds/text/cora4nlp/models/bert-base-cased-ner-ontonotes \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-3 \
  trainer=gpu \
  seed=1031,1097,1153,1223,1579 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  -m
  ```

- wandb runs

  - Runs with numbers -140 to -163 (except -147) at https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training

- results:

|                  |        25% |       50% |       75% | count |       max |      mean |        min |        std |
| :--------------- | ---------: | --------: | --------: | ----: | --------: | --------: | ---------: | ---------: |
| train/f1         |   0.972717 |  0.973486 |  0.974331 |     5 |  0.974639 |  0.973394 |   0.971795 |  0.0011673 |
| train/loss       |   0.028742 | 0.0295178 | 0.0314421 |     5 | 0.0335904 | 0.0302926 |  0.0281709 | 0.00221942 |
| train/loss_epoch |   0.028742 | 0.0295178 | 0.0314421 |     5 | 0.0335904 | 0.0302926 |  0.0281709 | 0.00221942 |
| train/loss_step  | 0.00602266 | 0.0141531 | 0.0262175 |     5 |  0.188285 | 0.0472922 | 0.00178264 |  0.0793658 |
| val/f1           |   0.731052 |  0.733444 |  0.736203 |     5 |  0.751472 |  0.736534 |     0.7305 | 0.00864941 |
| val/loss         |     1.7009 |   1.78245 |   1.88908 |     5 |   1.89592 |   1.77755 |    1.61942 |   0.119746 |

### RE - frozen pre-trained target-model + frozen QA model (learning rate 1e-3)

- command:

  ```bash
  python src/train.py \
  experiment=tacred_multimodel_base \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.pretrained_models.bert-base-cased-qa-squad2=/ds/text/cora4nlp/models/bert-base-cased-qa-squad2 \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-3 \
  trainer=gpu \
  seed=1031,1097,1153,1223,1579 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  -m
  ```

- wandb runs

  - Runs with numbers -140 to -163 (except -147) at https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training

- results:

|                  |        25% |        50% |       75% | count |       max |      mean |        min |         std |
| :--------------- | ---------: | ---------: | --------: | ----: | --------: | --------: | ---------: | ----------: |
| train/f1         |   0.972794 |    0.97364 |  0.974331 |     5 |  0.974485 |  0.973547 |   0.972487 | 0.000893608 |
| train/loss       |  0.0290063 |  0.0291671 | 0.0292249 |     5 | 0.0311898 | 0.0294617 |  0.0287205 |  0.00098564 |
| train/loss_epoch |  0.0290063 |  0.0291671 | 0.0292249 |     5 | 0.0311898 | 0.0294617 |  0.0287205 |  0.00098564 |
| train/loss_step  | 0.00649406 | 0.00959971 | 0.0239457 |     5 |  0.173741 | 0.0430638 | 0.00153869 |   0.0735246 |
| val/f1           |   0.733996 |   0.735835 |  0.738043 |     5 |  0.755151 |  0.738742 |   0.730684 |  0.00956091 |
| val/loss         |    1.71358 |    1.77702 |    1.8848 |     5 |   1.91307 |    1.7747 |    1.58506 |    0.133202 |

### RE - frozen pre-trained target-model + frozen Coref model (learning rate 1e-3)

- command:

  ```bash
  python src/train.py \
  experiment=tacred_multimodel_base \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.pretrained_models.bert-base-cased-coref-hoi=/ds/text/cora4nlp/models/bert-base-cased-coref-hoi \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-coref-hoi] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-3 \
  trainer=gpu \
  seed=1031,1097,1153,1223,1579 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  -m
  ```

- wandb runs

  - Runs with numbers -140 to -163 (except -147) at https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training

- results:

|                  |        25% |       50% |       75% | count |       max |      mean |        min |         std |
| :--------------- | ---------: | --------: | --------: | ----: | --------: | --------: | ---------: | ----------: |
| train/f1         |   0.973102 |   0.97387 |  0.974716 |     5 |  0.974869 |  0.973732 |   0.972103 |  0.00115457 |
| train/loss       |  0.0288969 | 0.0291348 | 0.0296901 |     5 | 0.0311432 | 0.0294986 |  0.0286281 | 0.000999218 |
| train/loss_epoch |  0.0288969 | 0.0291348 | 0.0296901 |     5 | 0.0311432 | 0.0294986 |  0.0286281 | 0.000999218 |
| train/loss_step  | 0.00735919 | 0.0102697 | 0.0263926 |     5 |  0.150773 | 0.0392107 | 0.00125884 |   0.0630534 |
| val/f1           |     0.7305 |   0.73234 |  0.738043 |     5 |   0.75092 |  0.736387 |   0.730132 |  0.00871915 |
| val/loss         |    1.74941 |   1.76399 |   1.85395 |     5 |   1.92623 |   1.71937 |    1.30326 |    0.243382 |

### RE - frozen pre-trained target-model + frozen MRPC model (learning rate 1e-3)

- command:

  ```bash
  python src/train.py \
  experiment=tacred_multimodel_base \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.pretrained_models.bert-base-cased-mrpc=bert-base-cased-finetuned-mrpc \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-3 \
  trainer=gpu \
  seed=1031,1097,1153,1223,1579 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  -m
  ```

- wandb runs

  - Runs with numbers -140 to -163 (except -147) at https://wandb.ai/leonhardhennig/tacred-multi_model_re_text_classification-training

- results:

  |                  |       25% |        50% |       75% | count |      max |     mean |        min |         std |
  | :--------------- | --------: | ---------: | --------: | ----: | -------: | -------: | ---------: | ----------: |
  | train/f1         |  0.973179 |   0.974101 |  0.974331 |     5 | 0.974716 | 0.973578 |   0.971565 |  0.00125976 |
  | train/loss       | 0.0290088 |   0.029706 |  0.029738 |     5 | 0.030946 | 0.029572 |  0.0284613 | 0.000933314 |
  | train/loss_epoch | 0.0290088 |   0.029706 |  0.029738 |     5 | 0.030946 | 0.029572 |  0.0284613 | 0.000933314 |
  | train/loss_step  | 0.0054442 | 0.00997965 | 0.0239383 |     5 | 0.184513 | 0.045318 | 0.00271478 |   0.0782401 |
  | val/f1           |  0.729765 |   0.730316 |  0.737859 |     5 | 0.746321 | 0.734695 |   0.729213 |  0.00739438 |
  | val/loss         |   1.61852 |     1.7736 |    1.8762 |     5 |  1.89542 |  1.71494 |    1.41099 |      0.2023 |

## 2023-11-01

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC truncated to 2 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=2 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/im9sk2pj
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/vuw0r6gw
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/69su8xd5

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('val/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                          | ('val/loss',) | ('train/loss',) |
  | --: | -------------------: | ------------: | ----------: | --------------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | --------------: |
  |   0 |                    0 |      0.956503 |    0.739843 |               4.60134 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_09-58-17 |       105.824 |         4.60134 |
  |   1 |              7.87041 |      0.950582 |    0.738118 |               5.43119 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_12-05-31 |       96.6548 |         5.43119 |
  |   2 |              15.3614 |      0.960068 |    0.738703 |               4.11392 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_13-44-23 |       116.085 |         4.11392 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |         std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ----------: |
  | train/f1         | 0.953543 | 0.956503 | 0.958285 |     3 | 0.960068 | 0.955718 | 0.950582 |  0.00479108 |
  | train/loss       |  4.35763 |  4.60134 |  5.01627 |     3 |  5.43119 |  4.71549 |  4.11392 |    0.666012 |
  | train/loss_epoch |  4.35763 |  4.60134 |  5.01627 |     3 |  5.43119 |  4.71549 |  4.11392 |    0.666012 |
  | train/loss_step  |   3.9352 |  7.87041 |  11.6159 |     3 |  15.3614 |  7.74393 |        0 |     7.68147 |
  | val/f1           |  0.73841 | 0.738703 | 0.739273 |     3 | 0.739843 | 0.738888 | 0.738118 | 0.000877578 |
  | val/loss         |  101.239 |  105.824 |  110.955 |     3 |  116.085 |  106.188 |  96.6548 |     9.72023 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC truncated to 4 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=4 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/uq7ity7l
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ximalunv
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/gyyzhviw

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                                          | ('val/loss',) | ('train/loss_step',) | ('val/f1',) | ('train/loss',) | ('train/loss_epoch',) |
  | --: | ------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | ----------: | --------------: | --------------------: |
  |   0 |       0.94885 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_04-54-58 |       96.4069 |           0.00508129 |    0.738447 |          5.5082 |                5.5082 |
  |   1 |      0.950388 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_06-34-57 |       95.0506 |               42.294 |    0.739138 |         5.66095 |               5.66095 |
  |   2 |      0.953539 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_08-12-10 |       98.2425 |              20.1861 |    0.738532 |         5.12105 |               5.12105 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |        min |         std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ---------: | ----------: |
  | train/f1         | 0.949619 | 0.950388 | 0.951963 |     3 | 0.953539 | 0.950926 |    0.94885 |  0.00239017 |
  | train/loss       |  5.31463 |   5.5082 |  5.58458 |     3 |  5.66095 |  5.43007 |    5.12105 |    0.278302 |
  | train/loss_epoch |  5.31463 |   5.5082 |  5.58458 |     3 |  5.66095 |  5.43007 |    5.12105 |    0.278302 |
  | train/loss_step  |  10.0956 |  20.1861 |    31.24 |     3 |   42.294 |  20.8284 | 0.00508129 |     21.1518 |
  | val/f1           |  0.73849 | 0.738532 | 0.738835 |     3 | 0.739138 | 0.738706 |   0.738447 | 0.000376939 |
  | val/loss         |  95.7288 |  96.4069 |  97.3247 |     3 |  98.2425 |  96.5667 |    95.0506 |     1.60191 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC truncated to 6 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=6 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qiks7v6u
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/tes33qrr
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/2il5e3nr

- metric values per seed

  |     | ('model_save_dir',)                                                                                          | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('train/loss_step',) | ('val/f1',) | ('train/loss',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | ------------: | ------------: | --------------------: | -------------------: | ----------: | --------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-31_22-41-23 |      0.939869 |       83.3963 |               6.83493 |              0.21568 |    0.732919 |         6.83493 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-31_23-55-35 |      0.964715 |       117.143 |               3.63914 |              11.4469 |    0.741913 |         3.63914 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_02-53-58 |      0.955971 |       102.507 |               4.82261 |          0.000116348 |    0.739309 |         4.82261 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |  0.94792 | 0.955971 | 0.960343 |     3 | 0.964715 | 0.953518 |    0.939869 |  0.0126035 |
  | train/loss       |  4.23087 |  4.82261 |  5.82877 |     3 |  6.83493 |  5.09889 |     3.63914 |    1.61571 |
  | train/loss_epoch |  4.23087 |  4.82261 |  5.82877 |     3 |  6.83493 |  5.09889 |     3.63914 |    1.61571 |
  | train/loss_step  | 0.107898 |  0.21568 |  5.83128 |     3 |  11.4469 |  3.88756 | 0.000116348 |    6.54745 |
  | val/f1           | 0.736114 | 0.739309 | 0.740611 |     3 | 0.741913 | 0.738047 |    0.732919 | 0.00462785 |
  | val/loss         |  92.9519 |  102.507 |  109.825 |     3 |  117.143 |  101.016 |     83.3963 |    16.9229 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC truncated to 8 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=8 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/080bc6vr
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rumna1bu
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/uznvourt

- metric values per seed

  |     | ('train/f1',) | ('train/loss_step',) | ('val/loss',) | ('train/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                          | ('train/loss_epoch',) |
  | --: | ------------: | -------------------: | ------------: | --------------: | ----------: | :----------------------------------------------------------------------------------------------------------- | --------------------: |
  |   0 |      0.949598 |              46.1368 |        93.143 |         5.67624 |    0.736345 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_11-59-57 |               5.67624 |
  |   1 |      0.943453 |                    0 |       84.8816 |         6.20013 |    0.735298 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_13-41-18 |               6.20013 |
  |   2 |      0.959979 |          0.000193834 |       109.935 |         4.53502 |    0.739708 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_15-07-16 |               4.53502 |

- aggregated values:

  |                  |         25% |         50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ----------: | ----------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |    0.946526 |    0.949598 | 0.954788 |     3 | 0.959979 |  0.95101 | 0.943453 | 0.00835263 |
  | train/loss       |     5.10563 |     5.67624 |  5.93818 |     3 |  6.20013 |  5.47046 |  4.53502 |   0.851412 |
  | train/loss_epoch |     5.10563 |     5.67624 |  5.93818 |     3 |  6.20013 |  5.47046 |  4.53502 |   0.851412 |
  | train/loss_step  | 9.69171e-05 | 0.000193834 |  23.0685 |     3 |  46.1368 |   15.379 |        0 |     26.637 |
  | val/f1           |    0.735821 |    0.736345 | 0.738026 |     3 | 0.739708 | 0.737117 | 0.735298 | 0.00230396 |
  | val/loss         |     89.0123 |      93.143 |  101.539 |     3 |  109.935 |  95.9866 |  84.8816 |    12.7666 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC truncated to 10 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rbszom54
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/nt2at6ey
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ah7nvay9

- metric values per seed

  |     | ('model_save_dir',)                                                                                          | ('val/f1',) | ('train/loss_step',) | ('train/f1',) | ('val/loss',) | ('train/loss',) | ('train/loss_epoch',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | ----------: | -------------------: | ------------: | ------------: | --------------: | --------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_05-51-53 |    0.736748 |             0.107841 |      0.951916 |       98.3079 |         5.42417 |               5.42417 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_07-45-07 |    0.740065 |          2.90871e-05 |      0.958756 |       109.197 |         4.58526 |               4.58526 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_10-13-02 |       0.736 |              23.6542 |      0.950343 |       95.4887 |         5.64798 |               5.64798 |

- aggregated values:

  |                  |       25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | --------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |  0.951129 | 0.951916 | 0.955336 |     3 | 0.958756 | 0.953672 |    0.950343 | 0.00447298 |
  | train/loss       |   5.00471 |  5.42417 |  5.53607 |     3 |  5.64798 |  5.21913 |     4.58526 |   0.560244 |
  | train/loss_epoch |   5.00471 |  5.42417 |  5.53607 |     3 |  5.64798 |  5.21913 |     4.58526 |   0.560244 |
  | train/loss_step  | 0.0539351 | 0.107841 |   11.881 |     3 |  23.6542 |  7.92071 | 2.90871e-05 |    13.6258 |
  | val/f1           |  0.736374 | 0.736748 | 0.738407 |     3 | 0.740065 | 0.737605 |       0.736 |  0.0021635 |
  | val/loss         |   96.8983 |  98.3079 |  103.753 |     3 |  109.197 |  100.998 |     95.4887 |     7.2394 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC truncated to 11 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=11 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/080bc6vr
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rumna1bu
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/uznvourt

- metric values per seed

  |     | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                          | ('val/f1',) | ('train/loss',) | ('val/loss',) | ('train/loss_step',) |
  | --: | ------------: | --------------------: | :----------------------------------------------------------------------------------------------------------- | ----------: | --------------: | ------------: | -------------------: |
  |   0 |      0.960751 |               4.44881 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-31_22-43-36 |    0.739224 |         4.44881 |       110.474 |             0.755332 |
  |   1 |      0.960427 |               4.30437 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_01-26-34 |    0.741293 |         4.30437 |       111.818 |           0.00160038 |
  |   2 |      0.949386 |               5.78162 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_04-09-27 |     0.73622 |         5.78162 |       94.3725 |              28.3298 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         | 0.954906 | 0.960427 | 0.960589 |     3 | 0.960751 | 0.956855 |   0.949386 | 0.00647031 |
  | train/loss       |  4.37659 |  4.44881 |  5.11522 |     3 |  5.78162 |  4.84493 |    4.30437 |     0.8144 |
  | train/loss_epoch |  4.37659 |  4.44881 |  5.11522 |     3 |  5.78162 |  4.84493 |    4.30437 |     0.8144 |
  | train/loss_step  | 0.378466 | 0.755332 |  14.5426 |     3 |  28.3298 |  9.69558 | 0.00160038 |    16.1421 |
  | val/f1           | 0.737722 | 0.739224 | 0.740258 |     3 | 0.741293 | 0.738912 |    0.73622 | 0.00255071 |
  | val/loss         |  102.423 |  110.474 |  111.146 |     3 |  111.818 |  105.555 |    94.3725 |    9.70765 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC and coref truncated to 6 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-coref-hoi=6 \
  +model.truncate_models.bert-base-cased-mrpc=6 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/z4603ez5
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yr6oupmy
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/7pp8tudm

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) | ('train/f1',) | ('model_save_dir',)                                                                                          | ('train/loss_step',) | ('train/loss',) |
  | --: | --------------------: | ------------: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------- | -------------------: | --------------: |
  |   0 |               24.2653 |       132.121 |    0.678564 |      0.849682 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_19-38-14 |             0.407075 |         24.2653 |
  |   1 |               15.5781 |       141.729 |    0.685766 |      0.894626 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_21-15-05 |              2.68974 |         15.5781 |
  |   2 |               9.82337 |        154.14 |    0.689022 |      0.921697 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_23-48-40 |             0.055498 |         9.82337 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.872154 | 0.894626 | 0.908162 |     3 | 0.921697 | 0.888668 | 0.849682 |   0.036375 |
  | train/loss       |  12.7008 |  15.5781 |  19.9217 |     3 |  24.2653 |  16.5556 |  9.82337 |    7.27041 |
  | train/loss_epoch |  12.7008 |  15.5781 |  19.9217 |     3 |  24.2653 |  16.5556 |  9.82337 |    7.27041 |
  | train/loss_step  | 0.231286 | 0.407075 |  1.54841 |     3 |  2.68974 |  1.05077 | 0.055498 |    1.43024 |
  | val/f1           | 0.682165 | 0.685766 | 0.687394 |     3 | 0.689022 | 0.684451 | 0.678564 | 0.00535159 |
  | val/loss         |  136.925 |  141.729 |  147.934 |     3 |   154.14 |  142.663 |  132.121 |    11.0394 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC and coref truncated to 8 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-coref-hoi=8 \
  +model.truncate_models.bert-base-cased-mrpc=8 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/wxgvut1s
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/n3mcw7i3
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/k8rtosul

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('model_save_dir',)                                                                                          | ('val/loss',) | ('train/loss_step',) | ('train/loss',) | ('train/loss_epoch',) |
  | --: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | --------------: | --------------------: |
  |   0 |    0.718342 |      0.924871 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_12-49-50 |       194.923 |              25.3459 |         15.3611 |               15.3611 |
  |   1 |    0.718042 |      0.931715 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_15-03-55 |       209.201 |             0.824375 |         13.3327 |               13.3327 |
  |   2 |    0.714161 |      0.921903 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_17-37-35 |       200.833 |                    0 |         16.4375 |               16.4375 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.923387 | 0.924871 | 0.928293 |     3 | 0.931715 | 0.926163 | 0.921903 | 0.00503199 |
  | train/loss       |  14.3469 |  15.3611 |  15.8993 |     3 |  16.4375 |  15.0438 |  13.3327 |    1.57651 |
  | train/loss_epoch |  14.3469 |  15.3611 |  15.8993 |     3 |  16.4375 |  15.0438 |  13.3327 |    1.57651 |
  | train/loss_step  | 0.412188 | 0.824375 |  13.0851 |     3 |  25.3459 |  8.72343 |        0 |    14.4014 |
  | val/f1           | 0.716102 | 0.718042 | 0.718192 |     3 | 0.718342 | 0.716848 | 0.714161 | 0.00233218 |
  | val/loss         |  197.878 |  200.833 |  205.017 |     3 |  209.201 |  201.652 |  194.923 |    7.17426 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC and coref truncated to 10 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-coref-hoi=10 \
  +model.truncate_models.bert-base-cased-mrpc=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dbtkans2
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/a8yzi6t7
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ndym99dr

- metric values per seed

  |     | ('train/loss',) | ('val/loss',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          | ('train/f1',) | ('train/loss_epoch',) |
  | --: | --------------: | ------------: | ----------: | -------------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | --------------------: |
  |   0 |         11.7866 |       251.856 |    0.734522 |              28.5144 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_05-33-34 |      0.947057 |               11.7866 |
  |   1 |         14.0975 |       239.333 |    0.731711 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_07-43-00 |      0.939978 |               14.0975 |
  |   2 |         8.52985 |       274.497 |    0.736524 |              14.4928 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_09-33-17 |      0.958739 |               8.52985 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.943517 | 0.947057 | 0.952898 |     3 | 0.958739 | 0.948591 | 0.939978 | 0.00947435 |
  | train/loss       |  10.1582 |  11.7866 |  12.9421 |     3 |  14.0975 |  11.4713 |  8.52985 |    2.79718 |
  | train/loss_epoch |  10.1582 |  11.7866 |  12.9421 |     3 |  14.0975 |  11.4713 |  8.52985 |    2.79718 |
  | train/loss_step  |   7.2464 |  14.4928 |  21.5036 |     3 |  28.5144 |  14.3357 |        0 |    14.2578 |
  | val/f1           | 0.733117 | 0.734522 | 0.735523 |     3 | 0.736524 | 0.734252 | 0.731711 | 0.00241762 |
  | val/loss         |  245.594 |  251.856 |  263.176 |     3 |  274.497 |  255.228 |  239.333 |     17.823 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model (MRPC and coref truncated to 11 layers)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-coref-hoi=11 \
  +model.truncate_models.bert-base-cased-mrpc=11 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/mdl5uq2q
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3tjr2qso
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/24190ftp

- metric values per seed

  |     | ('val/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('model_save_dir',)                                                                                          | ('train/loss_step',) | ('val/f1',) | ('train/loss',) |
  | --: | ------------: | --------------------: | ------------: | :----------------------------------------------------------------------------------------------------------- | -------------------: | ----------: | --------------: |
  |   0 |       191.524 |               12.1365 |      0.935728 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-10-31_22-43-34 |              70.8124 |    0.730461 |         12.1365 |
  |   1 |       214.888 |                9.9297 |      0.945462 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_00-26-53 |          1.19209e-06 |    0.734877 |          9.9297 |
  |   2 |        230.54 |               7.84307 |      0.955916 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-01_02-38-00 |              11.1448 |    0.736321 |         7.84307 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | --------: |
  | train/f1         | 0.940595 | 0.945462 | 0.950689 |     3 | 0.955916 | 0.945702 |    0.935728 | 0.0100958 |
  | train/loss       |  8.88639 |   9.9297 |  11.0331 |     3 |  12.1365 |  9.96976 |     7.84307 |     2.147 |
  | train/loss_epoch |  8.88639 |   9.9297 |  11.0331 |     3 |  12.1365 |  9.96976 |     7.84307 |     2.147 |
  | train/loss_step  |   5.5724 |  11.1448 |  40.9786 |     3 |  70.8124 |  27.3191 | 1.19209e-06 |   38.0763 |
  | val/f1           | 0.732669 | 0.734877 | 0.735599 |     3 | 0.736321 | 0.733886 |    0.730461 |  0.003053 |
  | val/loss         |  203.206 |  214.888 |  222.714 |     3 |   230.54 |  212.317 |     191.524 |   19.6351 |

## 2023-11-02

### LR Tuning for NER (frozen pretrained RE + frozen bert)

- commands:

  ```bash
  python src/train.py \
  experiment=conll2012_ner-multimodel_frozen+frozen_bert \
  model.learning_rate=1e-4,3e-5,1e-5,3e-6,1e-6 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  validate=true \
  "tags=[ner_multimodel-frozen+frozen_bert,multirun,seeds,lrs]" \
  --multirun
  ```

  ```bash
  python src/train.py \
  experiment=conll2012_ner-multimodel_frozen+frozen_bert \
  model.learning_rate=3e-4,1e-3,3e-3,1e-2 \
  trainer=gpu seed=1,2,3,4,5 \
  "tags=[ner_multimodel-frozen+frozen_bert,multirun,seeds,lrs]" \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs

  - filter for ner_multimodel-frozen+frozen_bert in https://wandb.ai/david-harbecke/conll2012-multi_model_token_classification-training

metric values (averaged over 5 seeds):

| learning rate | mean val/f1 |
| ------------: | ----------: |
|         1e-06 |   0.8939328 |
|         3e-06 |    0.898562 |
|         1e-05 |   0.9019142 |
|         3e-05 |    0.902688 |
|        0.0001 |   0.9049214 |
|        0.0003 |   0.9068722 |
|         0.001 |   0.9084206 |
|         0.003 |   0.8956398 |
|          0.01 |   0.8897254 |

## 2023-11-07

### frozen NER + frozen other model results

- commands:
  frozen+re

  ```bash
  python src/train.py \
  experiment=conll2012_ner-multimodel_base \
  +model.pretrained_models.bert-base-cased-ner-ontonotes=/ds/text/cora4nlp/models/bert-base-cased-ner-ontonotes \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred] \
  +model.aggregate=attention \
  model.learning_rate=1e-3 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  "tags=[ner_multimodel-frozen+frozen_re,multirun,seeds]" \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  -m
  ```

  frozen+qa

  ```bash
  python src/train.py \
  experiment=conll2012_ner-multimodel_base \
  +model.pretrained_models.bert-base-cased-ner-ontonotes=/ds/text/cora4nlp/models/bert-base-cased-ner-ontonotes \
  +model.pretrained_models.bert-base-cased-qa-squad2=/ds/text/cora4nlp/models/bert-base-cased-qa-squad2 \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.learning_rate=1e-3 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  "tags=[ner_multimodel-frozen+frozen_qa,multirun,seeds]" \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  -m
  ```

  frozen+coref

  ```bash
  python src/train.py \
  experiment=conll2012_ner-multimodel_frozen+frozen_coref \
  model.learning_rate=1e-3 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  validate=true \
  "tags=[ner_multimodel-frozen+frozen_coref,multirun,seeds]" \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs

  - runs 132 to 146 in https://wandb.ai/david-harbecke/conll2012-multi_model_token_classification-training

model performances:

| model_combo  |    val_f1 |        std |
| :----------- | --------: | ---------: |
| frozen+re    |  0.909682 | 0.00225593 |
| frozen+qa    |  0.908479 |  0.0010215 |
| frozen+coref |  0.908432 | 0.00109978 |
| frozen+bert  | 0.9084206 | 0.00101738 |

## 2023-11-10

### Coreference resolution - frozen BERT + frozen MRPC model

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/6zwktx8c
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yrzqiu5b
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/4z17r608

- metric values per seed

  |     | ('model_save_dir',)                                                                                          | ('train/loss',) | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) | ('train/loss_step',) | ('val/f1',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | --------------: | --------------------: | ------------: | ------------: | -------------------: | ----------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-20-40 |         10.5841 |               10.5841 |       76.7812 |      0.884537 |          0.000346296 |    0.686408 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_10-46-41 |         12.6906 |               12.6906 |       70.1695 |      0.869121 |             0.923897 |    0.683898 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_13-41-26 |         16.6346 |               16.6346 |       64.8824 |      0.834646 |            0.0108115 |    0.681294 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |   0.851884 |  0.869121 | 0.876829 |     3 | 0.884537 | 0.862768 |    0.834646 |  0.0255448 |
  | train/loss       |    11.6374 |   12.6906 |  14.6626 |     3 |  16.6346 |  13.3031 |     10.5841 |     3.0714 |
  | train/loss_epoch |    11.6374 |   12.6906 |  14.6626 |     3 |  16.6346 |  13.3031 |     10.5841 |     3.0714 |
  | train/loss_step  | 0.00557892 | 0.0108115 | 0.467354 |     3 | 0.923897 | 0.311685 | 0.000346296 |   0.530217 |
  | val/f1           |   0.682596 |  0.683898 | 0.685153 |     3 | 0.686408 | 0.683866 |    0.681294 | 0.00255689 |
  | val/loss         |    67.5259 |   70.1695 |  73.4753 |     3 |  76.7812 |   70.611 |     64.8824 |    5.96168 |

### Coreference resolution - frozen BERT + frozen NER model

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-ner-ontonotes] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/bacd2682
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hi242pfs
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/le6ky0w9

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          | ('train/loss_epoch',) |
  | --: | ------------: | ----------: | --------------: | ------------: | -------------------: | :----------------------------------------------------------------------------------------------------------- | --------------------: |
  |   0 |        82.516 |    0.685581 |         11.3929 |      0.880686 |              23.3543 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-27-40 |               11.3929 |
  |   1 |       70.1193 |    0.676921 |         19.5692 |      0.818133 |              5.27809 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_12-38-14 |               19.5692 |
  |   2 |        78.023 |    0.682847 |         14.0106 |      0.854279 |              19.4495 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_15-30-55 |               14.0106 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.836206 | 0.854279 | 0.867483 |     3 | 0.880686 | 0.851033 | 0.818133 | 0.0314025 |
  | train/loss       |  12.7018 |  14.0106 |  16.7899 |     3 |  19.5692 |  14.9909 |  11.3929 |   4.17536 |
  | train/loss_epoch |  12.7018 |  14.0106 |  16.7899 |     3 |  19.5692 |  14.9909 |  11.3929 |   4.17536 |
  | train/loss_step  |  12.3638 |  19.4495 |  21.4019 |     3 |  23.3543 |  16.0273 |  5.27809 |   9.51162 |
  | val/f1           | 0.679884 | 0.682847 | 0.684214 |     3 | 0.685581 | 0.681783 | 0.676921 | 0.0044269 |
  | val/loss         |  74.0712 |   78.023 |  80.2695 |     3 |   82.516 |  76.8861 |  70.1193 |   6.27602 |

### Coreference resolution - frozen BERT + frozen RE model

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-re-tacred] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cgqif40p
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/sjh3hykx
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/i34kdym9

- metric values per seed

  |     | ('model_save_dir',)                                                                                          | ('train/f1',) | ('val/loss',) | ('train/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss_step',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | ------------: | ------------: | --------------: | --------------------: | ----------: | -------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-29-26 |      0.840297 |       75.0377 |         15.8176 |               15.8176 |    0.674398 |              6.30578 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-27-46 |      0.861832 |       78.8389 |         12.6213 |               12.6213 |    0.675046 |          1.09672e-05 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_11-52-17 |      0.844064 |       75.8241 |          15.415 |                15.415 |    0.675193 |              27.1694 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |         std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ----------: |
  | train/f1         | 0.842181 | 0.844064 | 0.852948 |     3 | 0.861832 | 0.848731 |    0.840297 |   0.0115009 |
  | train/loss       |  14.0181 |   15.415 |  15.6163 |     3 |  15.8176 |   14.618 |     12.6213 |     1.74084 |
  | train/loss_epoch |  14.0181 |   15.415 |  15.6163 |     3 |  15.8176 |   14.618 |     12.6213 |     1.74084 |
  | train/loss_step  |  3.15289 |  6.30578 |  16.7376 |     3 |  27.1694 |  11.1584 | 1.09672e-05 |     14.2199 |
  | val/f1           | 0.674722 | 0.675046 |  0.67512 |     3 | 0.675193 | 0.674879 |    0.674398 | 0.000422932 |
  | val/loss         |  75.4309 |  75.8241 |  77.3315 |     3 |  78.8389 |  76.5669 |     75.0377 |     2.00648 |

### Coreference resolution - frozen BERT + frozen SQUAD model

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/wc40w3qj
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/v43nxqih
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rqsd7xlm

- metric values per seed

  |     | ('train/f1',) | ('train/loss',) | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                          | ('train/loss_step',) | ('train/loss_epoch',) |
  | --: | ------------: | --------------: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------- | -------------------: | --------------------: |
  |   0 |       0.85009 |         14.9106 |    0.684289 |       71.9845 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-35-12 |              20.7907 |               14.9106 |
  |   1 |      0.855491 |         14.6224 |     0.68196 |       73.5919 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-35-57 |            0.0324691 |               14.6224 |
  |   2 |      0.805277 |         21.3112 |    0.671078 |       66.7225 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_11-54-27 |              7.12233 |               21.3112 |

- aggregated values:

  |                  |      25% |     50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | ------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.827684 | 0.85009 |  0.85279 |     3 | 0.855491 | 0.836953 |  0.805277 |  0.0275644 |
  | train/loss       |  14.7665 | 14.9106 |  18.1109 |     3 |  21.3112 |  16.9481 |   14.6224 |    3.78137 |
  | train/loss_epoch |  14.7665 | 14.9106 |  18.1109 |     3 |  21.3112 |  16.9481 |   14.6224 |    3.78137 |
  | train/loss_step  |   3.5774 | 7.12233 |  13.9565 |     3 |  20.7907 |  9.31517 | 0.0324691 |    10.5514 |
  | val/f1           | 0.676519 | 0.68196 | 0.683124 |     3 | 0.684289 | 0.679109 |  0.671078 | 0.00705202 |
  | val/loss         |  69.3535 | 71.9845 |  72.7882 |     3 |  73.5919 |  70.7663 |   66.7225 |    3.59306 |

### Coreference resolution - frozen BERT + frozen NER + RE + SQUAD model

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dxg4v4b8
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/33l33001
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dftju2i5

- metric values per seed

  |     | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/loss',) | ('model_save_dir',)                                                                                          | ('val/f1',) | ('train/f1',) |
  | --: | --------------: | -------------------: | --------------------: | ------------: | :----------------------------------------------------------------------------------------------------------- | ----------: | ------------: |
  |   0 |         19.0141 |              10.5105 |               19.0141 |       62.5853 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-42-31 |    0.681362 |      0.818202 |
  |   1 |         12.0522 |          2.62257e-05 |               12.0522 |       74.3121 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-23-35 |    0.689534 |      0.875125 |
  |   2 |         21.0401 |              22.2709 |               21.0401 |       60.1588 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_12-19-48 |    0.678156 |      0.804767 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         | 0.811485 | 0.818202 | 0.846664 |     3 | 0.875125 | 0.832698 |    0.804767 |   0.037352 |
  | train/loss       |  15.5331 |  19.0141 |  20.0271 |     3 |  21.0401 |  17.3688 |     12.0522 |    4.71446 |
  | train/loss_epoch |  15.5331 |  19.0141 |  20.0271 |     3 |  21.0401 |  17.3688 |     12.0522 |    4.71446 |
  | train/loss_step  |  5.25525 |  10.5105 |  16.3907 |     3 |  22.2709 |  10.9271 | 2.62257e-05 |    11.1413 |
  | val/f1           | 0.679759 | 0.681362 | 0.685448 |     3 | 0.689534 | 0.683018 |    0.678156 | 0.00586709 |
  | val/loss         |   61.372 |  62.5853 |  68.4487 |     3 |  74.3121 |  65.6854 |     60.1588 |     7.5688 |

### Coreference resolution - frozen BERT + frozen MRPC model (10 truncated)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-mrpc] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-mrpc=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/tuuif6m1
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ur5ycnaf
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/knubadh7

- metric values per seed

  |     | ('model_save_dir',)                                                                                          | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('train/loss',) | ('val/loss',) | ('val/f1',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | --------------------: | --------------: | ------------: | ----------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-20-32 |       0.89086 |            0.0152402 |               9.96982 |         9.96982 |       88.4593 |    0.694281 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_10-56-31 |      0.867315 |              6.32578 |               13.4216 |         13.4216 |        83.028 |    0.689885 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_13-42-20 |      0.883699 |              32.0544 |               11.5595 |         11.5595 |       87.6661 |    0.693033 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.875507 | 0.883699 | 0.887279 |     3 |  0.89086 | 0.880625 |  0.867315 |  0.0120697 |
  | train/loss       |  10.7647 |  11.5595 |  12.4906 |     3 |  13.4216 |  11.6503 |   9.96982 |    1.72769 |
  | train/loss_epoch |  10.7647 |  11.5595 |  12.4906 |     3 |  13.4216 |  11.6503 |   9.96982 |    1.72769 |
  | train/loss_step  |  3.17051 |  6.32578 |  19.1901 |     3 |  32.0544 |  12.7985 | 0.0152402 |     16.972 |
  | val/f1           | 0.691459 | 0.693033 | 0.693657 |     3 | 0.694281 |   0.6924 |  0.689885 | 0.00226545 |
  | val/loss         |   85.347 |  87.6661 |  88.0627 |     3 |  88.4593 |  86.3845 |    83.028 |    2.93372 |

### Coreference resolution - frozen BERT + frozen NER model (10 truncated)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-ner-ontonotes] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-ner-ontonotes=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/fx3egjci
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/oww54br0
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/l80ncs9x

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) | ('train/loss',) | ('val/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          |
  | --: | --------------------: | ------------: | ----------: | --------------: | ------------: | -------------------: | :----------------------------------------------------------------------------------------------------------- |
  |   0 |               21.8685 |      0.807399 |    0.679007 |         21.8685 |        71.531 |              19.8663 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-27-10 |
  |   1 |               16.7707 |      0.841518 |     0.68112 |         16.7707 |       76.0788 |              13.2198 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-39-40 |
  |   2 |               14.8018 |      0.854245 |    0.683533 |         14.8018 |       78.3265 |              29.6485 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_12-31-34 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.824459 | 0.841518 | 0.847882 |     3 | 0.854245 | 0.834388 | 0.807399 |  0.0242233 |
  | train/loss       |  15.7863 |  16.7707 |  19.3196 |     3 |  21.8685 |  17.8137 |  14.8018 |    3.64696 |
  | train/loss_epoch |  15.7863 |  16.7707 |  19.3196 |     3 |  21.8685 |  17.8137 |  14.8018 |    3.64696 |
  | train/loss_step  |   16.543 |  19.8663 |  24.7574 |     3 |  29.6485 |  20.9115 |  13.2198 |    8.26407 |
  | val/f1           | 0.680063 |  0.68112 | 0.682326 |     3 | 0.683533 |  0.68122 | 0.679007 | 0.00226472 |
  | val/loss         |  73.8049 |  76.0788 |  77.2027 |     3 |  78.3265 |  75.3121 |   71.531 |    3.46204 |

### Coreference resolution - frozen BERT + frozen RE model (10 truncated)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-re-tacred] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-re-tacred=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0x9z99x3
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/vzpfeupz
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ng84src4

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss',) | ('model_save_dir',)                                                                                          | ('val/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) |
  | --: | -------------------: | --------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | ------------: | --------------------: | ----------: |
  |   0 |              4.42325 |         13.5646 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-30-05 |       83.9001 |      0.863092 |               13.5646 |    0.682587 |
  |   1 |              14.3212 |         12.2174 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-37-15 |       87.5798 |      0.870506 |               12.2174 |    0.679439 |
  |   2 |          0.000112176 |         10.3293 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_11-52-57 |        89.224 |      0.882723 |               10.3293 |    0.685633 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         | 0.866799 | 0.870506 | 0.876614 |     3 | 0.882723 | 0.872107 |    0.863092 | 0.00991307 |
  | train/loss       |  11.2733 |  12.2174 |   12.891 |     3 |  13.5646 |  12.0371 |     10.3293 |    1.62519 |
  | train/loss_epoch |  11.2733 |  12.2174 |   12.891 |     3 |  13.5646 |  12.0371 |     10.3293 |    1.62519 |
  | train/loss_step  |  2.21168 |  4.42325 |  9.37221 |     3 |  14.3212 |  6.24818 | 0.000112176 |    7.33287 |
  | val/f1           | 0.681013 | 0.682587 |  0.68411 |     3 | 0.685633 | 0.682553 |    0.679439 |  0.0030968 |
  | val/loss         |  85.7399 |  87.5798 |  88.4019 |     3 |   89.224 |  86.9013 |     83.9001 |    2.72604 |

### Coreference resolution - frozen BERT + frozen SQUAD model (10 truncated)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-qa-squad2=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/bg1bd0zp
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/oc3u6svz
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ixjsi7y7

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                          | ('val/loss',) | ('train/loss_step',) | ('train/f1',) | ('train/loss',) |
  | --: | --------------------: | ----------: | :----------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | ------------: | --------------: |
  |   0 |               7.57548 |    0.698884 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-34-50 |       95.0514 |              6.36381 |      0.918435 |         7.57548 |
  |   1 |               10.4117 |    0.688572 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_10-42-09 |       89.4354 |              3.55435 |      0.886734 |         10.4117 |
  |   2 |               13.2561 |    0.685383 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_12-57-34 |       83.9019 |              23.4384 |      0.867167 |         13.2561 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |  0.87695 | 0.886734 | 0.902584 |     3 | 0.918435 | 0.890778 | 0.867167 |  0.0258723 |
  | train/loss       |  8.99361 |  10.4117 |  11.8339 |     3 |  13.2561 |  10.4145 |  7.57548 |    2.84033 |
  | train/loss_epoch |  8.99361 |  10.4117 |  11.8339 |     3 |  13.2561 |  10.4145 |  7.57548 |    2.84033 |
  | train/loss_step  |  4.95908 |  6.36381 |  14.9011 |     3 |  23.4384 |  11.1189 |  3.55435 |    10.7611 |
  | val/f1           | 0.686977 | 0.688572 | 0.693728 |     3 | 0.698884 | 0.690946 | 0.685383 | 0.00705697 |
  | val/loss         |  86.6687 |  89.4354 |  92.2434 |     3 |  95.0514 |  89.4629 |  83.9019 |     5.5748 |

### Coreference resolution - frozen BERT + frozen NER + RE + SQUAD model (10 truncated)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased-ner-ontonotes=10 \
  +model.truncate_models.bert-base-cased-re-tacred=10 \
  +model.truncate_models.bert-base-cased-qa-squad2=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/su6pnmy5
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/g1wnhyi7
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/t53qtgs7

- metric values per seed

  |     | ('train/loss',) | ('train/f1',) | ('val/loss',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          |
  | --: | --------------: | ------------: | ------------: | ----------: | --------------------: | -------------------: | :----------------------------------------------------------------------------------------------------------- |
  |   0 |         6.32486 |      0.927737 |       92.9067 |    0.704242 |               6.32486 |              17.7658 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_07-43-09 |
  |   1 |         11.3126 |      0.879467 |       76.9141 |    0.697019 |               11.3126 |              9.63715 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_11-09-32 |
  |   2 |         7.50792 |       0.91795 |       89.5164 |    0.704615 |               7.50792 |            0.0126144 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_13-10-10 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.898709 |  0.91795 | 0.922844 |     3 | 0.927737 | 0.908385 |  0.879467 |  0.0255173 |
  | train/loss       |  6.91639 |  7.50792 |  9.41023 |     3 |  11.3126 |  8.38178 |   6.32486 |    2.60615 |
  | train/loss_epoch |  6.91639 |  7.50792 |  9.41023 |     3 |  11.3126 |  8.38178 |   6.32486 |    2.60615 |
  | train/loss_step  |  4.82488 |  9.63715 |  13.7015 |     3 |  17.7658 |  9.13853 | 0.0126144 |    8.88711 |
  | val/f1           |  0.70063 | 0.704242 | 0.704428 |     3 | 0.704615 | 0.701959 |  0.697019 | 0.00428169 |
  | val/loss         |  83.2152 |  89.5164 |  91.2115 |     3 |  92.9067 |  86.4457 |   76.9141 |     8.4269 |

### Coreference resolution - only frozen BERT

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/q3fb97zc
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8goquwuh
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dtprc212

- metric values per seed

  |     | ('train/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('model_save_dir',)                                                                                          | ('val/f1',) | ('train/loss_step',) | ('train/loss',) |
  | --: | ------------: | --------------------: | ------------: | :----------------------------------------------------------------------------------------------------------- | ----------: | -------------------: | --------------: |
  |   0 |       0.80369 |               22.3254 |       80.0549 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-06-37 |    0.662777 |              28.0013 |         22.3254 |
  |   1 |      0.775203 |               27.4399 |       77.7269 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_11-58-31 |    0.659077 |            0.0292011 |         27.4399 |
  |   2 |      0.818853 |               20.4354 |        79.936 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_14-19-24 |    0.662991 |              1.08551 |         20.4354 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.789446 |  0.80369 | 0.811271 |     3 | 0.818853 | 0.799248 |  0.775203 |  0.0221614 |
  | train/loss       |  21.3804 |  22.3254 |  24.8826 |     3 |  27.4399 |  23.4002 |   20.4354 |    3.62383 |
  | train/loss_epoch |  21.3804 |  22.3254 |  24.8826 |     3 |  27.4399 |  23.4002 |   20.4354 |    3.62383 |
  | train/loss_step  | 0.557357 |  1.08551 |  14.5434 |     3 |  28.0013 |  9.70533 | 0.0292011 |    15.8536 |
  | val/f1           | 0.660927 | 0.662777 | 0.662884 |     3 | 0.662991 | 0.661615 |  0.659077 | 0.00220043 |
  | val/loss         |  78.8314 |   79.936 |  79.9954 |     3 |  80.0549 |  79.2392 |   77.7269 |    1.31109 |

### Coreference resolution - frozen BERT + frozen BERT model

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased2:bert-base-cased} \
  +model.freeze_models=[bert-base-cased,bert-base-cased2] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ts3b6fg7
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/q6dg6t2l
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/lzhr9h3c

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss',) | ('model_save_dir',)                                                                                          | ('train/f1',) | ('train/loss_step',) | ('val/loss',) | ('val/f1',) |
  | --: | --------------------: | --------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | ------------: | ----------: |
  |   0 |               13.5734 |         13.5734 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_09-06-27 |      0.857325 |             0.145913 |       72.0982 |    0.674364 |
  |   1 |               14.9625 |         14.9625 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_12-14-56 |      0.845529 |              2.29504 |       72.0664 |    0.668144 |
  |   2 |               9.71805 |         9.71805 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-09_15-09-19 |      0.889045 |          4.17233e-06 |       81.2134 |    0.674663 |

- aggregated values:

  |                  |       25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | --------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |  0.851427 | 0.857325 | 0.873185 |     3 | 0.889045 | 0.863966 |    0.845529 |  0.0225053 |
  | train/loss       |   11.6457 |  13.5734 |   14.268 |     3 |  14.9625 |  12.7513 |     9.71805 |    2.71716 |
  | train/loss_epoch |   11.6457 |  13.5734 |   14.268 |     3 |  14.9625 |  12.7513 |     9.71805 |    2.71716 |
  | train/loss_step  | 0.0729585 | 0.145913 |  1.22048 |     3 |  2.29504 | 0.813652 | 4.17233e-06 |    1.28499 |
  | val/f1           |  0.671254 | 0.674364 | 0.674514 |     3 | 0.674663 |  0.67239 |    0.668144 | 0.00368031 |
  | val/loss         |   72.0823 |  72.0982 |  76.6558 |     3 |  81.2134 |   75.126 |     72.0664 |    5.27185 |

## 2023-11-13

### Coreference resolution - frozen BERT + frozen BERT 3x

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased2:bert-base-cased,bert-base-cased3:bert-base-cased,bert-base-cased4:bert-base-cased} \
  +model.freeze_models=[bert-base-cased,bert-base-cased2,bert-base-cased3,bert-base-cased4] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/008sq0vi
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pzo7v0yf
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qvonj2xh

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                          | ('train/loss',) | ('train/loss_step',) | ('train/f1',) |
  | --: | --------------------: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------- | --------------: | -------------------: | ------------: |
  |   0 |               11.0679 |    0.671336 |       69.1052 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-13_07-52-29 |         11.0679 |              78.4393 |      0.870907 |
  |   1 |               10.2549 |    0.675043 |       73.8249 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-13_10-41-18 |         10.2549 |              2.62202 |      0.886521 |
  |   2 |               17.0521 |    0.668907 |        59.043 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-13_13-52-33 |         17.0521 |              51.8622 |      0.826871 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.848889 | 0.870907 | 0.878714 |     3 | 0.886521 | 0.861433 | 0.826871 |  0.0309329 |
  | train/loss       |  10.6614 |  11.0679 |    14.06 |     3 |  17.0521 |  12.7916 |  10.2549 |    3.71202 |
  | train/loss_epoch |  10.6614 |  11.0679 |    14.06 |     3 |  17.0521 |  12.7916 |  10.2549 |    3.71202 |
  | train/loss_step  |  27.2421 |  51.8622 |  65.1508 |     3 |  78.4393 |  44.3078 |  2.62202 |     38.469 |
  | val/f1           | 0.670121 | 0.671336 |  0.67319 |     3 | 0.675043 | 0.671762 | 0.668907 | 0.00309035 |
  | val/loss         |  64.0741 |  69.1052 |   71.465 |     3 |  73.8249 |  67.3244 |   59.043 |    7.55016 |

### Coreference resolution - frozen BERT + frozen BERT 3x (10 truncated)

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased2:bert-base-cased,bert-base-cased3:bert-base-cased,bert-base-cased4:bert-base-cased} \
  +model.freeze_models=[bert-base-cased,bert-base-cased2,bert-base-cased3,bert-base-cased4] \
  +model.aggregate=attention \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  +model.truncate_models.bert-base-cased2=10 \
  +model.truncate_models.bert-base-cased3=10 \
  +model.truncate_models.bert-base-cased4=10 \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/2686roqh
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/lnkbq9t0
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/p3gv9qyf

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss_epoch',) | ('train/loss',) | ('train/f1',) | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                          |
  | --: | -------------------: | --------------------: | --------------: | ------------: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------- |
  |   0 |            0.0156929 |               8.36614 |         8.36614 |      0.911241 |    0.693032 |       91.5534 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-13_07-52-39 |
  |   1 |           0.00188104 |               8.08781 |         8.08781 |      0.912119 |    0.691283 |       90.7219 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-13_11-21-59 |
  |   2 |              2.84362 |               10.4116 |         10.4116 |      0.886894 |     0.69246 |       77.2555 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-13_14-59-29 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |     mean |        min |         std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | -------: | ---------: | ----------: |
  | train/f1         |   0.899067 |  0.911241 |  0.91168 |     3 | 0.912119 | 0.903418 |   0.886894 |   0.0143172 |
  | train/loss       |    8.22697 |   8.36614 |  9.38885 |     3 |  10.4116 |  8.95517 |    8.08781 |     1.26893 |
  | train/loss_epoch |    8.22697 |   8.36614 |  9.38885 |     3 |  10.4116 |  8.95517 |    8.08781 |     1.26893 |
  | train/loss_step  | 0.00878699 | 0.0156929 |  1.42966 |     3 |  2.84362 | 0.953731 | 0.00188104 |     1.63671 |
  | val/f1           |   0.691871 |   0.69246 | 0.692746 |     3 | 0.693032 | 0.692258 |   0.691283 | 0.000891977 |
  | val/loss         |    83.9887 |   90.7219 |  91.1377 |     3 |  91.5534 |  86.5103 |    77.2555 |     8.02567 |

## 2023-11-16

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/5crxriyp
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0inmbtz0
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/jm8v7un4

- metric values per seed

  |     | ('model_save_dir',)                                                                                          | ('val/loss',) | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | ------------: | --------------------: | ----------: | --------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-15_23-03-37 |       170.254 |            0.0402214 |      0.825021 |               47.6317 |    0.712913 |         47.6317 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_01-25-46 |       163.803 |              59.5883 |      0.812954 |               53.9894 |    0.704383 |         53.9894 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_03-20-54 |       162.376 |            0.0229799 |      0.821196 |               49.5471 |    0.712752 |         49.5471 |

- aggregated values:

  |                  |       25% |       50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | --------: | --------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         |  0.817075 |  0.821196 | 0.823108 |     3 | 0.825021 | 0.819724 |  0.812954 | 0.00616682 |
  | train/loss       |   48.5894 |   49.5471 |  51.7683 |     3 |  53.9894 |  50.3894 |   47.6317 |    3.26149 |
  | train/loss_epoch |   48.5894 |   49.5471 |  51.7683 |     3 |  53.9894 |  50.3894 |   47.6317 |    3.26149 |
  | train/loss_step  | 0.0316006 | 0.0402214 |  29.8143 |     3 |  59.5883 |  19.8838 | 0.0229799 |    34.3851 |
  | val/f1           |  0.708568 |  0.712752 | 0.712832 |     3 | 0.712913 | 0.710016 |  0.704383 | 0.00487871 |
  | val/loss         |    163.09 |   163.803 |  167.029 |     3 |  170.254 |  165.478 |   162.376 |    4.19771 |

### Coreference resolution - frozen pre-trained target-model + frozen MRPC model with sum aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate=sum \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0w5213um
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/1kglce1w
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/h2hwkut1

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('val/loss',) | ('model_save_dir',)                                                                                          | ('train/f1',) | ('train/loss',) | ('val/f1',) |
  | --: | --------------------: | -------------------: | ------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | --------------: | ----------: |
  |   0 |               203.842 |          1.54972e-06 |       611.204 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-15_23-04-03 |      0.803246 |         203.842 |    0.703389 |
  |   1 |               228.301 |                    0 |       600.567 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_01-36-36 |      0.786511 |         228.301 |    0.696807 |
  |   2 |               147.375 |            0.0159391 |       596.709 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_03-36-34 |      0.835945 |         147.375 |    0.715437 |

- aggregated values:

  |                  |        25% |         50% |        75% | count |       max |       mean |      min |        std |
  | :--------------- | ---------: | ----------: | ---------: | ----: | --------: | ---------: | -------: | ---------: |
  | train/f1         |   0.794879 |    0.803246 |   0.819596 |     3 |  0.835945 |   0.808568 | 0.786511 |  0.0251431 |
  | train/loss       |    175.609 |     203.842 |    216.072 |     3 |   228.301 |    193.173 |  147.375 |    41.5046 |
  | train/loss_epoch |    175.609 |     203.842 |    216.072 |     3 |   228.301 |    193.173 |  147.375 |    41.5046 |
  | train/loss_step  | 7.7486e-07 | 1.54972e-06 | 0.00797033 |     3 | 0.0159391 | 0.00531355 |        0 |   0.009202 |
  | val/f1           |   0.700098 |    0.703389 |   0.709413 |     3 |  0.715437 |   0.705211 | 0.696807 | 0.00944762 |
  | val/loss         |    598.638 |     600.567 |    605.886 |     3 |   611.204 |    602.827 |  596.709 |    7.50692 |

### Coreference resolution - frozen BERT + frozen MRPC model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-mrpc] \
  +model.aggregate=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/20cd29eu
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/bsizzsgy
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/r6gxnrw0

- metric values per seed

  |     | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          |
  | --: | ----------: | --------------------: | ------------: | ------------: | --------------: | -------------------: | :----------------------------------------------------------------------------------------------------------- |
  |   0 |    0.606199 |               181.579 |       236.746 |      0.620931 |         181.579 |              88.3647 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-15_23-04-40 |
  |   1 |    0.642766 |               120.206 |       252.035 |      0.704447 |         120.206 |              3.03073 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_00-18-49 |
  |   2 |    0.648812 |                101.93 |        248.69 |      0.732535 |          101.93 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_03-04-47 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.662689 | 0.704447 | 0.718491 |     3 | 0.732535 | 0.685971 | 0.620931 | 0.0580506 |
  | train/loss       |  111.068 |  120.206 |  150.893 |     3 |  181.579 |  134.572 |   101.93 |   41.7226 |
  | train/loss_epoch |  111.068 |  120.206 |  150.893 |     3 |  181.579 |  134.572 |   101.93 |   41.7226 |
  | train/loss_step  |  1.51536 |  3.03073 |  45.6977 |     3 |  88.3647 |  30.4652 |        0 |   50.1654 |
  | val/f1           | 0.624482 | 0.642766 | 0.645789 |     3 | 0.648812 | 0.632592 | 0.606199 | 0.0230561 |
  | val/loss         |  242.718 |   248.69 |  250.363 |     3 |  252.035 |  245.824 |  236.746 |   8.03726 |

### Coreference resolution - frozen BERT + frozen MRPC model with sum aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-mrpc] \
  +model.aggregate=sum \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/6772eqtu
  - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/p2wd6zvz
  - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/o6uc89gi

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                          | ('train/loss_step',) | ('train/f1',) | ('train/loss',) |
  | --: | --------------------: | ------------: | ----------: | :----------------------------------------------------------------------------------------------------------- | -------------------: | ------------: | --------------: |
  |   0 |               645.012 |       929.096 |    0.612491 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-15_23-04-29 |                    0 |      0.640408 |         645.012 |
  |   1 |               467.937 |        907.93 |    0.633671 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_00-56-08 |                    0 |      0.695304 |         467.937 |
  |   2 |               601.655 |       943.636 |    0.623996 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-11-16_04-01-27 |              479.947 |      0.658383 |         601.655 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.649395 | 0.658383 | 0.676844 |     3 | 0.695304 | 0.664698 | 0.640408 | 0.0279879 |
  | train/loss       |  534.796 |  601.655 |  623.333 |     3 |  645.012 |  571.534 |  467.937 |      92.3 |
  | train/loss_epoch |  534.796 |  601.655 |  623.333 |     3 |  645.012 |  571.534 |  467.937 |      92.3 |
  | train/loss_step  |        0 |        0 |  239.974 |     3 |  479.947 |  159.982 |        0 |   277.098 |
  | val/f1           | 0.618244 | 0.623996 | 0.628834 |     3 | 0.633671 | 0.623386 | 0.612491 | 0.0106029 |
  | val/loss         |  918.513 |  929.096 |  936.366 |     3 |  943.636 |  926.887 |   907.93 |   17.9555 |

## 2023-11-25

### Bert LR Tuning for Coref (frozen pretrained Coref + frozen MRPC), attention aggregation with query, key and value projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=1e-4,5e-5,1e-5,1e-6 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs

  - lr=1e-4

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/41srr7dt
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/o5fjyx0d
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yft5lm39

  - lr=5e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/9ef63xca
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ouvdbqjj
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zsm9qiub

  - lr=1e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ru33ldve
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ka4vjfpi
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/nc1pdk86

  - lr=1e-6

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/9fea5fzt
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/g9v17luf
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/mdyyjmyw

metric values (averaged over 5 seeds):

| learning rate | mean val/f1 | mean val/loss |
| ------------: | ----------: | ------------: |
|          1e-4 |      0.7294 |       102.149 |
|          5e-5 |      0.7332 |       103.068 |
|          1e-5 |      0.7372 |       89.4812 |
|          1e-6 |      0.7335 |       78.4313 |

### BERT LR Tuning for Coref (frozen pretrained Coref + frozen MRPC), attention aggregation without query, key and value projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.project_target_query=False \
  +model.aggregate.project_target_key=False \
  +model.aggregate.project_target_value=False \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=1e-4,5e-5,1e-5,1e-6 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun
  ```

- wandb runs

  - lr=1e-4

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/i58mquzs
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/26mqvit1
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8xuquxv3

  - lr=5e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/4ng9o63h
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pw3407qw
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/uz4snov4

  - lr=1e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/5u27ptw6
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yvpeqk1c
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/xys39eem

  - lr=1e-6

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/91830t3n
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ltrv4m3e
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/928zva5d

metric values (averaged over 5 seeds):

| learning rate | mean val/f1 | mean val/loss |
| ------------: | ----------: | ------------: |
|          1e-4 |      0.7358 |       121.089 |
|          5e-5 |      0.7376 |       124.365 |
|          1e-5 |      0.7372 |       121.953 |
|          1e-6 |      0.7370 |       126.127 |

### BERT LR Tuning for Coref (frozen pretrained Coref + frozen MRPC), attention aggregation with query projection but w/o key and value projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.project_target_key=False \
  +model.aggregate.project_target_value=False \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=1e-4,5e-5,1e-5,1e-6 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs

  - lr=1e-4

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0i0d916y
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/vk3mr2a9
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/s9tvr00u

  - lr=5e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/laspio2n
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/m43zvcb3
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/46fno3vr

  - lr=1e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yeaqnb14
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/6riyh6qo
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/1vtoi4xf

  - lr=1e-6

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/4fgs54p8
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/er607zrr
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ap9cyv3t

metric values (averaged over 5 seeds):

| learning rate | mean val/f1 | mean val/loss |
| ------------: | ----------: | ------------: |
|          1e-4 |      0.7401 |       134.469 |
|          5e-5 |      0.7387 |       126.815 |
|          1e-5 |      0.7357 |       108.715 |
|          1e-6 |      0.7356 |       113.756 |

### BERT LR Tuning for Coref (frozen pretrained Coref + frozen MRPC), attention aggregation without query projection but with key and value projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.project_target_query=False \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=1e-4,5e-5,1e-5,1e-6 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs

  - lr=1e-4

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/auyvwg51
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/81fa2ol4
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/asi02c80

  - lr=5e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/bt5tyqpq
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/bmn9ub6l
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/w1d62auj

  - lr=1e-5

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/c8hqgaxy
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ziwvs2jp
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/yam6vv55

  - lr=1e-6

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/b5op2hqy
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/nmu0a8e0
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/jfx7rcyv

metric values (averaged over 5 seeds):

| learning rate | mean val/f1 | mean val/loss |
| ------------: | ----------: | ------------: |
|          1e-4 |      0.7305 |       116.805 |
|          5e-5 |      0.7339 |       106.148 |
|          1e-5 |      0.7406 |       110.213 |
|          1e-6 |      0.7317 |       85.3664 |

## 2023-11-27

### Coreference probing - frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/7sp7i7z8
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/x4joy5q7
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/20zujuab
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/sviwzhlj
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/sxbz4lg6

- metric values per seed

  |     | ('train/loss_step',) | ('model_save_dir',)                                                                             | ('val/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) | ('train/loss',) |
  | --: | -------------------: | :---------------------------------------------------------------------------------------------- | ------------: | --------------------: | ------------: | ----------: | --------------: |
  |   0 |               1.4731 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-39-25 |       158.822 |               5.55277 |      0.958079 |    0.741642 |         5.55277 |
  |   1 |          0.000248669 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_15-15-13 |       127.793 |               7.32384 |      0.942904 |    0.735015 |         7.32384 |
  |   2 |           0.00107598 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_16-28-39 |       117.221 |               8.48779 |       0.93677 |    0.734824 |         8.48779 |
  |   3 |              3.17588 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_17-24-31 |       125.025 |               7.65056 |      0.941428 |    0.739178 |         7.65056 |
  |   4 |             0.297291 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_18-30-18 |       137.035 |               6.92449 |      0.948987 |    0.736825 |         6.92449 |

- aggregated values:

  |                  |        25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | ---------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |   0.941428 | 0.942904 | 0.948987 |     5 | 0.958079 | 0.945634 |     0.93677 | 0.00821361 |
  | train/loss       |    6.92449 |  7.32384 |  7.65056 |     5 |  8.48779 |  7.18789 |     5.55277 |    1.07997 |
  | train/loss_epoch |    6.92449 |  7.32384 |  7.65056 |     5 |  8.48779 |  7.18789 |     5.55277 |    1.07997 |
  | train/loss_step  | 0.00107598 | 0.297291 |   1.4731 |     5 |  3.17588 |  0.98952 | 0.000248669 |    1.36463 |
  | val/f1           |   0.735015 | 0.736825 | 0.739178 |     5 | 0.741642 | 0.737497 |    0.734824 | 0.00290528 |
  | val/loss         |    125.025 |  127.793 |  137.035 |     5 |  158.822 |  133.179 |     117.221 |    15.9887 |

### Coreference probing - frozen BERT model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/eb72975u
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/8lgo8bn2
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/56y05fft
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/8gv04dm9
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/mr0m7y6c

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                             | ('train/loss_epoch',) | ('train/loss',) | ('val/f1',) | ('val/loss',) | ('train/loss_step',) |
  | --: | ------------: | :---------------------------------------------------------------------------------------------- | --------------------: | --------------: | ----------: | ------------: | -------------------: |
  |   0 |      0.733135 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-40-16 |               120.961 |         120.961 |    0.640133 |       304.027 |              259.175 |
  |   1 |      0.763081 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_15-49-04 |               96.4745 |         96.4745 |    0.652958 |       297.137 |              217.692 |
  |   2 |      0.724118 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_19-42-52 |               126.606 |         126.606 |     0.64114 |       293.755 |              87.4643 |
  |   3 |      0.799676 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_22-39-30 |               65.4952 |         65.4952 |    0.664069 |       288.014 |             0.439429 |
  |   4 |      0.758565 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-28_03-54-45 |               98.4221 |         98.4221 |    0.649033 |       296.229 |              106.844 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.733135 | 0.758565 | 0.763081 |     5 | 0.799676 | 0.755715 | 0.724118 |  0.0295938 |
  | train/loss       |  96.4745 |  98.4221 |  120.961 |     5 |  126.606 |  101.592 |  65.4952 |    24.1871 |
  | train/loss_epoch |  96.4745 |  98.4221 |  120.961 |     5 |  126.606 |  101.592 |  65.4952 |    24.1871 |
  | train/loss_step  |  87.4643 |  106.844 |  217.692 |     5 |  259.175 |  134.323 | 0.439429 |     104.18 |
  | val/f1           |  0.64114 | 0.649033 | 0.652958 |     5 | 0.664069 | 0.649467 | 0.640133 | 0.00977304 |
  | val/loss         |  293.755 |  296.229 |  297.137 |     5 |  304.027 |  295.832 |  288.014 |    5.79765 |

### Coreference probing - frozen RE model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/wxvxocwe
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/06woj96v
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/a9w24ttd
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/kq48bbmp
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/4rqolpum

- metric values per seed

  |     | ('model_save_dir',)                                                                             | ('train/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss',) | ('train/loss_step',) | ('val/f1',) |
  | --: | :---------------------------------------------------------------------------------------------- | ------------: | --------------------: | ------------: | --------------: | -------------------: | ----------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-23-19 |      0.455409 |               763.513 |       776.472 |         763.513 |              123.829 |    0.524302 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_14-12-25 |      0.483603 |               638.121 |       664.463 |         638.121 |              400.634 |    0.542233 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_16-52-03 |      0.503018 |               567.024 |        608.99 |         567.024 |              1664.13 |    0.549073 |
  |   3 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_20-05-11 |      0.435768 |                827.78 |       702.319 |          827.78 |              201.165 |    0.505498 |
  |   4 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_21-25-27 |      0.422276 |                868.95 |       705.394 |          868.95 |              2594.05 |    0.492537 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.435768 | 0.455409 | 0.483603 |     5 | 0.503018 | 0.460015 | 0.422276 | 0.0333131 |
  | train/loss       |  638.121 |  763.513 |   827.78 |     5 |   868.95 |  733.077 |  567.024 |   127.423 |
  | train/loss_epoch |  638.121 |  763.513 |   827.78 |     5 |   868.95 |  733.077 |  567.024 |   127.423 |
  | train/loss_step  |  201.165 |  400.634 |  1664.13 |     5 |  2594.05 |  996.763 |  123.829 |   1089.39 |
  | val/f1           | 0.505498 | 0.524302 | 0.542233 |     5 | 0.549073 | 0.522729 | 0.492537 | 0.0239027 |
  | val/loss         |  664.463 |  702.319 |  705.394 |     5 |  776.472 |  691.528 |   608.99 |   61.3781 |

### Coreference probing - frozen NER model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/js4yslo7
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/wv6q9uni
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/ncjnkkdx
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/nhoy5hu0
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/h0mcub6i

- metric values per seed

  |     | ('val/loss',) | ('train/f1',) | ('model_save_dir',)                                                                             | ('val/f1',) | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) |
  | --: | ------------: | ------------: | :---------------------------------------------------------------------------------------------- | ----------: | --------------: | -------------------: | --------------------: |
  |   0 |       497.918 |      0.233634 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-13-40 |    0.360306 |         1102.93 |              174.784 |               1102.93 |
  |   1 |       513.598 |      0.263322 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_14-02-42 |    0.386692 |         949.213 |              2044.66 |               949.213 |
  |   2 |       450.247 |      0.240942 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_16-33-51 |    0.362483 |         1043.54 |              1209.72 |               1043.54 |
  |   3 |       484.081 |      0.210222 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_18-35-48 |    0.335006 |         1076.97 |              318.819 |               1076.97 |
  |   4 |       617.714 |      0.232183 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_19-55-54 |    0.337092 |            1087 |              245.693 |                  1087 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |      std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | -------: |
  | train/f1         | 0.232183 | 0.233634 | 0.240942 |     5 | 0.263322 |  0.23606 | 0.210222 | 0.019076 |
  | train/loss       |  1043.54 |  1076.97 |     1087 |     5 |  1102.93 |  1051.93 |  949.213 |   61.398 |
  | train/loss_epoch |  1043.54 |  1076.97 |     1087 |     5 |  1102.93 |  1051.93 |  949.213 |   61.398 |
  | train/loss_step  |  245.693 |  318.819 |  1209.72 |     5 |  2044.66 |  798.736 |  174.784 |   813.44 |
  | val/f1           | 0.337092 | 0.360306 | 0.362483 |     5 | 0.386692 | 0.356316 | 0.335006 | 0.021215 |
  | val/loss         |  484.081 |  497.918 |  513.598 |     5 |  617.714 |  512.712 |  450.247 |  63.1797 |

### Coreference probing - frozen QA model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/3icnjjr1
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/i65btsmt
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/pcys2gt6
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/jbh6oh6k
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/4qik98wr

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                             | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('train/loss',) |
  | --: | ------------: | ----------: | :---------------------------------------------------------------------------------------------- | ------------: | -------------------: | --------------------: | --------------: |
  |   0 |       627.943 |    0.475061 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-26-09 |      0.351961 |              259.486 |                  1055 |            1055 |
  |   1 |       644.343 |    0.481027 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_13-11-35 |       0.36805 |              93.5476 |               955.346 |         955.346 |
  |   2 |       558.387 |    0.514253 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_14-10-34 |      0.377747 |              1543.43 |               958.217 |         958.217 |
  |   3 |       496.607 |    0.548026 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_15-19-31 |      0.423342 |              1624.32 |               734.051 |         734.051 |
  |   4 |       558.839 |    0.521168 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_17-32-34 |      0.394363 |              2299.08 |               864.595 |         864.595 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         |  0.36805 | 0.377747 | 0.394363 |     5 | 0.423342 | 0.383093 | 0.351961 | 0.0272538 |
  | train/loss       |  864.595 |  955.346 |  958.217 |     5 |     1055 |  913.442 |  734.051 |   120.796 |
  | train/loss_epoch |  864.595 |  955.346 |  958.217 |     5 |     1055 |  913.442 |  734.051 |   120.796 |
  | train/loss_step  |  259.486 |  1543.43 |  1624.32 |     5 |  2299.08 |  1163.97 |  93.5476 |   949.774 |
  | val/f1           | 0.481027 | 0.514253 | 0.521168 |     5 | 0.548026 | 0.507907 | 0.475061 | 0.0301127 |
  | val/loss         |  558.387 |  558.839 |  627.943 |     5 |  644.343 |  577.224 |  496.607 |   59.7271 |

### Coreference probing - frozen MRPC model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/eweziqil
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/av6avffq
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/9o771wxw
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/hxt74vj7
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/gor1qh7z

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('train/loss_step',) | ('train/loss',) | ('train/f1',) | ('val/f1',) | ('val/loss',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------- | -------------------: | --------------: | ------------: | ----------: | ------------: |
  |   0 |               188.633 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-27-58 |              185.031 |         188.633 |      0.626526 |    0.643156 |       242.417 |
  |   1 |               370.482 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_15-37-46 |              6.73763 |         370.482 |      0.507083 |    0.567381 |       294.321 |
  |   2 |               274.123 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_16-30-07 |              307.393 |         274.123 |      0.575522 |    0.612532 |       274.487 |
  |   3 |               242.128 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_18-27-08 |              573.903 |         242.128 |      0.589892 |     0.62735 |        263.39 |
  |   4 |               289.404 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_20-40-25 |              235.768 |         289.404 |       0.56451 |     0.60735 |        281.01 |

- aggregated values:

  |                  |     25% |      50% |      75% | count |      max |     mean |      min |      std |
  | :--------------- | ------: | -------: | -------: | ----: | -------: | -------: | -------: | -------: |
  | train/f1         | 0.56451 | 0.575522 | 0.589892 |     5 | 0.626526 | 0.572707 | 0.507083 | 0.043513 |
  | train/loss       | 242.128 |  274.123 |  289.404 |     5 |  370.482 |  272.954 |  188.633 |  66.7907 |
  | train/loss_epoch | 242.128 |  274.123 |  289.404 |     5 |  370.482 |  272.954 |  188.633 |  66.7907 |
  | train/loss_step  | 185.031 |  235.768 |  307.393 |     5 |  573.903 |  261.767 |  6.73763 |   206.83 |
  | val/f1           | 0.60735 | 0.612532 |  0.62735 |     5 | 0.643156 | 0.611554 | 0.567381 | 0.028364 |
  | val/loss         |  263.39 |  274.487 |   281.01 |     5 |  294.321 |  271.125 |  242.417 |  19.5644 |

### Coreference probing - frozen NER-RE-QA models with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/mx64s1vg
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/uf4jgjw6
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/w3qmynd1
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/a9xmm0hl
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/n28x42dp

- metric values per seed

  |     | ('model_save_dir',)                                                                             | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) |
  | --: | :---------------------------------------------------------------------------------------------- | ------------: | ----------: | --------------: | -------------------: | ------------: | --------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-33-46 |       274.809 |     0.59684 |         259.254 |              224.631 |       0.53631 |               259.254 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_14-24-34 |       287.573 |    0.574819 |          290.68 |              1.10599 |      0.514521 |                290.68 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_15-51-59 |       235.945 |    0.620982 |         178.743 |              511.779 |      0.583443 |               178.743 |
  |   3 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_19-01-00 |       279.827 |    0.540992 |         353.591 |              117.746 |      0.463194 |               353.591 |
  |   4 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_19-46-14 |       269.939 |    0.562947 |         316.907 |              1227.01 |      0.491758 |               316.907 |

- aggregated values:

  |                  |      25% |      50% |     75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | ------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.491758 | 0.514521 | 0.53631 |     5 | 0.583443 | 0.517845 | 0.463194 | 0.0456134 |
  | train/loss       |  259.254 |   290.68 | 316.907 |     5 |  353.591 |  279.835 |  178.743 |   66.2854 |
  | train/loss_epoch |  259.254 |   290.68 | 316.907 |     5 |  353.591 |  279.835 |  178.743 |   66.2854 |
  | train/loss_step  |  117.746 |  224.631 | 511.779 |     5 |  1227.01 |  416.455 |  1.10599 |   491.077 |
  | val/f1           | 0.562947 | 0.574819 | 0.59684 |     5 | 0.620982 | 0.579316 | 0.540992 | 0.0308223 |
  | val/loss         |  269.939 |  274.809 | 279.827 |     5 |  287.573 |  269.619 |  235.945 |   19.9218 |

### Coreference probing - frozen BERT-3x models with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased1:bert-base-cased,bert-base-cased2:bert-base-cased,bert-base-cased3:bert-base-cased} \
  +model.freeze_models=[bert-base-cased1,bert-base-cased2,bert-base-cased3] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3,4,5 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/coref-task \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-task-training/runs/h43bdhsx
  - seed2: https://wandb.ai/tanikina/probing-coref-task-training/runs/0a8rhdp9
  - seed3: https://wandb.ai/tanikina/probing-coref-task-training/runs/mf3g0j7y
  - seed4: https://wandb.ai/tanikina/probing-coref-task-training/runs/dwwtj3yy
  - seed5: https://wandb.ai/tanikina/probing-coref-task-training/runs/v31flf2w

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('train/f1',) | ('train/loss_step',) | ('val/f1',) | ('train/loss',) | ('val/loss',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------- | ------------: | -------------------: | ----------: | --------------: | ------------: |
  |   0 |               110.859 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_12-36-13 |      0.742393 |              30.2792 |    0.620946 |         110.859 |       329.183 |
  |   1 |               121.623 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_14-24-42 |      0.724747 |              199.266 |    0.615854 |         121.623 |         317.2 |
  |   2 |               93.0457 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_16-02-43 |      0.762884 |               518.12 |     0.62584 |         93.0457 |       319.408 |
  |   3 |               90.1024 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_18-17-52 |      0.768413 |              148.791 |    0.625622 |         90.1024 |       321.355 |
  |   4 |               136.488 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-task/2023-11-27_20-35-47 |      0.701205 |              669.913 |    0.609145 |         136.488 |       314.382 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.724747 | 0.742393 | 0.762884 |     5 | 0.768413 | 0.739928 | 0.701205 |  0.0277235 |
  | train/loss       |  93.0457 |  110.859 |  121.623 |     5 |  136.488 |  110.424 |  90.1024 |    19.4931 |
  | train/loss_epoch |  93.0457 |  110.859 |  121.623 |     5 |  136.488 |  110.424 |  90.1024 |    19.4931 |
  | train/loss_step  |  148.791 |  199.266 |   518.12 |     5 |  669.913 |  313.274 |  30.2792 |    268.929 |
  | val/f1           | 0.615854 | 0.620946 | 0.625622 |     5 |  0.62584 | 0.619482 | 0.609145 | 0.00707622 |
  | val/loss         |    317.2 |  319.408 |  321.355 |     5 |  329.183 |  320.306 |  314.382 |    5.60019 |

## 2023-12-04

### Coreference probing - frozen target model with mean aggregation, truncated to 6 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-coref-hoi=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/re4imkkv
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/6zjs5sq4
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/2ccxjchn

- metric values per seed

  |     | ('train/loss_step',) | ('val/f1',) | ('train/loss',) | ('val/loss',) | ('model_save_dir',)                                                                                         | ('train/loss_epoch',) | ('train/f1',) |
  | --: | -------------------: | ----------: | --------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | --------------------: | ------------: |
  |   0 |                    0 |    0.673419 |         277.198 |        1101.4 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_14-50-53 |               277.198 |      0.817711 |
  |   1 |               870.28 |    0.673573 |         286.565 |       1085.88 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_16-45-20 |               286.565 |      0.809235 |
  |   2 |                    0 |    0.669878 |         315.441 |       1025.97 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_18-27-37 |               315.441 |      0.796406 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.802821 | 0.809235 | 0.813473 |     3 | 0.817711 | 0.807784 | 0.796406 |  0.0107264 |
  | train/loss       |  281.881 |  286.565 |  301.003 |     3 |  315.441 |  293.068 |  277.198 |    19.9336 |
  | train/loss_epoch |  281.881 |  286.565 |  301.003 |     3 |  315.441 |  293.068 |  277.198 |    19.9336 |
  | train/loss_step  |        0 |        0 |   435.14 |     3 |   870.28 |  290.093 |        0 |    502.456 |
  | val/f1           | 0.671649 | 0.673419 | 0.673496 |     3 | 0.673573 |  0.67229 | 0.669878 | 0.00209014 |
  | val/loss         |  1055.93 |  1085.88 |  1093.64 |     3 |   1101.4 |  1071.08 |  1025.97 |    39.8292 |

### Coreference probing - frozen target model with mean aggregation, truncated to 7 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-coref-hoi=7 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/w422hh7k
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/eu7zzyed
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/lt6nmxht

- metric values per seed

  |     | ('train/loss',) | ('val/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('train/f1',) | ('val/f1',) | ('train/loss_step',) |
  | --: | --------------: | ------------: | --------------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | ----------: | -------------------: |
  |   0 |         246.269 |       1197.41 |               246.269 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_19-55-25 |      0.852635 |    0.696186 |              20.7894 |
  |   1 |         211.708 |       1237.56 |               211.708 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_21-32-57 |      0.863807 |    0.697073 |                    0 |
  |   2 |         250.596 |       1194.98 |               250.596 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_23-25-51 |      0.848114 |    0.692542 |                    0 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.850374 | 0.852635 | 0.858221 |     3 | 0.863807 | 0.854852 | 0.848114 | 0.00807809 |
  | train/loss       |  228.988 |  246.269 |  248.432 |     3 |  250.596 |  236.191 |  211.708 |    21.3131 |
  | train/loss_epoch |  228.988 |  246.269 |  248.432 |     3 |  250.596 |  236.191 |  211.708 |    21.3131 |
  | train/loss_step  |        0 |        0 |  10.3947 |     3 |  20.7894 |   6.9298 |        0 |    12.0028 |
  | val/f1           | 0.694364 | 0.696186 | 0.696629 |     3 | 0.697073 | 0.695267 | 0.692542 |  0.0024011 |
  | val/loss         |   1196.2 |  1197.41 |  1217.49 |     3 |  1237.56 |  1209.98 |  1194.98 |    23.9114 |

### Coreference probing - frozen target model with mean aggregation, truncated to 8 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-coref-hoi=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/qy97f2jg
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/j6ag0bhy
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ojabhln7

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss_step',) | ('train/f1',) | ('val/loss',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | --------------: | --------------------: | ----------: | -------------------: | ------------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-56-53 |         173.272 |               173.272 |    0.714368 |                    0 |      0.898469 |       1430.46 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-43-00 |         174.206 |               174.206 |    0.713674 |              139.698 |      0.896948 |        1429.6 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_04-20-46 |         167.977 |               167.977 |    0.716134 |              162.297 |      0.899586 |        1437.1 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.897709 | 0.898469 | 0.899028 |     3 | 0.899586 | 0.898335 | 0.896948 | 0.00132437 |
  | train/loss       |  170.624 |  173.272 |  173.739 |     3 |  174.206 |  171.818 |  167.977 |    3.35929 |
  | train/loss_epoch |  170.624 |  173.272 |  173.739 |     3 |  174.206 |  171.818 |  167.977 |    3.35929 |
  | train/loss_step  |  69.8492 |  139.698 |  150.998 |     3 |  162.297 |  100.665 |        0 |    87.9076 |
  | val/f1           | 0.714021 | 0.714368 | 0.715251 |     3 | 0.716134 | 0.714725 | 0.713674 | 0.00126862 |
  | val/loss         |  1430.03 |  1430.46 |  1433.78 |     3 |   1437.1 |  1432.39 |   1429.6 |    4.10605 |

### Coreference probing - frozen target model with mean aggregation, truncated to 9 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-coref-hoi=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/a00zxa1s
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/rwgalqkj
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/nntnqot7

- metric values per seed

  |     | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('train/loss',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('val/f1',) |
  | --: | ------------: | ------------: | --------------------: | --------------: | :---------------------------------------------------------------------------------------------------------- | -------------------: | ----------: |
  |   0 |      0.940851 |       1749.45 |               97.4456 |         97.4456 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_16-00-39 |              28.3827 |    0.731681 |
  |   1 |       0.93142 |       1657.84 |               119.212 |         119.212 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_18-58-58 |              62.4193 |    0.729081 |
  |   2 |      0.915451 |       1508.24 |               156.126 |         156.126 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_21-14-11 |                    0 |    0.724934 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.923436 |  0.93142 | 0.936136 |     3 | 0.940851 | 0.929241 | 0.915451 | 0.0128395 |
  | train/loss       |  108.329 |  119.212 |  137.669 |     3 |  156.126 |  124.261 |  97.4456 |   29.6645 |
  | train/loss_epoch |  108.329 |  119.212 |  137.669 |     3 |  156.126 |  124.261 |  97.4456 |   29.6645 |
  | train/loss_step  |  14.1914 |  28.3827 |   45.401 |     3 |  62.4193 |  30.2673 |        0 |   31.2523 |
  | val/f1           | 0.727007 | 0.729081 | 0.730381 |     3 | 0.731681 | 0.728565 | 0.724934 | 0.0034029 |
  | val/loss         |  1583.04 |  1657.84 |  1703.65 |     3 |  1749.45 |  1638.51 |  1508.24 |   121.763 |

### Coreference probing - frozen target model with mean aggregation, truncated to 10 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-coref-hoi=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/yf9592ct
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/wq2pqy47
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/sx8arn5m

- metric values per seed

  |     | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('val/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) |
  | --: | ------------: | ------------: | --------------: | ----------: | --------------------: | :---------------------------------------------------------------------------------------------------------- | -------------------: |
  |   0 |       1530.94 |      0.936763 |         91.9766 |    0.731377 |               91.9766 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_22-41-58 |                    0 |
  |   1 |       1096.24 |      0.903825 |         139.337 |    0.719927 |               139.337 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-50-28 |              93.4095 |
  |   2 |       1185.13 |      0.909636 |         131.859 |    0.724025 |               131.859 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_01-40-26 |              598.417 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.906731 | 0.909636 |   0.9232 |     3 | 0.936763 | 0.916742 | 0.903825 |  0.0175811 |
  | train/loss       |  111.918 |  131.859 |  135.598 |     3 |  139.337 |  121.058 |  91.9766 |    25.4609 |
  | train/loss_epoch |  111.918 |  131.859 |  135.598 |     3 |  139.337 |  121.058 |  91.9766 |    25.4609 |
  | train/loss_step  |  46.7048 |  93.4095 |  345.913 |     3 |  598.417 |  230.609 |        0 |    321.937 |
  | val/f1           | 0.721976 | 0.724025 | 0.727701 |     3 | 0.731377 |  0.72511 | 0.719927 | 0.00580174 |
  | val/loss         |  1140.69 |  1185.13 |  1358.03 |     3 |  1530.94 |  1270.77 |  1096.24 |    229.654 |

### Coreference probing - frozen target model with mean aggregation, truncated to 11 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-coref-hoi=11 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/6y3vl3ak
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/fdgcukam
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/t9wlcag8

- metric values per seed

  |     | ('val/loss',) | ('model_save_dir',)                                                                                         | ('train/loss_epoch',) | ('val/f1',) | ('train/f1',) | ('train/loss_step',) | ('train/loss',) |
  | --: | ------------: | :---------------------------------------------------------------------------------------------------------- | --------------------: | ----------: | ------------: | -------------------: | --------------: |
  |   0 |       1054.86 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-42-59 |               95.0389 |    0.724827 |       0.91795 |          1.52588e-05 |         95.0389 |
  |   1 |       1166.77 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_04-18-53 |               70.3278 |    0.733928 |      0.934911 |                    0 |         70.3278 |
  |   2 |       1189.95 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_07-00-55 |               67.2834 |    0.732118 |      0.938262 |              136.298 |         67.2834 |

- aggregated values:

  |                  |         25% |         50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ----------: | ----------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |     0.92643 |    0.934911 | 0.936586 |     3 | 0.938262 | 0.930374 |  0.91795 |  0.0108895 |
  | train/loss       |     68.8056 |     70.3278 |  82.6834 |     3 |  95.0389 |    77.55 |  67.2834 |    15.2221 |
  | train/loss_epoch |     68.8056 |     70.3278 |  82.6834 |     3 |  95.0389 |    77.55 |  67.2834 |    15.2221 |
  | train/loss_step  | 7.62939e-06 | 1.52588e-05 |  68.1491 |     3 |  136.298 |  45.4328 |        0 |    78.6918 |
  | val/f1           |    0.728473 |    0.732118 | 0.733023 |     3 | 0.733928 | 0.730291 | 0.724827 | 0.00481765 |
  | val/loss         |     1110.82 |     1166.77 |  1178.36 |     3 |  1189.95 |  1137.19 |  1054.86 |    72.2399 |

### Coreference probing - frozen BERT model with mean aggregation, truncated to 6 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9ptx0kqm
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/mzxwjkts
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ajrzg485

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('train/loss',) |
  | --: | ------------: | ----------: | ------------: | --------------------: | :---------------------------------------------------------------------------------------------------------- | -------------------: | --------------: |
  |   0 |       836.871 |    0.612279 |      0.671173 |               469.639 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_14-49-01 |              133.837 |         469.639 |
  |   1 |       769.506 |    0.628856 |      0.686462 |               431.716 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_16-09-29 |              301.209 |         431.716 |
  |   2 |       816.964 |    0.628782 |      0.708347 |               387.207 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-43-02 |               564.24 |         387.207 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.678818 | 0.686462 | 0.697404 |     3 | 0.708347 | 0.688661 | 0.671173 |  0.0186839 |
  | train/loss       |  409.461 |  431.716 |  450.677 |     3 |  469.639 |  429.521 |  387.207 |    41.2599 |
  | train/loss_epoch |  409.461 |  431.716 |  450.677 |     3 |  469.639 |  429.521 |  387.207 |    41.2599 |
  | train/loss_step  |  217.523 |  301.209 |  432.725 |     3 |   564.24 |  333.096 |  133.837 |    216.966 |
  | val/f1           | 0.620531 | 0.628782 | 0.628819 |     3 | 0.628856 | 0.623306 | 0.612279 | 0.00954965 |
  | val/loss         |  793.235 |  816.964 |  826.917 |     3 |  836.871 |   807.78 |  769.506 |    34.6085 |

### Coreference probing - frozen BERT model with mean aggregation, truncated to 7 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=7 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/nfyb02ok
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/fvspoo8o
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/waqoi667

- metric values per seed

  |     | ('train/loss',) | ('model_save_dir',)                                                                                         | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('train/loss_step',) | ('val/f1',) |
  | --: | --------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | ------------: | --------------------: | -------------------: | ----------: |
  |   0 |         409.339 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_19-32-43 |      0.713185 |       823.663 |               409.339 |              612.312 |    0.637111 |
  |   1 |         161.534 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_21-07-39 |      0.826994 |       859.422 |               161.534 |                    0 |    0.668587 |
  |   2 |         367.803 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-54-32 |      0.731488 |       844.842 |               367.803 |              516.243 |    0.644009 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.722337 | 0.731488 | 0.779241 |     3 | 0.826994 | 0.757222 | 0.713185 | 0.0611131 |
  | train/loss       |  264.668 |  367.803 |  388.571 |     3 |  409.339 |  312.892 |  161.534 |   132.715 |
  | train/loss_epoch |  264.668 |  367.803 |  388.571 |     3 |  409.339 |  312.892 |  161.534 |   132.715 |
  | train/loss_step  |  258.121 |  516.243 |  564.277 |     3 |  612.312 |  376.185 |        0 |   329.308 |
  | val/f1           |  0.64056 | 0.644009 | 0.656298 |     3 | 0.668587 | 0.649902 | 0.637111 |  0.016545 |
  | val/loss         |  834.252 |  844.842 |  852.132 |     3 |  859.422 |  842.642 |  823.663 |   17.9809 |

### Coreference probing - frozen BERT model with mean aggregation, truncated to 8 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/x8j3s2h3
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/l69gdkfc
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/r6k640cr

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('train/loss_step',) | ('train/f1',) | ('train/loss',) | ('val/loss',) | ('model_save_dir',)                                                                                         |
  | --: | --------------------: | ----------: | -------------------: | ------------: | --------------: | ------------: | :---------------------------------------------------------------------------------------------------------- |
  |   0 |               471.945 |    0.630764 |          1.43051e-06 |      0.684436 |         471.945 |       772.338 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-45-35 |
  |   1 |                396.85 |    0.650449 |               411.66 |      0.722958 |          396.85 |       796.303 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_03-56-09 |
  |   2 |                251.28 |    0.667569 |              151.809 |      0.788861 |          251.28 |       841.885 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_05-31-51 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | --------: |
  | train/f1         | 0.703697 | 0.722958 |  0.75591 |     3 | 0.788861 | 0.732085 |    0.684436 | 0.0528071 |
  | train/loss       |  324.065 |   396.85 |  434.397 |     3 |  471.945 |  373.358 |      251.28 |   112.192 |
  | train/loss_epoch |  324.065 |   396.85 |  434.397 |     3 |  471.945 |  373.358 |      251.28 |   112.192 |
  | train/loss_step  |  75.9045 |  151.809 |  281.735 |     3 |   411.66 |  187.823 | 1.43051e-06 |    208.18 |
  | val/f1           | 0.640607 | 0.650449 | 0.659009 |     3 | 0.667569 | 0.649594 |    0.630764 | 0.0184175 |
  | val/loss         |  784.321 |  796.303 |  819.094 |     3 |  841.885 |  803.509 |     772.338 |   35.3288 |

### Coreference probing - frozen BERT model with mean aggregation, truncated to 9 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/2lxmld8g
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/dkaf85px
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1fbnttqo

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('val/f1',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | ------------: | --------------: | -------------------: | ----------: |
  |   0 |               392.857 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_16-00-27 |       839.618 |      0.733818 |         392.857 |              86.7589 |    0.655107 |
  |   1 |               354.818 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-58-13 |       863.372 |      0.754405 |         354.818 |              36.4539 |    0.662005 |
  |   2 |               253.673 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_20-15-39 |       889.547 |      0.794871 |         253.673 |              762.407 |    0.670857 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.744112 | 0.754405 | 0.774638 |     3 | 0.794871 | 0.761032 | 0.733818 |  0.0310614 |
  | train/loss       |  304.245 |  354.818 |  373.838 |     3 |  392.857 |  333.783 |  253.673 |     71.937 |
  | train/loss_epoch |  304.245 |  354.818 |  373.838 |     3 |  392.857 |  333.783 |  253.673 |     71.937 |
  | train/loss_step  |  61.6064 |  86.7589 |  424.583 |     3 |  762.407 |  295.207 |  36.4539 |    405.389 |
  | val/f1           | 0.658556 | 0.662005 | 0.666431 |     3 | 0.670857 | 0.662657 | 0.655107 | 0.00789511 |
  | val/loss         |  851.495 |  863.372 |  876.459 |     3 |  889.547 |  864.179 |  839.618 |    24.9745 |

### Coreference probing - frozen BERT model with mean aggregation, truncated to 10 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/4u8netgm
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/uojk870t
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/zxyp3nqk

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                                         | ('train/loss',) |
  | --: | ----------: | ------------: | --------------------: | -------------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |    0.679743 |       876.194 |               194.724 |              381.639 |      0.819091 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_23-36-10 |         194.724 |
  |   1 |    0.665709 |       892.401 |               297.767 |              63.6331 |      0.775371 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_03-57-48 |         297.767 |
  |   2 |    0.646602 |       826.655 |                429.01 |                    0 |      0.715262 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_06-58-14 |          429.01 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.745316 | 0.775371 | 0.797231 |     3 | 0.819091 | 0.769908 | 0.715262 | 0.0521298 |
  | train/loss       |  246.245 |  297.767 |  363.389 |     3 |   429.01 |  307.167 |  194.724 |   117.426 |
  | train/loss_epoch |  246.245 |  297.767 |  363.389 |     3 |   429.01 |  307.167 |  194.724 |   117.426 |
  | train/loss_step  |  31.8166 |  63.6331 |  222.636 |     3 |  381.639 |  148.424 |        0 |   204.461 |
  | val/f1           | 0.656155 | 0.665709 | 0.672726 |     3 | 0.679743 | 0.664018 | 0.646602 | 0.0166354 |
  | val/loss         |  851.425 |  876.194 |  884.298 |     3 |  892.401 |  865.084 |  826.655 |   34.2522 |

### Coreference probing - frozen BERT model with mean aggregation, truncated to 11 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=11 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/za4hwowe
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1s983y9l
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/kb8j2q4c

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) | ('model_save_dir',)                                                                                         | ('val/loss',) |
  | --: | -------------------: | ------------: | --------------------: | ----------: | --------------: | :---------------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |              727.761 |       0.70076 |                429.65 |    0.637412 |          429.65 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_08-49-20 |       796.293 |
  |   1 |              53.6652 |      0.699085 |               429.662 |    0.637002 |         429.662 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_10-57-32 |       786.718 |
  |   2 |              207.618 |      0.701225 |               432.384 |    0.631445 |         432.384 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_13-06-09 |       792.742 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.699922 |  0.70076 | 0.700992 |     3 | 0.701225 | 0.700356 | 0.699085 | 0.00112546 |
  | train/loss       |  429.656 |  429.662 |  431.023 |     3 |  432.384 |  430.566 |   429.65 |    1.57488 |
  | train/loss_epoch |  429.656 |  429.662 |  431.023 |     3 |  432.384 |  430.566 |   429.65 |    1.57488 |
  | train/loss_step  |  130.642 |  207.618 |   467.69 |     3 |  727.761 |  329.682 |  53.6652 |    353.236 |
  | val/f1           | 0.634223 | 0.637002 | 0.637207 |     3 | 0.637412 | 0.635286 | 0.631445 | 0.00333311 |
  | val/loss         |   789.73 |  792.742 |  794.517 |     3 |  796.293 |  791.918 |  786.718 |    4.84027 |

### Coreference probing - frozen NER model with mean aggregation, truncated to 6 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/2c68qgeh
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9c1eimq7
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ag9lhg02

- metric values per seed

  |     | ('val/f1',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) | ('train/loss',) |
  | --: | ----------: | :---------------------------------------------------------------------------------------------------------- | -------------------: | --------------------: | ------------: | ------------: | --------------: |
  |   0 |    0.634545 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_15-34-17 |              114.324 |               371.679 |       893.052 |      0.718289 |         371.679 |
  |   1 |    0.604783 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_19-01-01 |              485.769 |                632.48 |       970.326 |      0.637846 |          632.48 |
  |   2 |    0.613324 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_21-00-20 |              141.538 |               624.773 |       913.856 |      0.640067 |         624.773 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.638956 | 0.640067 | 0.679178 |     3 | 0.718289 | 0.665401 | 0.637846 | 0.0458161 |
  | train/loss       |  498.226 |  624.773 |  628.626 |     3 |   632.48 |  542.977 |  371.679 |   148.399 |
  | train/loss_epoch |  498.226 |  624.773 |  628.626 |     3 |   632.48 |  542.977 |  371.679 |   148.399 |
  | train/loss_step  |  127.931 |  141.538 |  313.653 |     3 |  485.769 |   247.21 |  114.324 |   207.045 |
  | val/f1           | 0.609053 | 0.613324 | 0.623935 |     3 | 0.634545 | 0.617551 | 0.604783 | 0.0153248 |
  | val/loss         |  903.454 |  913.856 |  942.091 |     3 |  970.326 |  925.745 |  893.052 |   39.9855 |

### Coreference probing - frozen NER model with mean aggregation, truncated to 7 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=7 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/e4za3osg
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9ahv0eey
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9gam6ypr

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('train/loss',) | ('model_save_dir',)                                                                                         | ('val/loss',) | ('val/f1',) | ('train/f1',) |
  | --: | --------------------: | -------------------: | --------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | ----------: | ------------: |
  |   0 |                726.68 |              27.0994 |          726.68 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_22-59-56 |       944.161 |    0.602646 |      0.613564 |
  |   1 |               770.409 |          0.000137329 |         770.409 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-40-24 |       934.756 |    0.599192 |      0.602281 |
  |   2 |               978.835 |              698.907 |         978.835 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-13-07 |       880.198 |    0.544555 |      0.535246 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | --------: |
  | train/f1         | 0.568764 | 0.602281 | 0.607923 |     3 | 0.613564 | 0.583697 |    0.535246 | 0.0423374 |
  | train/loss       |  748.544 |  770.409 |  874.622 |     3 |  978.835 |  825.308 |      726.68 |   134.744 |
  | train/loss_epoch |  748.544 |  770.409 |  874.622 |     3 |  978.835 |  825.308 |      726.68 |   134.744 |
  | train/loss_step  |  13.5498 |  27.0994 |  363.003 |     3 |  698.907 |  242.002 | 0.000137329 |   395.923 |
  | val/f1           | 0.571874 | 0.599192 | 0.600919 |     3 | 0.602646 | 0.582131 |    0.544555 | 0.0325874 |
  | val/loss         |  907.477 |  934.756 |  939.459 |     3 |  944.161 |  919.705 |     880.198 |   34.5358 |

### Coreference probing - frozen NER model with mean aggregation, truncated to 8 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ie8imtvm
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/etn3pz97
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/3mgv2j7r

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | ------------: | --------------: | -------------------: | ------------: | --------------------: | ----------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-58-00 |      0.638915 |         655.397 |              327.867 |         935.1 |               655.397 |    0.611711 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_05-49-53 |      0.576354 |         895.204 |             0.619636 |       942.552 |               895.204 |    0.588729 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_07-23-11 |      0.576716 |         897.114 |              309.124 |       903.592 |               897.114 |    0.594363 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.576535 | 0.576716 | 0.607815 |     3 | 0.638915 | 0.597328 | 0.576354 | 0.0360154 |
  | train/loss       |    775.3 |  895.204 |  896.159 |     3 |  897.114 |  815.905 |  655.397 |   139.007 |
  | train/loss_epoch |    775.3 |  895.204 |  896.159 |     3 |  897.114 |  815.905 |  655.397 |   139.007 |
  | train/loss_step  |  154.872 |  309.124 |  318.496 |     3 |  327.867 |  212.537 | 0.619636 |   183.765 |
  | val/f1           | 0.591546 | 0.594363 | 0.603037 |     3 | 0.611711 | 0.598268 | 0.588729 | 0.0119781 |
  | val/loss         |  919.346 |    935.1 |  938.826 |     3 |  942.552 |  927.081 |  903.592 |   20.6809 |

### Coreference probing - frozen NER model with mean aggregation, truncated to 9 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/xcffaz4n
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/xyi1jpbm
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/58jvmx77

- metric values per seed

  |     | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                                         | ('val/loss',) |
  | --: | ------------: | --------------------: | ----------: | --------------: | -------------------: | :---------------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |      0.506315 |                1357.8 |    0.565049 |          1357.8 |              529.725 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_15-59-53 |       1084.62 |
  |   1 |      0.544893 |               1096.13 |    0.585104 |         1096.13 |               188.82 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-21-07 |       1038.01 |
  |   2 |      0.482918 |               1441.24 |    0.536788 |         1441.24 |              1210.88 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_19-26-21 |       1146.13 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.494617 | 0.506315 | 0.525604 |     3 | 0.544893 | 0.511375 | 0.482918 | 0.0312959 |
  | train/loss       |  1226.96 |   1357.8 |  1399.52 |     3 |  1441.24 |  1298.39 |  1096.13 |   180.064 |
  | train/loss_epoch |  1226.96 |   1357.8 |  1399.52 |     3 |  1441.24 |  1298.39 |  1096.13 |   180.064 |
  | train/loss_step  |  359.272 |  529.725 |  870.301 |     3 |  1210.88 |  643.141 |   188.82 |   520.382 |
  | val/f1           | 0.550919 | 0.565049 | 0.575077 |     3 | 0.585104 | 0.562314 | 0.536788 | 0.0242743 |
  | val/loss         |  1061.32 |  1084.62 |  1115.38 |     3 |  1146.13 |  1089.59 |  1038.01 |   54.2299 |

### Coreference probing - frozen NER model with mean aggregation, truncated to 10 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9tar4mq8
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9zm8fqll
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ota18fv1

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss',) | ('val/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('train/f1',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | --------------: | ------------: | -------------------: | --------------------: | ----------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_20-23-29 |         1370.72 |       1108.97 |              1016.25 |               1370.72 |    0.551761 |      0.482313 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_22-01-37 |         1070.73 |       944.946 |              1892.92 |               1070.73 |    0.577874 |       0.52275 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-47-28 |         1491.97 |       1076.44 |              1544.17 |               1491.97 |     0.54296 |       0.46448 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |     min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ------: | --------: |
  | train/f1         | 0.473396 | 0.482313 | 0.502531 |     3 |  0.52275 | 0.489848 | 0.46448 | 0.0298568 |
  | train/loss       |  1220.72 |  1370.72 |  1431.34 |     3 |  1491.97 |  1311.14 | 1070.73 |   216.847 |
  | train/loss_epoch |  1220.72 |  1370.72 |  1431.34 |     3 |  1491.97 |  1311.14 | 1070.73 |   216.847 |
  | train/loss_step  |  1280.21 |  1544.17 |  1718.54 |     3 |  1892.92 |  1484.44 | 1016.25 |   441.378 |
  | val/f1           | 0.547361 | 0.551761 | 0.564817 |     3 | 0.577874 | 0.557532 | 0.54296 | 0.0181579 |
  | val/loss         |  1010.69 |  1076.44 |   1092.7 |     3 |  1108.97 |  1043.45 | 944.946 |   86.8434 |

### Coreference probing - frozen NER model with mean aggregation, truncated to 11 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=11 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/kkr3rec5
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/tr6v4sky
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/3nrbmsu6

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('train/loss',) | ('val/f1',) | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | -------------------: | --------------: | ----------: | ------------: | ------------: | --------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-07-06 |              875.183 |         1302.27 |    0.530786 |      0.434167 |       933.774 |               1302.27 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_04-45-55 |              872.996 |         1453.07 |    0.517201 |       0.41164 |       975.126 |               1453.07 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_06-33-12 |              1596.15 |         1103.55 |    0.546624 |      0.454297 |       883.739 |               1103.55 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.422904 | 0.434167 | 0.444232 |     3 | 0.454297 | 0.433368 |  0.41164 | 0.0213394 |
  | train/loss       |  1202.91 |  1302.27 |  1377.67 |     3 |  1453.07 |  1286.29 |  1103.55 |   175.307 |
  | train/loss_epoch |  1202.91 |  1302.27 |  1377.67 |     3 |  1453.07 |  1286.29 |  1103.55 |   175.307 |
  | train/loss_step  |  874.089 |  875.183 |  1235.67 |     3 |  1596.15 |  1114.78 |  872.996 |   416.882 |
  | val/f1           | 0.523994 | 0.530786 | 0.538705 |     3 | 0.546624 | 0.531537 | 0.517201 | 0.0147259 |
  | val/loss         |  908.757 |  933.774 |   954.45 |     3 |  975.126 |   930.88 |  883.739 |    45.762 |

### Coreference probing - frozen RE model with mean aggregation, truncated to 6 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/b612jctf
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/bw47wfrv
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/7l3m8h5z

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss',) | ('train/f1',) | ('val/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | --------------: | ------------: | ------------: | -------------------: | --------------------: | ----------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_15-47-55 |         559.534 |      0.666653 |       983.949 |              1368.08 |               559.534 |    0.611004 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-36-07 |         672.746 |      0.620436 |       958.134 |              452.086 |               672.746 |    0.590444 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_18-59-37 |         435.478 |      0.709183 |        1047.3 |              234.708 |               435.478 |    0.618055 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.643545 | 0.666653 | 0.687918 |     3 | 0.709183 | 0.665424 | 0.620436 |  0.044386 |
  | train/loss       |  497.506 |  559.534 |   616.14 |     3 |  672.746 |  555.919 |  435.478 |   118.675 |
  | train/loss_epoch |  497.506 |  559.534 |   616.14 |     3 |  672.746 |  555.919 |  435.478 |   118.675 |
  | train/loss_step  |  343.397 |  452.086 |  910.084 |     3 |  1368.08 |  684.959 |  234.708 |   601.504 |
  | val/f1           | 0.600724 | 0.611004 | 0.614529 |     3 | 0.618055 | 0.606501 | 0.590444 | 0.0143456 |
  | val/loss         |  971.041 |  983.949 |  1015.62 |     3 |   1047.3 |   996.46 |  958.134 |   45.8789 |

### Coreference probing - frozen RE model with mean aggregation, truncated to 7 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=7 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/0lbo2vnu
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/sf5hgyw4
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/sed32llc

- metric values per seed

  |     | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) | ('val/loss',) | ('model_save_dir',)                                                                                         |
  | --: | ------------: | -------------------: | --------------------: | ----------: | --------------: | ------------: | :---------------------------------------------------------------------------------------------------------- |
  |   0 |      0.661691 |                    0 |               606.113 |    0.610503 |         606.113 |       1014.92 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_21-46-44 |
  |   1 |      0.686114 |              737.214 |               532.157 |    0.616566 |         532.157 |       994.673 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_23-36-23 |
  |   2 |      0.671288 |              89.6875 |               566.072 |    0.617224 |         566.072 |       994.304 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_01-50-07 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.666489 | 0.671288 | 0.678701 |     3 | 0.686114 | 0.673031 | 0.661691 |  0.0123048 |
  | train/loss       |  549.114 |  566.072 |  586.093 |     3 |  606.113 |  568.114 |  532.157 |    37.0206 |
  | train/loss_epoch |  549.114 |  566.072 |  586.093 |     3 |  606.113 |  568.114 |  532.157 |    37.0206 |
  | train/loss_step  |  44.8437 |  89.6875 |  413.451 |     3 |  737.214 |  275.634 |        0 |    402.247 |
  | val/f1           | 0.613535 | 0.616566 | 0.616895 |     3 | 0.617224 | 0.614765 | 0.610503 | 0.00370488 |
  | val/loss         |  994.489 |  994.673 |   1004.8 |     3 |  1014.92 |   1001.3 |  994.304 |    11.7962 |

### Coreference probing - frozen RE model with mean aggregation, truncated to 8 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/jyl3v3av
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/neg287tq
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/z4f7rp4l

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('val/loss',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss',) |
  | --: | ------------: | :---------------------------------------------------------------------------------------------------------- | -------------------: | ------------: | ----------: | --------------------: | --------------: |
  |   0 |      0.658013 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_03-52-00 |              1280.26 |       977.398 |    0.614569 |               598.518 |         598.518 |
  |   1 |      0.641694 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_05-55-28 |              166.327 |       969.286 |    0.610376 |               637.587 |         637.587 |
  |   2 |      0.654506 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_07-45-39 |              29.6246 |       1007.79 |    0.614556 |               605.334 |         605.334 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |   0.6481 | 0.654506 |  0.65626 |     3 | 0.658013 | 0.651404 | 0.641694 | 0.00859043 |
  | train/loss       |  601.926 |  605.334 |   621.46 |     3 |  637.587 |  613.813 |  598.518 |    20.8691 |
  | train/loss_epoch |  601.926 |  605.334 |   621.46 |     3 |  637.587 |  613.813 |  598.518 |    20.8691 |
  | train/loss_step  |  97.9759 |  166.327 |  723.295 |     3 |  1280.26 |  492.071 |  29.6246 |    686.007 |
  | val/f1           | 0.612466 | 0.614556 | 0.614563 |     3 | 0.614569 | 0.613167 | 0.610376 | 0.00241727 |
  | val/loss         |  973.342 |  977.398 |  992.592 |     3 |  1007.79 |  984.823 |  969.286 |    20.2954 |

### Coreference probing - frozen RE model with mean aggregation, truncated to 9 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/i7fl4p07
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/dlfnvvts
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/op582upr

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss',) | ('model_save_dir',)                                                                                         |
  | --: | ----------: | ------------: | -------------------: | --------------------: | ------------: | --------------: | :---------------------------------------------------------------------------------------------------------- |
  |   0 |    0.635543 |      0.728237 |              166.557 |                358.52 |       955.923 |          358.52 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_15-59-29 |
  |   1 |    0.608227 |      0.633063 |              72.1614 |               665.852 |       1014.27 |         665.852 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_20-20-59 |
  |   2 |    0.628072 |      0.692193 |              655.799 |               477.047 |        988.35 |         477.047 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_22-29-12 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.662628 | 0.692193 | 0.710215 |     3 | 0.728237 | 0.684498 | 0.633063 | 0.0480514 |
  | train/loss       |  417.783 |  477.047 |   571.45 |     3 |  665.852 |  500.473 |   358.52 |       155 |
  | train/loss_epoch |  417.783 |  477.047 |   571.45 |     3 |  665.852 |  500.473 |   358.52 |       155 |
  | train/loss_step  |  119.359 |  166.557 |  411.178 |     3 |  655.799 |  298.173 |  72.1614 |   313.289 |
  | val/f1           |  0.61815 | 0.628072 | 0.631807 |     3 | 0.635543 | 0.623947 | 0.608227 | 0.0141174 |
  | val/loss         |  972.137 |   988.35 |  1001.31 |     3 |  1014.27 |  986.183 |  955.923 |   29.2362 |

### Coreference probing - frozen RE model with mean aggregation, truncated to 10 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/kchl52mx
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/4t1ahrsn
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/uykwyq6c

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                                         | ('train/loss_epoch',) | ('val/loss',) | ('train/loss',) | ('val/f1',) |
  | --: | -------------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | --------------------: | ------------: | --------------: | ----------: |
  |   0 |              542.927 |      0.605795 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_01-57-52 |               745.127 |       1051.45 |         745.127 |    0.593218 |
  |   1 |              75.2204 |      0.614023 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_04-06-46 |                707.23 |       1057.34 |          707.23 |    0.595704 |
  |   2 |              55.7645 |      0.610029 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_06-28-13 |               720.127 |       1052.63 |         720.127 |    0.596387 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.607912 | 0.610029 | 0.612026 |     3 | 0.614023 | 0.609949 | 0.605795 | 0.00411488 |
  | train/loss       |  713.678 |  720.127 |  732.627 |     3 |  745.127 |  724.161 |   707.23 |    19.2678 |
  | train/loss_epoch |  713.678 |  720.127 |  732.627 |     3 |  745.127 |  724.161 |   707.23 |    19.2678 |
  | train/loss_step  |  65.4924 |  75.2204 |  309.074 |     3 |  542.927 |  224.637 |  55.7645 |    275.818 |
  | val/f1           | 0.594461 | 0.595704 | 0.596045 |     3 | 0.596387 | 0.595103 | 0.593218 | 0.00166781 |
  | val/loss         |  1052.04 |  1052.63 |  1054.99 |     3 |  1057.34 |  1053.81 |  1051.45 |    3.11348 |

### Coreference probing - frozen RE model with mean aggregation, truncated to 11 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=11 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ywsoab61
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/axjgmzxu
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/l1atd19f

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('val/f1',) | ('train/loss',) | ('train/loss_step',) | ('val/loss',) | ('train/f1',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------------------- | ----------: | --------------: | -------------------: | ------------: | ------------: |
  |   0 |               797.974 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_08-45-43 |    0.578792 |         797.974 |              20.0955 |        1013.8 |      0.575329 |
  |   1 |                639.57 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_11-03-31 |    0.591795 |          639.57 |              137.005 |       962.198 |      0.608261 |
  |   2 |               692.812 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_14-15-51 |    0.590058 |         692.812 |              11.3951 |       1015.89 |      0.596403 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.585866 | 0.596403 | 0.602332 |     3 | 0.608261 | 0.593331 | 0.575329 |  0.0166794 |
  | train/loss       |  666.191 |  692.812 |  745.393 |     3 |  797.974 |  710.119 |   639.57 |    80.6077 |
  | train/loss_epoch |  666.191 |  692.812 |  745.393 |     3 |  797.974 |  710.119 |   639.57 |    80.6077 |
  | train/loss_step  |  15.7453 |  20.0955 |  78.5503 |     3 |  137.005 |  56.1652 |  11.3951 |    70.1444 |
  | val/f1           | 0.584425 | 0.590058 | 0.590927 |     3 | 0.591795 | 0.586882 | 0.578792 | 0.00705968 |
  | val/loss         |  987.997 |   1013.8 |  1014.84 |     3 |  1015.89 |  997.296 |  962.198 |    30.4136 |

### Coreference probing - frozen SQUAD model with mean aggregation, truncated to 6 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/0x052ur0
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/jso1pkvy
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/vq74q1g2

- metric values per seed

  |     | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('train/loss',) | ('val/f1',) | ('val/loss',) | ('train/loss_step',) |
  | --: | ------------: | --------------------: | :---------------------------------------------------------------------------------------------------------- | --------------: | ----------: | ------------: | -------------------: |
  |   0 |      0.704213 |               434.066 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_15-31-49 |         434.066 |    0.629891 |       898.015 |              301.221 |
  |   1 |       0.73457 |               363.117 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-43-53 |         363.117 |    0.640599 |       927.834 |                    0 |
  |   2 |      0.756957 |               305.966 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_20-44-46 |         305.966 |    0.643117 |       935.676 |               280.19 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.719391 |  0.73457 | 0.745763 |     3 | 0.756957 | 0.731913 | 0.704213 |  0.0264723 |
  | train/loss       |  334.542 |  363.117 |  398.591 |     3 |  434.066 |  367.716 |  305.966 |    64.1735 |
  | train/loss_epoch |  334.542 |  363.117 |  398.591 |     3 |  434.066 |  367.716 |  305.966 |    64.1735 |
  | train/loss_step  |  140.095 |   280.19 |  290.705 |     3 |  301.221 |  193.804 |        0 |    168.168 |
  | val/f1           | 0.635245 | 0.640599 | 0.641858 |     3 | 0.643117 | 0.637869 | 0.629891 | 0.00702298 |
  | val/loss         |  912.925 |  927.834 |  931.755 |     3 |  935.676 |  920.508 |  898.015 |    19.8708 |

### Coreference probing - frozen SQUAD model with mean aggregation, truncated to 7 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=7 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/3anbu5b2
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/6i58lxaw
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/izy63cl1

- metric values per seed

  |     | ('val/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                         | ('train/f1',) | ('train/loss',) |
  | --: | ------------: | -------------------: | --------------------: | ----------: | :---------------------------------------------------------------------------------------------------------- | ------------: | --------------: |
  |   0 |       926.659 |              126.844 |               488.714 |    0.634186 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-20-16 |       0.69647 |         488.714 |
  |   1 |       942.293 |                    0 |               402.416 |    0.648733 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-20-40 |       0.73147 |         402.416 |
  |   2 |       937.404 |              208.656 |               452.743 |    0.640602 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_04-56-53 |      0.712354 |         452.743 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.704412 | 0.712354 | 0.721912 |     3 |  0.73147 | 0.713431 |  0.69647 |  0.0175246 |
  | train/loss       |   427.58 |  452.743 |  470.729 |     3 |  488.714 |  447.958 |  402.416 |    43.3472 |
  | train/loss_epoch |   427.58 |  452.743 |  470.729 |     3 |  488.714 |  447.958 |  402.416 |    43.3472 |
  | train/loss_step  |  63.4218 |  126.844 |   167.75 |     3 |  208.656 |  111.833 |        0 |    105.135 |
  | val/f1           | 0.637394 | 0.640602 | 0.644668 |     3 | 0.648733 | 0.641174 | 0.634186 | 0.00729033 |
  | val/loss         |  932.031 |  937.404 |  939.848 |     3 |  942.293 |  935.452 |  926.659 |    7.99758 |

### Coreference probing - frozen SQUAD model with mean aggregation, truncated to 8 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/awv7cybe
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/lanh1kx3
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/n9hb7f2e

- metric values per seed

  |     | ('train/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('val/loss',) | ('val/f1',) |
  | --: | --------------: | --------------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | -------------------: | ------------: | ----------: |
  |   0 |         494.782 |               494.782 |      0.699244 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_07-13-39 |              1133.27 |       899.235 |    0.645045 |
  |   1 |         508.936 |               508.936 |      0.695349 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_09-12-20 |              203.974 |       901.372 |    0.642917 |
  |   2 |          365.08 |                365.08 |      0.747141 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_11-04-39 |              363.528 |       935.271 |    0.655147 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.697296 | 0.699244 | 0.723192 |     3 | 0.747141 | 0.713911 | 0.695349 |  0.0288436 |
  | train/loss       |  429.931 |  494.782 |  501.859 |     3 |  508.936 |  456.266 |   365.08 |    79.2859 |
  | train/loss_epoch |  429.931 |  494.782 |  501.859 |     3 |  508.936 |  456.266 |   365.08 |    79.2859 |
  | train/loss_step  |  283.751 |  363.528 |  748.398 |     3 |  1133.27 |  566.923 |  203.974 |    496.914 |
  | val/f1           | 0.643981 | 0.645045 | 0.650096 |     3 | 0.655147 | 0.647703 | 0.642917 | 0.00653405 |
  | val/loss         |  900.304 |  901.372 |  918.322 |     3 |  935.271 |   911.96 |  899.235 |    20.2168 |

### Coreference probing - frozen SQUAD model with mean aggregation, truncated to 9 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/iwwzze8y
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/vpxzkqj2
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/o33t4rtg

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('val/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                                         | ('train/loss_epoch',) | ('train/loss',) |
  | --: | ----------: | ------------: | ------------: | -------------------: | :---------------------------------------------------------------------------------------------------------- | --------------------: | --------------: |
  |   0 |    0.644198 |      0.683587 |       919.712 |              916.869 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_16-00-09 |               555.208 |         555.208 |
  |   1 |    0.639638 |      0.685353 |       939.453 |              213.323 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-57-49 |               558.148 |         558.148 |
  |   2 |    0.645367 |      0.703005 |       952.806 |              583.578 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_19-55-42 |               500.651 |         500.651 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |  0.68447 | 0.685353 | 0.694179 |     3 | 0.703005 | 0.690648 | 0.683587 |  0.0107375 |
  | train/loss       |   527.93 |  555.208 |  556.678 |     3 |  558.148 |  538.002 |  500.651 |    32.3802 |
  | train/loss_epoch |   527.93 |  555.208 |  556.678 |     3 |  558.148 |  538.002 |  500.651 |    32.3802 |
  | train/loss_step  |   398.45 |  583.578 |  750.223 |     3 |  916.869 |  571.256 |  213.323 |    351.935 |
  | val/f1           | 0.641918 | 0.644198 | 0.644783 |     3 | 0.645367 | 0.643068 | 0.639638 | 0.00302741 |
  | val/loss         |  929.583 |  939.453 |  946.129 |     3 |  952.806 |  937.324 |  919.712 |    16.6493 |

### Coreference probing - frozen SQUAD model with mean aggregation, truncated to 10 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/o7w6k9vw
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/js15velo
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/rz6s1y2y

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('train/f1',) | ('train/loss',) | ('val/f1',) | ('val/loss',) | ('train/loss_step',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | --------------: | ----------: | ------------: | -------------------: |
  |   0 |               413.205 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_22-14-07 |      0.727413 |         413.205 |    0.659748 |       918.834 |              145.506 |
  |   1 |               628.327 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_01-47-26 |      0.661107 |         628.327 |    0.636203 |       932.179 |              477.728 |
  |   2 |               656.033 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_03-46-52 |      0.655001 |         656.033 |    0.629741 |       955.188 |                    0 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.658054 | 0.661107 |  0.69426 |     3 | 0.727413 | 0.681174 | 0.655001 | 0.0401605 |
  | train/loss       |  520.766 |  628.327 |   642.18 |     3 |  656.033 |  565.855 |  413.205 |   132.923 |
  | train/loss_epoch |  520.766 |  628.327 |   642.18 |     3 |  656.033 |  565.855 |  413.205 |   132.923 |
  | train/loss_step  |  72.7528 |  145.506 |  311.617 |     3 |  477.728 |  207.745 |        0 |    244.87 |
  | val/f1           | 0.632972 | 0.636203 | 0.647976 |     3 | 0.659748 | 0.641897 | 0.629741 | 0.0157933 |
  | val/loss         |  925.506 |  932.179 |  943.683 |     3 |  955.188 |    935.4 |  918.834 |   18.3896 |

### Coreference probing - frozen SQUAD model with mean aggregation, truncated to 11 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=11 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9r55f9pg
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1nwiozzy
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/21v3ovp3

- metric values per seed

  |     | ('val/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                         | ('train/loss',) |
  | --: | ------------: | --------------------: | ------------: | ----------: | -------------------: | :---------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |       903.312 |               574.104 |       0.66321 |    0.642944 |              533.465 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_05-38-17 |         574.104 |
  |   1 |       978.989 |               897.469 |      0.586732 |     0.60169 |              730.321 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_09-09-18 |         897.469 |
  |   2 |       912.968 |               584.201 |      0.656636 |    0.640091 |              188.365 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_10-49-03 |         584.201 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.621684 | 0.656636 | 0.659923 |     3 |  0.66321 | 0.635526 | 0.586732 | 0.0423845 |
  | train/loss       |  579.152 |  584.201 |  740.835 |     3 |  897.469 |  685.258 |  574.104 |   183.849 |
  | train/loss_epoch |  579.152 |  584.201 |  740.835 |     3 |  897.469 |  685.258 |  574.104 |   183.849 |
  | train/loss_step  |  360.915 |  533.465 |  631.893 |     3 |  730.321 |   484.05 |  188.365 |   274.336 |
  | val/f1           |  0.62089 | 0.640091 | 0.641517 |     3 | 0.642944 | 0.628241 |  0.60169 | 0.0230385 |
  | val/loss         |   908.14 |  912.968 |  945.978 |     3 |  978.989 |  931.756 |  903.312 |   41.1888 |

### Coreference probing - frozen MRPC model with mean aggregation, truncated to 6 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/r3oc1mbd
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/81p10ovn
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1rx8z3k0

- metric values per seed

  |     | ('train/loss_step',) | ('model_save_dir',)                                                                                         | ('val/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) |
  | --: | -------------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | ------------: | --------------------: | ----------: | --------------: |
  |   0 |              116.197 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_15-30-16 |       867.764 |       0.68277 |               498.585 |    0.628923 |         498.585 |
  |   1 |              496.674 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_17-12-25 |       862.545 |      0.686414 |               479.646 |    0.627496 |         479.646 |
  |   2 |              813.102 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_18-58-16 |       889.539 |      0.692825 |                472.94 |     0.62753 |          472.94 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |         std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ----------: |
  | train/f1         | 0.684592 | 0.686414 |  0.68962 |     3 | 0.692825 | 0.687336 |  0.68277 |  0.00509066 |
  | train/loss       |  476.293 |  479.646 |  489.115 |     3 |  498.585 |  483.723 |   472.94 |     13.2999 |
  | train/loss_epoch |  476.293 |  479.646 |  489.115 |     3 |  498.585 |  483.723 |   472.94 |     13.2999 |
  | train/loss_step  |  306.436 |  496.674 |  654.888 |     3 |  813.102 |  475.325 |  116.197 |     348.943 |
  | val/f1           | 0.627513 |  0.62753 | 0.628227 |     3 | 0.628923 | 0.627983 | 0.627496 | 0.000814514 |
  | val/loss         |  865.155 |  867.764 |  878.651 |     3 |  889.539 |  873.283 |  862.545 |     14.3182 |

### Coreference probing - frozen MRPC model with mean aggregation, truncated to 7 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=7 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/uh7rr9go
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1k5y9hu3
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/vuoo0q4b

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) |
  | --: | ------------: | ----------: | ------------: | --------------------: | --------------: | :---------------------------------------------------------------------------------------------------------- | -------------------: |
  |   0 |           890 |    0.667832 |      0.809154 |               199.485 |         199.485 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_20-48-05 |              32.5551 |
  |   1 |       874.348 |    0.649547 |      0.723643 |               410.532 |         410.532 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-41-20 |              601.259 |
  |   2 |        894.28 |    0.647704 |      0.721041 |                 428.8 |           428.8 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_02-50-46 |              283.849 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.722342 | 0.723643 | 0.766398 |     3 | 0.809154 | 0.751279 | 0.721041 | 0.0501375 |
  | train/loss       |  305.009 |  410.532 |  419.666 |     3 |    428.8 |  346.272 |  199.485 |   127.449 |
  | train/loss_epoch |  305.009 |  410.532 |  419.666 |     3 |    428.8 |  346.272 |  199.485 |   127.449 |
  | train/loss_step  |  158.202 |  283.849 |  442.554 |     3 |  601.259 |  305.888 |  32.5551 |   284.992 |
  | val/f1           | 0.648626 | 0.649547 | 0.658689 |     3 | 0.667832 | 0.655028 | 0.647704 |  0.011127 |
  | val/loss         |  882.174 |      890 |   892.14 |     3 |   894.28 |  886.209 |  874.348 |   10.4929 |

### Coreference probing - frozen MRPC model with mean aggregation, truncated to 8 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/127wtvbd
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/0h67gq24
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/koipa2tc

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('val/f1',) | ('train/loss_epoch',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | -------------------: | ------------: | ------------: | --------------: | ----------: | --------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_04-53-09 |              56.3238 |       853.644 |      0.717055 |         440.188 |    0.654665 |               440.188 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_06-52-54 |             0.174842 |       869.774 |      0.760105 |         322.912 |     0.66663 |               322.912 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_09-36-06 |              968.995 |       864.378 |      0.743792 |         366.182 |    0.662723 |               366.182 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.730424 | 0.743792 | 0.751949 |     3 | 0.760105 | 0.740317 | 0.717055 |  0.0217342 |
  | train/loss       |  344.547 |  366.182 |  403.185 |     3 |  440.188 |  376.427 |  322.912 |    59.3054 |
  | train/loss_epoch |  344.547 |  366.182 |  403.185 |     3 |  440.188 |  376.427 |  322.912 |    59.3054 |
  | train/loss_step  |  28.2493 |  56.3238 |  512.659 |     3 |  968.995 |  341.831 | 0.174842 |    543.865 |
  | val/f1           | 0.658694 | 0.662723 | 0.664676 |     3 |  0.66663 | 0.661339 | 0.654665 | 0.00610112 |
  | val/loss         |  859.011 |  864.378 |  867.076 |     3 |  869.774 |  862.599 |  853.644 |    8.21105 |

### Coreference probing - frozen MRPC model with mean aggregation, truncated to 9 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ufub0ile
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/8s2ny128
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/q3tsgh1f

- metric values per seed

  |     | ('val/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('val/loss',) | ('model_save_dir',)                                                                                         | ('train/loss',) | ('train/f1',) |
  | --: | ----------: | --------------------: | -------------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | --------------: | ------------: |
  |   0 |    0.661864 |               451.918 |               48.167 |       890.111 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_16-00-20 |         451.918 |      0.715234 |
  |   1 |    0.658865 |               451.639 |              564.664 |       901.293 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_18-43-20 |         451.639 |       0.71357 |
  |   2 |    0.666933 |               379.875 |              142.865 |       870.047 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-01_21-26-39 |         379.875 |      0.736767 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.714402 | 0.715234 |    0.726 |     3 | 0.736767 | 0.721857 |  0.71357 |  0.0129395 |
  | train/loss       |  415.757 |  451.639 |  451.779 |     3 |  451.918 |  427.811 |  379.875 |    41.5141 |
  | train/loss_epoch |  415.757 |  451.639 |  451.779 |     3 |  451.918 |  427.811 |  379.875 |    41.5141 |
  | train/loss_step  |  95.5158 |  142.865 |  353.765 |     3 |  564.664 |  251.899 |   48.167 |     274.97 |
  | val/f1           | 0.660364 | 0.661864 | 0.664399 |     3 | 0.666933 | 0.662554 | 0.658865 | 0.00407821 |
  | val/loss         |  880.079 |  890.111 |  895.702 |     3 |  901.293 |   887.15 |  870.047 |    15.8321 |

### Coreference probing - frozen MRPC model with mean aggregation, truncated to 10 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/fwuiueoy
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/fcba8su0
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/6nxzwmuc

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('model_save_dir',)                                                                                         | ('val/loss',) | ('train/loss_step',) |
  | --: | --------------------: | ----------: | --------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | -------------------: |
  |   0 |               524.584 |    0.648986 |         524.584 |      0.679042 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_00-41-45 |       863.224 |              14.5433 |
  |   1 |               570.316 |    0.641829 |         570.316 |      0.662614 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_03-18-41 |       822.951 |              730.417 |
  |   2 |               335.781 |    0.666484 |         335.781 |      0.738664 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_05-31-47 |       786.005 |              481.218 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.670828 | 0.679042 | 0.708853 |     3 | 0.738664 |  0.69344 | 0.662614 | 0.0400171 |
  | train/loss       |  430.183 |  524.584 |   547.45 |     3 |  570.316 |  476.894 |  335.781 |   124.328 |
  | train/loss_epoch |  430.183 |  524.584 |   547.45 |     3 |  570.316 |  476.894 |  335.781 |   124.328 |
  | train/loss_step  |  247.881 |  481.218 |  605.818 |     3 |  730.417 |  408.726 |  14.5433 |   363.401 |
  | val/f1           | 0.645408 | 0.648986 | 0.657735 |     3 | 0.666484 | 0.652433 | 0.641829 | 0.0126838 |
  | val/loss         |  804.478 |  822.951 |  843.087 |     3 |  863.224 |   824.06 |  786.005 |   38.6211 |

### Coreference probing - frozen MRPC model with mean aggregation, truncated to 11 layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=11 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/puu36p3d
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/nynq8uyi
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/v24hp5cu

- metric values per seed

  |     | ('train/f1',) | ('train/loss',) | ('train/loss_epoch',) | ('val/loss',) | ('model_save_dir',)                                                                                         | ('val/f1',) | ('train/loss_step',) |
  | --: | ------------: | --------------: | --------------------: | ------------: | :---------------------------------------------------------------------------------------------------------- | ----------: | -------------------: |
  |   0 |      0.620668 |         681.775 |               681.775 |       814.411 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_09-41-40 |    0.628801 |              676.722 |
  |   1 |      0.656053 |         539.121 |               539.121 |       788.763 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_11-47-28 |    0.645336 |              256.086 |
  |   2 |      0.635499 |         622.338 |               622.338 |       789.562 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-02_14-45-28 |    0.638293 |              765.615 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.628083 | 0.635499 | 0.645776 |     3 | 0.656053 | 0.637407 | 0.620668 | 0.0177695 |
  | train/loss       |   580.73 |  622.338 |  652.057 |     3 |  681.775 |  614.411 |  539.121 |   71.6567 |
  | train/loss_epoch |   580.73 |  622.338 |  652.057 |     3 |  681.775 |  614.411 |  539.121 |   71.6567 |
  | train/loss_step  |  466.404 |  676.722 |  721.169 |     3 |  765.615 |  566.141 |  256.086 |   272.169 |
  | val/f1           | 0.633547 | 0.638293 | 0.641814 |     3 | 0.645336 | 0.637476 | 0.628801 | 0.0082974 |
  | val/loss         |  789.163 |  789.562 |  801.986 |     3 |  814.411 |  797.579 |  788.763 |   14.5823 |

## 2023-12-05

### Learning rate optimization for Q, K, V ablations - frozen target model and frozen MRPC with attention aggregation, with all Q, K, V projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=3e-6,3e-5,3e-4,1e-3,3e-3 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different learning rates:

  - lr 3e-6: 2023-12-03_10-46-19

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/r0xj4qnr
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/qk5v6cjl
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zed5lc8y

  - lr 3e-5:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rb6w9o5j
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ow56b0za
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cc2k60e3

  - lr 3e-4:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ksinrhz5
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hgwgp68g
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/acr2hqy4

  - lr 1e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/konrickd
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/etdn1ocy
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/eevtpox3

  - lr 3e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ipaqud23
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hic1rt44
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hnff9nuf

- metric values per learning rate and per seed

  |     | ('model_save_dir',)                                                                                          | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('train/loss_step',) |
  | --: | :----------------------------------------------------------------------------------------------------------- | ------------: | ----------: | --------------: | --------------------: | ------------: | -------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_10-46-23 |       90.3917 |    0.736685 |         5.49784 |               5.49784 |      0.950838 |          4.76837e-07 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_12-58-16 |       95.7792 |    0.741728 |         4.67594 |               4.67594 |      0.956538 |             0.650753 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_15-35-43 |       91.6284 |    0.737483 |         5.21134 |               5.21134 |      0.954076 |              7.66149 |
  |   3 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_17-59-11 |       98.9764 |    0.734576 |         6.66846 |               6.66846 |      0.944463 |           0.00424778 |
  |   4 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_19-19-58 |       106.576 |    0.736453 |         5.99738 |               5.99738 |      0.948242 |              58.5348 |
  |   5 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_21-01-52 |       106.917 |    0.737454 |         6.11588 |               6.11588 |      0.948053 |          0.000526222 |
  |   6 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_22-39-35 |       194.913 |    0.726582 |         17.8957 |               17.8957 |      0.917471 |             0.130908 |
  |   7 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_00-12-51 |       129.725 |    0.720768 |         17.6404 |               17.6404 |      0.904706 |             0.030089 |
  |   8 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_01-12-30 |       91.5339 |     0.72336 |         16.9795 |               16.9795 |      0.897415 |          4.05311e-06 |
  |   9 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_01-55-21 |       76.1384 |    0.708573 |         36.5994 |               36.5994 |      0.853806 |              15.2539 |
  |  10 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_02-25-40 |       95.0682 |    0.705751 |         45.9747 |               45.9747 |      0.852222 |              22.1443 |
  |  11 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_03-00-06 |       76.4159 |    0.709228 |         36.7929 |               36.7929 |      0.853613 |              30.3364 |
  |  12 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_03-30-22 |       4384.84 |    0.712273 |         635.893 |               635.893 |      0.877878 |              2751.07 |
  |  13 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_05-11-44 |       76.9546 |    0.693868 |         156.016 |               156.016 |       0.81556 |              319.922 |
  |  14 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_05-37-40 |       77.9449 |    0.694581 |         156.275 |               156.275 |      0.815388 |              64.7312 |

- aggregated values:

  |                  |       25% |      50% |      75% | count |      max |     mean |         min |       std |
  | :--------------- | --------: | -------: | -------: | ----: | -------: | -------: | ----------: | --------: |
  | train/f1         |  0.853709 | 0.904706 | 0.948148 |    15 | 0.956538 | 0.899351 |    0.815388 | 0.0514041 |
  | train/loss       |   6.05663 |  17.6404 |  41.3838 |    15 |  635.893 |   76.949 |     4.67594 |   162.619 |
  | train/loss_epoch |   6.05663 |  17.6404 |  41.3838 |    15 |  635.893 |   76.949 |     4.67594 |   162.619 |
  | train/loss_step  | 0.0171684 |  7.66149 |  44.4356 |    15 |  2751.07 |  218.031 | 4.76837e-07 |   705.438 |
  | val/f1           |    0.7089 |  0.72336 | 0.736569 |    15 | 0.741728 | 0.721291 |    0.693868 | 0.0162844 |
  | val/loss         |   84.1683 |  95.0682 |  106.746 |    15 |  4384.84 |  386.254 |     76.1384 |   1106.58 |

### Learning rate optimization for Q, K, V ablations - frozen target model and frozen MRPC with attention aggregation, w/o Q, K, V projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.project_target_query=False \
  +model.aggregate.project_target_key=False \
  +model.aggregate.project_target_value=False \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=3e-6,3e-5,3e-4,1e-3,3e-3 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different learning rates:

  - lr 3e-6:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pcllgn6v
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/2jnhnqyu
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/shis4gm0

  - lr 3e-5:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/kz4kq2s0
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ny3qqz01
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pd26wkr7

  - lr 3e-4:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/u8dchp4a
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/iz550ihb
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/do608e6r

  - lr 1e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/c6qb80io
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/890dnpqh
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/f9lao54q

  - lr 3e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/v8cszlp4
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/90amame2
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/z964oqqk

- metric values per learning rate and per seed

  |     | ('train/loss',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) |
  | --: | --------------: | ----------: | -------------------: | :----------------------------------------------------------------------------------------------------------- | --------------------: | ------------: | ------------: |
  |   0 |          5.7035 |    0.740362 |              1.77167 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_10-49-02 |                5.7035 |        137.51 |      0.955489 |
  |   1 |          7.1795 |    0.736991 |              28.9452 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_13-23-31 |                7.1795 |       118.876 |      0.942228 |
  |   2 |         6.48877 |    0.738909 |              8.89461 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_14-48-23 |               6.48877 |       131.484 |      0.950562 |
  |   3 |         6.14006 |    0.739948 |          1.90735e-06 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_16-54-59 |               6.14006 |       133.769 |      0.951041 |
  |   4 |         7.41002 |    0.737017 |              40.7763 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_19-01-39 |               7.41002 |       119.286 |       0.94268 |
  |   5 |         5.53306 |    0.741386 |              7.52306 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_20-26-34 |               5.53306 |       141.253 |      0.956938 |
  |   6 |         5.73979 |    0.742185 |              10.2745 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_23-15-24 |               5.73979 |       142.788 |      0.954993 |
  |   7 |         7.27489 |    0.734396 |              26.4146 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_01-41-24 |               7.27489 |        128.55 |      0.944103 |
  |   8 |         5.90571 |    0.741789 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_03-15-45 |               5.90571 |       149.427 |      0.954266 |
  |   9 |         8.37675 |    0.734229 |              5.57069 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_05-46-15 |               8.37675 |       111.024 |      0.935597 |
  |  10 |         7.42409 |     0.73579 |              31.2659 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_06-52-35 |               7.42409 |       131.746 |      0.942922 |
  |  11 |         8.20839 |     0.73387 |              12.8788 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_08-27-09 |               8.20839 |       116.403 |      0.937965 |
  |  12 |         7.68432 |    0.734402 |              9.94356 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_09-43-21 |               7.68432 |       123.781 |      0.941003 |
  |  13 |         7.74515 |    0.735413 |              30.4327 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_11-10-43 |               7.74515 |       124.891 |      0.940979 |
  |  14 |         6.21635 |    0.741021 |              5.66886 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_12-38-10 |               6.21635 |         151.1 |      0.953723 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.941616 | 0.944103 | 0.953994 |    15 | 0.956938 | 0.946966 | 0.935597 | 0.00713158 |
  | train/loss       |  6.02288 |   7.1795 |  7.55421 |    15 |  8.37675 |  6.86869 |  5.53306 |   0.956545 |
  | train/loss_epoch |  6.02288 |   7.1795 |  7.55421 |    15 |  8.37675 |  6.86869 |  5.53306 |   0.956545 |
  | train/loss_step  |  5.61977 |  9.94356 |  27.6799 |    15 |  40.7763 |  14.6907 |        0 |    13.1937 |
  | val/f1           | 0.734907 | 0.737017 | 0.740691 |    15 | 0.742185 | 0.737847 |  0.73387 |  0.0030832 |
  | val/loss         |  121.533 |  131.484 |  139.381 |    15 |    151.1 |  130.793 |  111.024 |    12.0404 |

### Learning rate optimization for Q, K, V ablations - frozen target model and frozen MRPC with attention aggregation, no Q projection, only K, V projections

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.project_target_query=False \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=3e-6,3e-5,3e-4,1e-3,3e-3 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different learning rates:

  - lr 3e-6:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cwsae0kp
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/xacd2gmg
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rqpc4f8u

  - lr 3e-5:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/gs687obo
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/9ydv1oaa
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ihqgwk1t

  - lr 3e-4:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/djfeq8e9
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/o5yu10h2
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0t7mp0yi

  - lr 1e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/q2ipxife
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cusrxoom
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/eze6j8gr

  - lr 3e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/q64v6yd5
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/0odxsz0f
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/cg94ryrk

- metric values per learning rate and per seed

  |     | ('train/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                                          | ('val/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('train/f1',) |
  | --: | --------------: | -------------------: | :----------------------------------------------------------------------------------------------------------- | ----------: | ------------: | --------------------: | ------------: |
  |   0 |         6.36651 |              17.7831 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_10-48-18 |    0.736583 |       88.6615 |               6.36651 |      0.944462 |
  |   1 |         5.31569 |              1.38434 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_12-55-44 |    0.739504 |       92.2307 |               5.31569 |      0.951661 |
  |   2 |         4.53131 |          0.000194411 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_15-28-02 |    0.740375 |       101.824 |               4.53131 |      0.959168 |
  |   3 |         7.14362 |           0.00153387 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_18-58-17 |    0.734707 |       89.6431 |               7.14362 |      0.937913 |
  |   4 |         5.86495 |              39.0168 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_20-05-59 |    0.734961 |       107.984 |               5.86495 |      0.949245 |
  |   5 |         6.49318 |              38.7097 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_21-51-49 |    0.735151 |       102.661 |               6.49318 |       0.94752 |
  |   6 |         15.4455 |              2.52634 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_23-20-41 |    0.729626 |       237.402 |               15.4455 |      0.930484 |
  |   7 |         18.8657 |              15.8015 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_01-38-40 |    0.719206 |       117.828 |               18.8657 |      0.899168 |
  |   8 |         15.7794 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_02-34-00 |    0.719877 |       85.1609 |               15.7794 |       0.89622 |
  |   9 |         37.3601 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_03-12-30 |    0.708218 |       77.9059 |               37.3601 |      0.856213 |
  |  10 |         58.7342 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_03-42-26 |    0.704407 |       115.961 |               58.7342 |      0.851302 |
  |  11 |         36.9539 |              65.5665 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_04-20-55 |    0.707446 |       78.6293 |               36.9539 |       0.85426 |
  |  12 |          590.32 |              440.527 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_04-51-04 |    0.696489 |       708.087 |                590.32 |      0.832636 |
  |  13 |         150.793 |              46.5254 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_05-36-40 |    0.693673 |       71.6562 |               150.793 |      0.816337 |
  |  14 |         625.387 |              376.447 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_06-02-35 |    0.695038 |       928.034 |               625.387 |      0.839005 |

- aggregated values:

  |                  |         25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | ----------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         |    0.852781 | 0.899168 | 0.945991 |    15 | 0.959168 | 0.897706 | 0.816337 | 0.0512496 |
  | train/loss       |     6.42985 |  15.7794 |  48.0472 |    15 |  625.387 |   105.69 |  4.53131 |   207.371 |
  | train/loss_epoch |     6.42985 |  15.7794 |  48.0472 |    15 |  625.387 |   105.69 |  4.53131 |   207.371 |
  | train/loss_step  | 0.000864139 |  15.8015 |  42.7711 |    15 |  440.527 |  69.6193 |        0 |   139.676 |
  | val/f1           |    0.705926 | 0.719877 | 0.735056 |    15 | 0.740375 | 0.719684 | 0.693673 |  0.017404 |
  | val/loss         |     86.9112 |  101.824 |  116.894 |    15 |  928.034 |  200.245 |  71.6562 |   257.247 |

### Learning rate optimization for Q, K, V ablations - frozen target model and frozen MRPC with attention aggregation, no K, V projections, only Q projection

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.project_target_key=False \
  +model.aggregate.project_target_value=False \
  model.task_learning_rate=1e-4 \
  model.bert_learning_rate=3e-6,3e-5,3e-4,1e-3,3e-3 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different learning rates:

  - lr 3e-6:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rbng65mx
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/vwkqtej3
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ie46lbtl

  - lr 3e-5:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/e7p195eq
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/gny0pdxy
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/rpqi9xkd

  - lr 3e-4:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/m2vg6kz7
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/n67houss
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/gs3l8am1

  - lr 1e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/18prz2b5
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/60xnay0w
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/hvz60ifh

  - lr 3e-3:

    - seed1: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/df4qtv3c
    - seed2: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8xq3pme1
    - seed3: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/y9rqr77n

- metric values per learning rate and per seed

  |     | ('val/f1',) | ('train/loss_step',) | ('train/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                          | ('train/f1',) | ('val/loss',) |
  | --: | ----------: | -------------------: | --------------: | --------------------: | :----------------------------------------------------------------------------------------------------------- | ------------: | ------------: |
  |   0 |    0.738854 |              43.7351 |         6.49285 |               6.49285 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_10-49-28 |      0.949846 |       124.034 |
  |   1 |    0.740581 |          1.78814e-06 |         6.12598 |               6.12598 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_12-57-16 |      0.950768 |       129.961 |
  |   2 |    0.739759 |                    0 |          6.7247 |                6.7247 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_15-10-10 |      0.948267 |       124.199 |
  |   3 |    0.739854 |              20.2515 |          5.9808 |                5.9808 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_17-08-18 |      0.952259 |       128.902 |
  |   4 |    0.739994 |          4.76837e-07 |          5.9807 |                5.9807 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_19-16-31 |      0.952044 |       130.434 |
  |   5 |    0.739649 |                    0 |         6.50989 |               6.50989 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_21-31-09 |      0.948699 |       126.526 |
  |   6 |    0.736821 |           0.00019454 |         7.42486 |               7.42486 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-03_23-30-54 |      0.944438 |       126.147 |
  |   7 |    0.739587 |             0.175386 |         6.28247 |               6.28247 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_01-05-58 |      0.953391 |        147.04 |
  |   8 |    0.738718 |              17.4024 |         7.37775 |               7.37775 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_03-45-28 |      0.945674 |       132.453 |
  |   9 |     0.73627 |              54.9368 |         6.86909 |               6.86909 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_05-40-35 |      0.949295 |       135.219 |
  |  10 |    0.739723 |             0.280123 |         7.46497 |               7.46497 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_07-49-40 |      0.945078 |       129.934 |
  |  11 |    0.738585 |           0.00566232 |         6.55533 |               6.55533 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_09-34-19 |      0.950455 |       139.888 |
  |  12 |     0.73702 |              22.3402 |         6.85007 |               6.85007 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_11-48-29 |      0.946905 |        140.46 |
  |  13 |    0.737917 |             0.220479 |         29459.3 |               29459.3 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_14-03-05 |      0.912687 |        130.66 |
  |  14 |    0.740333 |          5.96046e-07 |         6.91523 |               6.91523 | /netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-12-04_15-49-06 |      0.948573 |       137.411 |

- aggregated values:

  |                  |         25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | ----------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         |    0.946289 | 0.948699 | 0.950612 |    15 | 0.953391 | 0.946559 | 0.912687 | 0.0097355 |
  | train/loss       |     6.38766 |   6.7247 |  7.14649 |    15 |  29459.3 |  1970.19 |   5.9807 |   7604.63 |
  | train/loss_epoch |     6.38766 |   6.7247 |  7.14649 |    15 |  29459.3 |  1970.19 |   5.9807 |   7604.63 |
  | train/loss_step  | 1.19209e-06 | 0.175386 |  18.8269 |    15 |  54.9368 |  10.6232 |        0 |    17.829 |
  | val/f1           |    0.738251 | 0.739587 | 0.739806 |    15 | 0.740581 | 0.738911 |  0.73627 | 0.0013445 |
  | val/loss         |     127.714 |  130.434 |  136.315 |    15 |   147.04 |  132.218 |  124.034 |   6.60057 |

## 2023-12-11

### Coreference probing - frozen MRPC (truncated to 9 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/may7txjt
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/36p1njgk
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/d36nzjjl

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss_epoch',) | ('train/loss_step',) | ('train/loss',) | ('val/f1',) | ('val/loss',) | ('train/f1',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | --------------------: | -------------------: | --------------: | ----------: | ------------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_19-44-06 |               109.027 |              209.878 |         109.027 |    0.699649 |       360.653 |      0.811454 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_22-25-54 |               111.116 |              41.0018 |         111.116 |     0.69829 |       350.419 |      0.809004 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_01-03-14 |               88.0873 |               64.531 |         88.0873 |    0.703045 |       371.252 |      0.836925 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.810229 | 0.811454 | 0.824189 |     3 | 0.836925 | 0.819127 | 0.809004 |  0.0154614 |
  | train/loss       |  98.5574 |  109.027 |  110.072 |     3 |  111.116 |  102.744 |  88.0873 |    12.7357 |
  | train/loss_epoch |  98.5574 |  109.027 |  110.072 |     3 |  111.116 |  102.744 |  88.0873 |    12.7357 |
  | train/loss_step  |  52.7664 |   64.531 |  137.204 |     3 |  209.878 |  105.137 |  41.0018 |    91.4681 |
  | val/f1           | 0.698969 | 0.699649 | 0.701347 |     3 | 0.703045 | 0.700328 |  0.69829 | 0.00244923 |
  | val/loss         |  355.536 |  360.653 |  365.953 |     3 |  371.252 |  360.775 |  350.419 |    10.4169 |

### Coreference probing - frozen NER (truncated to 6 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1v3bktlo
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/4zgzbz8v
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/em4av5sl

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                         | ('train/loss',) |
  | --: | --------------------: | -------------------: | ------------: | ------------: | ----------: | :---------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |               140.423 |                    0 |      0.757351 |       335.004 |    0.678338 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_18-26-59 |         140.423 |
  |   1 |               94.1703 |                    0 |      0.812677 |       361.646 |    0.690821 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_20-16-41 |         94.1703 |
  |   2 |                103.19 |              29.6042 |      0.805458 |       342.687 |    0.691129 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_23-32-22 |          103.19 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.781404 | 0.805458 | 0.809068 |     3 | 0.812677 | 0.791829 | 0.757351 | 0.0300761 |
  | train/loss       |  98.6799 |   103.19 |  121.806 |     3 |  140.423 |  112.594 |  94.1703 |   24.5186 |
  | train/loss_epoch |  98.6799 |   103.19 |  121.806 |     3 |  140.423 |  112.594 |  94.1703 |   24.5186 |
  | train/loss_step  |        0 |        0 |  14.8021 |     3 |  29.6042 |  9.86806 |        0 |    17.092 |
  | val/f1           | 0.684579 | 0.690821 | 0.690975 |     3 | 0.691129 | 0.686762 | 0.678338 | 0.0072975 |
  | val/loss         |  338.846 |  342.687 |  352.166 |     3 |  361.646 |  346.446 |  335.004 |   13.7125 |

### Coreference probing - frozen RE (truncated to 9 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/jsujd86f
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/yjnq755n
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/ox8b311q

- metric values per seed

  |     | ('val/f1',) | ('train/loss_step',) | ('val/loss',) | ('train/loss',) | ('model_save_dir',)                                                                                         | ('train/f1',) | ('train/loss_epoch',) |
  | --: | ----------: | -------------------: | ------------: | --------------: | :---------------------------------------------------------------------------------------------------------- | ------------: | --------------------: |
  |   0 |    0.697567 |              1422.95 |       374.804 |         60.6188 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_20-52-49 |      0.859568 |               60.6188 |
  |   1 |    0.685378 |               167.42 |        383.52 |         124.214 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_02-27-03 |      0.789453 |               124.214 |
  |   2 |    0.687441 |              185.191 |       378.055 |         110.743 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_05-25-58 |      0.801862 |               110.743 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.795657 | 0.801862 | 0.830715 |     3 | 0.859568 | 0.816961 | 0.789453 |  0.0374169 |
  | train/loss       |  85.6808 |  110.743 |  117.478 |     3 |  124.214 |   98.525 |  60.6188 |    33.5116 |
  | train/loss_epoch |  85.6808 |  110.743 |  117.478 |     3 |  124.214 |   98.525 |  60.6188 |    33.5116 |
  | train/loss_step  |  176.306 |  185.191 |   804.07 |     3 |  1422.95 |  591.853 |   167.42 |    719.804 |
  | val/f1           | 0.686409 | 0.687441 | 0.692504 |     3 | 0.697567 | 0.690128 | 0.685378 | 0.00652371 |
  | val/loss         |  376.429 |  378.055 |  380.787 |     3 |   383.52 |  378.793 |  374.804 |    4.40446 |

### Coreference probing - frozen SQUAD (truncated to 8 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/egfyui74
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/91skanyr
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/2s31svxd

- metric values per seed

  |     | ('val/loss',) | ('train/loss',) | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                         |
  | --: | ------------: | --------------: | ------------: | -------------------: | --------------------: | ----------: | :---------------------------------------------------------------------------------------------------------- |
  |   0 |       365.864 |         86.0916 |      0.836793 |              62.9427 |               86.0916 |     0.69776 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_19-17-38 |
  |   1 |       363.192 |          103.11 |      0.817646 |                    0 |                103.11 |    0.694601 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_22-18-07 |
  |   2 |       361.929 |         98.6749 |      0.822278 |              2.96654 |               98.6749 |    0.693676 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_00-46-04 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.819962 | 0.822278 | 0.829536 |     3 | 0.836793 | 0.825573 | 0.817646 | 0.00998952 |
  | train/loss       |  92.3832 |  98.6749 |  100.893 |     3 |   103.11 |  95.9589 |  86.0916 |    8.82839 |
  | train/loss_epoch |  92.3832 |  98.6749 |  100.893 |     3 |   103.11 |  95.9589 |  86.0916 |    8.82839 |
  | train/loss_step  |  1.48327 |  2.96654 |  32.9546 |     3 |  62.9427 |  21.9698 |        0 |    35.5146 |
  | val/f1           | 0.694138 | 0.694601 |  0.69618 |     3 |  0.69776 | 0.695345 | 0.693676 | 0.00214125 |
  | val/loss         |   362.56 |  363.192 |  364.528 |     3 |  365.864 |  363.662 |  361.929 |    2.00919 |

### Coreference probing - frozen BERT (truncated to 10 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=10 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/sliyocua
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/jvh3tlae
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/gt43qaky

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('train/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('train/f1',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | -------------------: | --------------: | --------------------: | ----------: | ------------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_20-19-28 |              462.852 |         112.301 |               112.301 |    0.693504 |       346.297 |      0.807715 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_22-04-53 |             0.598089 |         73.8285 |               73.8285 |    0.706008 |       367.323 |      0.858254 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_01-10-40 |                    0 |         65.6256 |               65.6256 |    0.708891 |       393.226 |      0.870851 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.832984 | 0.858254 | 0.864552 |     3 | 0.870851 | 0.845606 | 0.807715 |  0.0334141 |
  | train/loss       |   69.727 |  73.8285 |  93.0649 |     3 |  112.301 |  83.9185 |  65.6256 |    24.9201 |
  | train/loss_epoch |   69.727 |  73.8285 |  93.0649 |     3 |  112.301 |  83.9185 |  65.6256 |    24.9201 |
  | train/loss_step  | 0.299044 | 0.598089 |  231.725 |     3 |  462.852 |  154.483 |        0 |    267.055 |
  | val/f1           | 0.699756 | 0.706008 |  0.70745 |     3 | 0.708891 | 0.702801 | 0.693504 | 0.00817943 |
  | val/loss         |   356.81 |  367.323 |  380.275 |     3 |  393.226 |  368.949 |  346.297 |    23.5067 |

### Coreference probing - frozen MRPC (9 layers), RE (9 layers), NER (6 layers) and SQUAD (8 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1rnzp565
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/6ulolzr9
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/vlmajaqj

- metric values per seed

  |     | ('model_save_dir',)                                                                                         | ('train/f1',) | ('train/loss',) | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss_step',) |
  | --: | :---------------------------------------------------------------------------------------------------------- | ------------: | --------------: | ------------: | --------------------: | ----------: | -------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_16-41-13 |      0.775797 |         184.437 |       536.094 |               184.437 |    0.668425 |              50.3115 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_18-23-04 |      0.831532 |         108.168 |       575.641 |               108.168 |    0.685327 |              141.029 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_21-12-52 |      0.781436 |         174.983 |       531.673 |               174.983 |    0.667975 |              48.3881 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.778616 | 0.781436 | 0.806484 |     3 | 0.831532 | 0.796255 | 0.775797 |  0.0306806 |
  | train/loss       |  141.575 |  174.983 |   179.71 |     3 |  184.437 |  155.862 |  108.168 |    41.5743 |
  | train/loss_epoch |  141.575 |  174.983 |   179.71 |     3 |  184.437 |  155.862 |  108.168 |    41.5743 |
  | train/loss_step  |  49.3498 |  50.3115 |  95.6703 |     3 |  141.029 |  79.9096 |  48.3881 |    52.9398 |
  | val/f1           |   0.6682 | 0.668425 | 0.676876 |     3 | 0.685327 | 0.673909 | 0.667975 | 0.00989112 |
  | val/loss         |  533.883 |  536.094 |  555.867 |     3 |  575.641 |  547.802 |  531.673 |    24.2096 |

### Coreference probing - frozen BERT (10 layers), MRPC (9 layers), RE (9 layers), NER (6 layers), SQUAD (8 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased:bert-base-cased,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased,bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=10 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/dqkgjkb2
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1g6vhxg6
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/1s15pv4x

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                                         |
  | --: | ------------: | ----------: | --------------: | --------------------: | -------------------: | ------------: | :---------------------------------------------------------------------------------------------------------- |
  |   0 |       591.178 |    0.679308 |         144.818 |               144.818 |            0.0616905 |      0.813244 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_18-01-57 |
  |   1 |       583.442 |    0.673251 |         149.563 |               149.563 |              55.8058 |      0.807695 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_20-28-39 |
  |   2 |        588.89 |    0.676898 |         134.517 |               134.517 |              522.018 |      0.818308 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_22-52-20 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.810469 | 0.813244 | 0.815776 |     3 | 0.818308 | 0.813082 |  0.807695 | 0.00530839 |
  | train/loss       |  139.668 |  144.818 |   147.19 |     3 |  149.563 |  142.966 |   134.517 |    7.69176 |
  | train/loss_epoch |  139.668 |  144.818 |   147.19 |     3 |  149.563 |  142.966 |   134.517 |    7.69176 |
  | train/loss_step  |  27.9337 |  55.8058 |  288.912 |     3 |  522.018 |  192.628 | 0.0616905 |    286.618 |
  | val/f1           | 0.675075 | 0.676898 | 0.678103 |     3 | 0.679308 | 0.676486 |  0.673251 | 0.00304959 |
  | val/loss         |  586.166 |   588.89 |  590.034 |     3 |  591.178 |  587.837 |   583.442 |    3.97411 |

### Coreference probing - frozen MRPC (9 layers), SQUAD (8 layers) + frozen target model with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-coref-hoi,bert-base-cased-mrpc,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/lc5i5fs1
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/h4ptqes1
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/7ct279vw

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                         | ('train/loss_step',) | ('train/loss',) | ('train/f1',) | ('val/loss',) |
  | --: | --------------------: | ----------: | :---------------------------------------------------------------------------------------------------------- | -------------------: | --------------: | ------------: | ------------: |
  |   0 |               124.507 |    0.690993 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-08_22-24-38 |              209.601 |         124.507 |      0.818725 |       477.775 |
  |   1 |               145.483 |    0.684904 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_01-39-06 |                    0 |         145.483 |      0.800875 |       471.637 |
  |   2 |               187.708 |    0.669888 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-09_04-21-01 |              111.389 |         187.708 |      0.759916 |       451.697 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.780395 | 0.800875 |   0.8098 |     3 | 0.818725 | 0.793172 | 0.759916 | 0.0301518 |
  | train/loss       |  134.995 |  145.483 |  166.595 |     3 |  187.708 |  152.566 |  124.507 |   32.1903 |
  | train/loss_epoch |  134.995 |  145.483 |  166.595 |     3 |  187.708 |  152.566 |  124.507 |   32.1903 |
  | train/loss_step  |  55.6945 |  111.389 |  160.495 |     3 |  209.601 |  106.997 |        0 |    104.87 |
  | val/f1           | 0.677396 | 0.684904 | 0.687949 |     3 | 0.690993 | 0.681928 | 0.669888 | 0.0108628 |
  | val/loss         |  461.667 |  471.637 |  474.706 |     3 |  477.775 |  467.037 |  451.697 |   13.6342 |

### Coreference probing - frozen BERT (10 layers), MRPC (9 layers), RE (9 layers), NER (6 layers) and SQUAD (8 layers) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased,bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=10 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  name=probing/coref-truncated-models \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/4a19lvgn
  - seed2: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/9mqq65yv
  - seed3: https://wandb.ai/tanikina/probing-coref-truncated-models-training/runs/m9usxyuq

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                                         | ('val/f1',) | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('train/loss_step',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------------------- | ----------: | ------------: | ------------: | --------------: | -------------------: |
  |   0 |               265.973 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-13_18-10-57 |     0.65414 |        741.04 |      0.757919 |         265.973 |                    0 |
  |   1 |               299.045 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-13_20-38-30 |     0.65399 |       685.569 |      0.740882 |         299.045 |              62.8341 |
  |   2 |               230.385 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/coref-truncated-models/2023-12-13_22-47-03 |    0.659264 |       711.461 |      0.777122 |         230.385 |              34.6284 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.749401 | 0.757919 | 0.767521 |     3 | 0.777122 | 0.758641 | 0.740882 |  0.0181307 |
  | train/loss       |  248.179 |  265.973 |  282.509 |     3 |  299.045 |  265.134 |  230.385 |    34.3379 |
  | train/loss_epoch |  248.179 |  265.973 |  282.509 |     3 |  299.045 |  265.134 |  230.385 |    34.3379 |
  | train/loss_step  |  17.3142 |  34.6284 |  48.7313 |     3 |  62.8341 |  32.4875 |        0 |    31.4717 |
  | val/f1           | 0.654065 |  0.65414 | 0.656702 |     3 | 0.659264 | 0.655798 |  0.65399 | 0.00300271 |
  | val/loss         |  698.515 |  711.461 |   726.25 |     3 |   741.04 |   712.69 |  685.569 |    27.7562 |

## 2023-12-19

### Coreference probing with normalization - frozen NER with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/gonfmbgn
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/8jgnixys
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/86zzg4yr

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                             | ('train/loss_step',) | ('train/loss',) | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) |
  | --: | ------------: | :---------------------------------------------------------------------------------------------- | -------------------: | --------------: | ------------: | --------------------: | ----------: |
  |   0 |      0.374454 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_19-13-47 |              90.6568 |         71.4751 |       84.5088 |               71.4751 |    0.492618 |
  |   1 |      0.331292 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_20-39-48 |              4.29522 |         74.9203 |       77.5316 |               74.9203 |    0.442246 |
  |   2 |      0.398591 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_21-34-00 |              30.7845 |          69.998 |       74.1357 |                69.998 |     0.50349 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.352873 | 0.374454 | 0.386522 |     3 | 0.398591 | 0.368112 | 0.331292 | 0.0340946 |
  | train/loss       |  70.7366 |  71.4751 |  73.1977 |     3 |  74.9203 |  72.1311 |   69.998 |   2.52584 |
  | train/loss_epoch |  70.7366 |  71.4751 |  73.1977 |     3 |  74.9203 |  72.1311 |   69.998 |   2.52584 |
  | train/loss_step  |  17.5399 |  30.7845 |  60.7207 |     3 |  90.6568 |  41.9122 |  4.29522 |   44.2431 |
  | val/f1           | 0.467432 | 0.492618 | 0.498054 |     3 |  0.50349 | 0.479451 | 0.442246 | 0.0326761 |
  | val/loss         |  75.8336 |  77.5316 |  81.0202 |     3 |  84.5088 |  78.7254 |  74.1357 |   5.28862 |

### Coreference probing with normalization - frozen RE with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/jivtinsz
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/tb8brjl3
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/3gg3fnwf

- metric values per seed

  |     | ('train/f1',) | ('val/loss',) | ('train/loss',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                             | ('train/loss_epoch',) |
  | --: | ------------: | ------------: | --------------: | ----------: | -------------------: | :---------------------------------------------------------------------------------------------- | --------------------: |
  |   0 |      0.628328 |       54.8789 |         39.1673 |    0.604592 |              44.6658 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_23-24-42 |               39.1673 |
  |   1 |      0.439866 |       25.7741 |         16.9885 |     0.42011 |              2.43843 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_02-16-11 |               16.9885 |
  |   2 |      0.423448 |       28.2419 |         19.2089 |    0.416315 |              63.3149 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_04-16-54 |               19.2089 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |      std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | -------: |
  | train/f1         | 0.431657 | 0.439866 | 0.534097 |     3 | 0.628328 | 0.497214 | 0.423448 | 0.113845 |
  | train/loss       |  18.0987 |  19.2089 |  29.1881 |     3 |  39.1673 |  25.1215 |  16.9885 |  12.2145 |
  | train/loss_epoch |  18.0987 |  19.2089 |  29.1881 |     3 |  39.1673 |  25.1215 |  16.9885 |  12.2145 |
  | train/loss_step  |  23.5521 |  44.6658 |  53.9904 |     3 |  63.3149 |  36.8064 |  2.43843 |    31.19 |
  | val/f1           | 0.418213 |  0.42011 | 0.512351 |     3 | 0.604592 | 0.480339 | 0.416315 | 0.107623 |
  | val/loss         |   27.008 |  28.2419 |  41.5604 |     3 |  54.8789 |  36.2983 |  25.7741 |  16.1385 |

### Coreference probing with normalization - frozen MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/rx62fgkh
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/7omeafs2
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/ow8w7ocj

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss',) | ('model_save_dir',)                                                                             | ('val/f1',) |
  | --: | -------------------: | ------------: | --------------------: | ------------: | --------------: | :---------------------------------------------------------------------------------------------- | ----------: |
  |   0 |          0.000328628 |      0.440812 |               17.3894 |       22.2899 |         17.3894 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_19-20-09 |    0.427555 |
  |   1 |               1.1371 |      0.729493 |                27.992 |       45.6755 |          27.992 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_20-22-38 |    0.665758 |
  |   2 |              29.5804 |      0.716185 |               29.9298 |       46.0592 |         29.9298 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_23-34-41 |    0.664903 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |      std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | -------: |
  | train/f1         | 0.578498 | 0.716185 | 0.722839 |     3 | 0.729493 |  0.62883 |    0.440812 | 0.162965 |
  | train/loss       |  22.6907 |   27.992 |  28.9609 |     3 |  29.9298 |  25.1037 |     17.3894 |  6.75071 |
  | train/loss_epoch |  22.6907 |   27.992 |  28.9609 |     3 |  29.9298 |  25.1037 |     17.3894 |  6.75071 |
  | train/loss_step  | 0.568717 |   1.1371 |  15.3587 |     3 |  29.5804 |  10.2393 | 0.000328628 |  16.7595 |
  | val/f1           | 0.546229 | 0.664903 | 0.665331 |     3 | 0.665758 | 0.586072 |    0.427555 |  0.13728 |
  | val/loss         |  33.9827 |  45.6755 |  45.8673 |     3 |  46.0592 |  38.0082 |     22.2899 |  13.6138 |

### Coreference probing with normalization - frozen QA with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/38rd8x07
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/fgkmr12u
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/nc24a4wh

- metric values per seed

  |     | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                             | ('val/loss',) | ('train/loss',) | ('train/f1',) | ('train/loss_epoch',) |
  | --: | ----------: | -------------------: | :---------------------------------------------------------------------------------------------- | ------------: | --------------: | ------------: | --------------------: |
  |   0 |    0.602521 |              65.2014 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_02-23-26 |       55.6594 |         50.2751 |      0.560974 |               50.2751 |
  |   1 |    0.594648 |              81.0149 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_05-26-09 |       57.0809 |         51.5113 |      0.550413 |               51.5113 |
  |   2 |    0.589616 |              7.79414 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_08-00-08 |       59.3776 |         52.1831 |      0.545206 |               52.1831 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.547809 | 0.550413 | 0.555693 |     3 | 0.560974 | 0.552198 | 0.545206 | 0.00803444 |
  | train/loss       |  50.8932 |  51.5113 |  51.8472 |     3 |  52.1831 |  51.3231 |  50.2751 |   0.967813 |
  | train/loss_epoch |  50.8932 |  51.5113 |  51.8472 |     3 |  52.1831 |  51.3231 |  50.2751 |   0.967813 |
  | train/loss_step  |  36.4978 |  65.2014 |  73.1082 |     3 |  81.0149 |  51.3368 |  7.79414 |    38.5291 |
  | val/f1           | 0.592132 | 0.594648 | 0.598584 |     3 | 0.602521 | 0.595595 | 0.589616 | 0.00650416 |
  | val/loss         |  56.3701 |  57.0809 |  58.2292 |     3 |  59.3776 |  57.3726 |  55.6594 |     1.8762 |

### Coreference probing with normalization - frozen BERT with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/qcxp814g
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/mt1ttjk6
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/habzbz55

- metric values per seed

  |     | ('train/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('val/loss',) | ('val/f1',) | ('train/f1',) | ('train/loss_step',) |
  | --: | --------------: | --------------------: | :---------------------------------------------------------------------------------------------- | ------------: | ----------: | ------------: | -------------------: |
  |   0 |         22.0661 |               22.0661 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_19-20-24 |       47.8592 |    0.664629 |      0.767183 |              17.4379 |
  |   1 |         25.6453 |               25.6453 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_21-03-01 |       46.8137 |    0.664283 |      0.747177 |             0.843221 |
  |   2 |         19.8816 |               19.8816 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_22-24-53 |       50.6923 |    0.666068 |      0.781902 |              34.2952 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |         std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ----------: |
  | train/f1         |  0.75718 | 0.767183 | 0.774543 |     3 | 0.781902 | 0.765421 | 0.747177 |   0.0174296 |
  | train/loss       |  20.9738 |  22.0661 |  23.8557 |     3 |  25.6453 |   22.531 |  19.8816 |     2.90983 |
  | train/loss_epoch |  20.9738 |  22.0661 |  23.8557 |     3 |  25.6453 |   22.531 |  19.8816 |     2.90983 |
  | train/loss_step  |  9.14057 |  17.4379 |  25.8666 |     3 |  34.2952 |  17.5254 | 0.843221 |     16.7262 |
  | val/f1           | 0.664456 | 0.664629 | 0.665349 |     3 | 0.666068 | 0.664993 | 0.664283 | 0.000946719 |
  | val/loss         |  47.3364 |  47.8592 |  49.2757 |     3 |  50.6923 |   48.455 |  46.8137 |     2.00676 |

### Coreference probing with normalization - frozen target with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/9c2egekb
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/vm3y8a7j
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/sh9acofq

- metric values per seed

  |     | ('train/loss_step',) | ('model_save_dir',)                                                                             | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) | ('val/f1',) | ('train/loss',) |
  | --: | -------------------: | :---------------------------------------------------------------------------------------------- | --------------------: | ------------: | ------------: | ----------: | --------------: |
  |   0 |          1.66893e-05 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_00-21-37 |               5.86364 |       68.8238 |      0.916157 |    0.721886 |         5.86364 |
  |   1 |          0.000743973 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_02-00-48 |               5.46372 |       70.6066 |      0.918693 |    0.723964 |         5.46372 |
  |   2 |           0.00563328 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_03-49-51 |               4.74006 |       75.2874 |      0.923055 |    0.724028 |         4.74006 |

- aggregated values:

  |                  |         25% |         50% |        75% | count |        max |       mean |         min |        std |
  | :--------------- | ----------: | ----------: | ---------: | ----: | ---------: | ---------: | ----------: | ---------: |
  | train/f1         |    0.917425 |    0.918693 |   0.920874 |     3 |   0.923055 |   0.919302 |    0.916157 | 0.00348942 |
  | train/loss       |     5.10189 |     5.46372 |    5.66368 |     3 |    5.86364 |    5.35581 |     4.74006 |   0.569507 |
  | train/loss_epoch |     5.10189 |     5.46372 |    5.66368 |     3 |    5.86364 |    5.35581 |     4.74006 |   0.569507 |
  | train/loss_step  | 0.000380331 | 0.000743973 | 0.00318862 |     3 | 0.00563328 | 0.00213131 | 1.66893e-05 | 0.00305451 |
  | val/f1           |    0.722925 |    0.723964 |   0.723996 |     3 |   0.724028 |   0.723293 |    0.721886 |  0.0012184 |
  | val/loss         |     69.7152 |     70.6066 |     72.947 |     3 |    75.2874 |    71.5726 |     68.8238 |     3.3383 |

### Coreference probing with normalization - frozen MRPC, RE, NER, QA with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/7soi8t0a
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/hh22jqzw
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/5klnma3z

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('train/loss',) |
  | --: | ------------: | ----------: | ------------: | -------------------: | --------------------: | :---------------------------------------------------------------------------------------------- | --------------: |
  |   0 |       48.1774 |    0.656211 |      0.713603 |              102.451 |               28.7021 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_19-41-31 |         28.7021 |
  |   1 |       47.2379 |    0.654313 |      0.701002 |              5.88729 |               30.9851 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_23-35-03 |         30.9851 |
  |   2 |        48.904 |    0.656252 |       0.70544 |              22.8048 |               30.5328 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_02-53-19 |         30.5328 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.703221 |  0.70544 | 0.709522 |     3 | 0.713603 | 0.706682 | 0.701002 | 0.00639166 |
  | train/loss       |  29.6174 |  30.5328 |   30.759 |     3 |  30.9851 |  30.0733 |  28.7021 |     1.2089 |
  | train/loss_epoch |  29.6174 |  30.5328 |   30.759 |     3 |  30.9851 |  30.0733 |  28.7021 |     1.2089 |
  | train/loss_step  |   14.346 |  22.8048 |   62.628 |     3 |  102.451 |  43.7144 |  5.88729 |     51.566 |
  | val/f1           | 0.655262 | 0.656211 | 0.656231 |     3 | 0.656252 | 0.655592 | 0.654313 | 0.00110803 |
  | val/loss         |  47.7076 |  48.1774 |  48.5407 |     3 |   48.904 |  48.1064 |  47.2379 |   0.835299 |

### Coreference probing with normalization - frozen RE, NER, QA with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/ummaqljy
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/pom01vfp
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/k40fvm18

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) | ('val/loss',) | ('train/loss_step',) | ('train/loss',) | ('model_save_dir',)                                                                             |
  | --: | --------------------: | ------------: | ----------: | ------------: | -------------------: | --------------: | :---------------------------------------------------------------------------------------------- |
  |   0 |               36.1398 |      0.661055 |    0.637295 |       52.4738 |            0.0703753 |         36.1398 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_19-46-56 |
  |   1 |               36.6651 |      0.656616 |    0.636053 |       51.9889 |              113.394 |         36.6651 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-17_23-33-40 |
  |   2 |               36.1854 |      0.659374 |    0.637668 |       51.3551 |             0.385166 |         36.1854 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_03-09-19 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |         std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ----------: |
  | train/f1         | 0.657995 | 0.659374 | 0.660214 |     3 | 0.661055 | 0.659015 |  0.656616 |  0.00224117 |
  | train/loss       |  36.1626 |  36.1854 |  36.4253 |     3 |  36.6651 |  36.3301 |   36.1398 |    0.291023 |
  | train/loss_epoch |  36.1626 |  36.1854 |  36.4253 |     3 |  36.6651 |  36.3301 |   36.1398 |    0.291023 |
  | train/loss_step  | 0.227771 | 0.385166 |  56.8896 |     3 |  113.394 |  37.9499 | 0.0703753 |     65.3368 |
  | val/f1           | 0.636674 | 0.637295 | 0.637481 |     3 | 0.637668 | 0.637005 |  0.636053 | 0.000845502 |
  | val/loss         |   51.672 |  51.9889 |  52.2313 |     3 |  52.4738 |  51.9393 |   51.3551 |    0.561004 |

### Coreference probing with normalization - frozen RE, NER, MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/envne1nu
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/qpczi5jh
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/tn3pn775

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                             | ('train/loss',) | ('train/f1',) | ('train/loss_step',) |
  | --: | --------------------: | ------------: | ----------: | :---------------------------------------------------------------------------------------------- | --------------: | ------------: | -------------------: |
  |   0 |               28.5054 |       48.6782 |    0.653039 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_06-55-12 |         28.5054 |      0.716517 |            0.0182618 |
  |   1 |                31.721 |       49.3379 |    0.649622 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_10-42-49 |          31.721 |      0.697944 |              3.99333 |
  |   2 |               31.1708 |       49.0504 |    0.651955 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_13-43-33 |         31.1708 |       0.70118 |              36.3688 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.699562 |  0.70118 | 0.708849 |     3 | 0.716517 | 0.705214 |  0.697944 | 0.00992185 |
  | train/loss       |  29.8381 |  31.1708 |  31.4459 |     3 |   31.721 |  30.4657 |   28.5054 |    1.71982 |
  | train/loss_epoch |  29.8381 |  31.1708 |  31.4459 |     3 |   31.721 |  30.4657 |   28.5054 |    1.71982 |
  | train/loss_step  |   2.0058 |  3.99333 |  20.1811 |     3 |  36.3688 |  13.4601 | 0.0182618 |    19.9388 |
  | val/f1           | 0.650789 | 0.651955 | 0.652497 |     3 | 0.653039 | 0.651539 |  0.649622 | 0.00174623 |
  | val/loss         |  48.8643 |  49.0504 |  49.1942 |     3 |  49.3379 |  49.0222 |   48.6782 |   0.330756 |

### Coreference probing with normalization - frozen QA, NER, MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/6ina6r97
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/3pa33avv
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/8evmpcod

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('val/f1',) | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('train/loss_step',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------- | ----------: | ------------: | ------------: | --------------: | -------------------: |
  |   0 |               37.1514 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_16-48-28 |    0.650186 |       47.8781 |      0.667914 |         37.1514 |              36.8341 |
  |   1 |               31.6446 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_19-37-13 |    0.658343 |        47.238 |      0.699212 |         31.6446 |              18.9016 |
  |   2 |               37.3797 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-18_23-53-17 |     0.64542 |       48.9429 |       0.65934 |         37.3797 |              10.1475 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |     min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ------: | --------: |
  | train/f1         | 0.663627 | 0.667914 | 0.683563 |     3 | 0.699212 | 0.675489 | 0.65934 | 0.0209876 |
  | train/loss       |   34.398 |  37.1514 |  37.2656 |     3 |  37.3797 |  35.3919 | 31.6446 |   3.24728 |
  | train/loss_epoch |   34.398 |  37.1514 |  37.2656 |     3 |  37.3797 |  35.3919 | 31.6446 |   3.24728 |
  | train/loss_step  |  14.5245 |  18.9016 |  27.8679 |     3 |  36.8341 |  21.9611 | 10.1475 |   13.6039 |
  | val/f1           | 0.647803 | 0.650186 | 0.654265 |     3 | 0.658343 | 0.651316 | 0.64542 | 0.0065352 |
  | val/loss         |   47.558 |  47.8781 |  48.4105 |     3 |  48.9429 |  48.0196 |  47.238 |  0.861238 |

### Coreference probing with normalization - frozen QA, RE, MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/fmhz3kcl
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/j63zc8s8
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/6tudtgag

- metric values per seed

  |     | ('train/loss_step',) | ('val/f1',) | ('model_save_dir',)                                                                             | ('train/f1',) | ('train/loss',) | ('train/loss_epoch',) | ('val/loss',) |
  | --: | -------------------: | ----------: | :---------------------------------------------------------------------------------------------- | ------------: | --------------: | --------------------: | ------------: |
  |   0 |              33.1225 |    0.653703 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-19_02-29-26 |      0.709982 |         29.5407 |               29.5407 |       46.4006 |
  |   1 |             0.204524 |    0.649948 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-19_05-37-57 |      0.701985 |         30.7002 |               30.7002 |       47.1603 |
  |   2 |                18.66 |    0.652928 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-19_08-26-42 |      0.706523 |          30.432 |                30.432 |        48.774 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.704254 | 0.706523 | 0.708252 |     3 | 0.709982 | 0.706163 | 0.701985 | 0.00401039 |
  | train/loss       |  29.9864 |   30.432 |  30.5661 |     3 |  30.7002 |  30.2243 |  29.5407 |    0.60701 |
  | train/loss_epoch |  29.9864 |   30.432 |  30.5661 |     3 |  30.7002 |  30.2243 |  29.5407 |    0.60701 |
  | train/loss_step  |  9.43227 |    18.66 |  25.8913 |     3 |  33.1225 |   17.329 | 0.204524 |    16.4993 |
  | val/f1           | 0.651438 | 0.652928 | 0.653315 |     3 | 0.653703 | 0.652193 | 0.649948 |  0.0019823 |
  | val/loss         |  46.7804 |  47.1603 |  47.9671 |     3 |   48.774 |   47.445 |  46.4006 |    1.21205 |

### Coreference probing - frozen RE, NER, QA with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/3udot8eg
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/9ttst7y5
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/y5rcy8c2

- metric values per seed

  |     | ('model_save_dir',)                                                                       | ('val/loss',) | ('train/f1',) | ('train/loss_step',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss',) |
  | --: | :---------------------------------------------------------------------------------------- | ------------: | ------------: | -------------------: | ----------: | --------------------: | --------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-17_20-02-59 |       248.951 |      0.549666 |              166.738 |    0.601872 |               230.768 |         230.768 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-17_23-12-20 |       247.667 |      0.520001 |              465.965 |    0.586383 |               279.048 |         279.048 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_01-27-18 |       290.776 |       0.52328 |              263.435 |    0.580256 |               272.851 |         272.851 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.521641 |  0.52328 | 0.536473 |     3 | 0.549666 | 0.530982 | 0.520001 | 0.0162634 |
  | train/loss       |   251.81 |  272.851 |  275.949 |     3 |  279.048 |  260.889 |  230.768 |   26.2687 |
  | train/loss_epoch |   251.81 |  272.851 |  275.949 |     3 |  279.048 |  260.889 |  230.768 |   26.2687 |
  | train/loss_step  |  215.086 |  263.435 |    364.7 |     3 |  465.965 |  298.712 |  166.738 |   152.701 |
  | val/f1           | 0.583319 | 0.586383 | 0.594127 |     3 | 0.601872 | 0.589504 | 0.580256 | 0.0111405 |
  | val/loss         |  248.309 |  248.951 |  269.863 |     3 |  290.776 |  262.465 |  247.667 |   24.5264 |

### Coreference probing - frozen QA, NER, MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/x1lofyy8
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/2u175ogt
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/qxbg8ieg

- metric values per seed

  |     | ('train/loss_step',) | ('val/loss',) | ('model_save_dir',)                                                                       | ('train/f1',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss',) |
  | --: | -------------------: | ------------: | :---------------------------------------------------------------------------------------- | ------------: | ----------: | --------------------: | --------------: |
  |   0 |              280.139 |       196.743 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_03-47-35 |       0.55502 |    0.619841 |               201.685 |         201.685 |
  |   1 |              69.9742 |       198.249 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_06-27-47 |      0.516303 |    0.596272 |                244.26 |          244.26 |
  |   2 |              40.2131 |       200.215 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_08-09-37 |      0.512569 |    0.594232 |               250.681 |         250.681 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.514436 | 0.516303 | 0.535661 |     3 |  0.55502 | 0.527964 | 0.512569 | 0.0235054 |
  | train/loss       |  222.973 |   244.26 |   247.47 |     3 |  250.681 |  232.209 |  201.685 |   26.6281 |
  | train/loss_epoch |  222.973 |   244.26 |   247.47 |     3 |  250.681 |  232.209 |  201.685 |   26.6281 |
  | train/loss_step  |  55.0936 |  69.9742 |  175.057 |     3 |  280.139 |  130.109 |  40.2131 |   130.779 |
  | val/f1           | 0.595252 | 0.596272 | 0.608056 |     3 | 0.619841 | 0.603448 | 0.594232 |  0.014233 |
  | val/loss         |  197.496 |  198.249 |  199.232 |     3 |  200.215 |  198.403 |  196.743 |   1.74105 |

### Coreference probing - frozen QA, RE, MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/c9l0dylz
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/grpetavx
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/h151rpvc

- metric values per seed

  |     | ('train/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('model_save_dir',)                                                                       | ('val/loss',) | ('val/f1',) | ('train/loss',) |
  | --: | ------------: | --------------------: | -------------------: | :---------------------------------------------------------------------------------------- | ------------: | ----------: | --------------: |
  |   0 |      0.597682 |               200.108 |              681.312 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_09-46-40 |        230.36 |     0.61542 |         200.108 |
  |   1 |      0.592396 |               206.583 |              376.216 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_12-22-34 |       231.394 |    0.616908 |         206.583 |
  |   2 |      0.590505 |               210.038 |               81.074 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_14-47-24 |       237.465 |    0.622694 |         210.038 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.591451 | 0.592396 | 0.595039 |     3 | 0.597682 | 0.593528 | 0.590505 | 0.00371961 |
  | train/loss       |  203.345 |  206.583 |   208.31 |     3 |  210.038 |  205.576 |  200.108 |    5.04089 |
  | train/loss_epoch |  203.345 |  206.583 |   208.31 |     3 |  210.038 |  205.576 |  200.108 |    5.04089 |
  | train/loss_step  |  228.645 |  376.216 |  528.764 |     3 |  681.312 |  379.534 |   81.074 |    300.133 |
  | val/f1           | 0.616164 | 0.616908 | 0.619801 |     3 | 0.622694 | 0.618341 |  0.61542 | 0.00384261 |
  | val/loss         |  230.877 |  231.394 |  234.429 |     3 |  237.465 |  233.073 |   230.36 |    3.83854 |

### Coreference probing - frozen RE, NER, MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/sef6i5oe
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/2axv0vl1
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/ps4riknw

- metric values per seed

  |     | ('train/loss_step',) | ('val/f1',) | ('val/loss',) | ('train/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('model_save_dir',)                                                                       |
  | --: | -------------------: | ----------: | ------------: | --------------: | --------------------: | ------------: | :---------------------------------------------------------------------------------------- |
  |   0 |               108.58 |    0.618648 |       227.437 |         182.612 |               182.612 |      0.599982 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-19_21-19-18 |
  |   1 |              230.278 |    0.645249 |       212.226 |         126.172 |               126.172 |      0.651422 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_00-13-11 |
  |   2 |              34.2899 |    0.604329 |       242.958 |         217.768 |               217.768 |      0.569193 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_04-40-56 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.584588 | 0.599982 | 0.625702 |     3 | 0.651422 | 0.606866 | 0.569193 | 0.0415443 |
  | train/loss       |  154.392 |  182.612 |   200.19 |     3 |  217.768 |  175.517 |  126.172 |   46.2081 |
  | train/loss_epoch |  154.392 |  182.612 |   200.19 |     3 |  217.768 |  175.517 |  126.172 |   46.2081 |
  | train/loss_step  |  71.4349 |   108.58 |  169.429 |     3 |  230.278 |  124.383 |  34.2899 |   98.9453 |
  | val/f1           | 0.611489 | 0.618648 | 0.631948 |     3 | 0.645249 | 0.622742 | 0.604329 | 0.0207646 |
  | val/loss         |  219.832 |  227.437 |  235.198 |     3 |  242.958 |   227.54 |  212.226 |   15.3661 |

### Coreference probing - frozen MRPC, RE, NER, QA with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/kf5q5q59
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/18t2ynve
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/gwe6qx5u

- metric values per seed

  |     | ('train/f1',) | ('val/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                       | ('train/loss',) | ('train/loss_epoch',) | ('val/f1',) |
  | --: | ------------: | ------------: | -------------------: | :---------------------------------------------------------------------------------------- | --------------: | --------------------: | ----------: |
  |   0 |      0.620618 |       211.469 |              128.068 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_04-05-32 |         145.443 |               145.443 |    0.631344 |
  |   1 |      0.603311 |       190.521 |              161.399 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_07-19-34 |         162.966 |               162.966 |    0.629587 |
  |   2 |      0.557692 |       200.257 |              766.528 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-18_10-01-42 |         199.632 |               199.632 |     0.60474 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.580502 | 0.603311 | 0.611965 |     3 | 0.620618 | 0.593874 | 0.557692 | 0.0325074 |
  | train/loss       |  154.205 |  162.966 |  181.299 |     3 |  199.632 |  169.347 |  145.443 |   27.6522 |
  | train/loss_epoch |  154.205 |  162.966 |  181.299 |     3 |  199.632 |  169.347 |  145.443 |   27.6522 |
  | train/loss_step  |  144.733 |  161.399 |  463.963 |     3 |  766.528 |  351.998 |  128.068 |    359.38 |
  | val/f1           | 0.617164 | 0.629587 | 0.630466 |     3 | 0.631344 | 0.621891 |  0.60474 |  0.014879 |
  | val/loss         |  195.389 |  200.257 |  205.863 |     3 |  211.469 |  200.749 |  190.521 |   10.4826 |

### Coreference probing with truncation - frozen RE (9 layer), NER (6 layer), QA (8 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/ufv393g0
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/hujn5gz2
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/61jydam7

- metric values per seed

  |     | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('val/f1',) | ('model_save_dir',)                                                                            | ('train/loss_epoch',) | ('val/loss',) |
  | --: | ------------: | --------------: | -------------------: | ----------: | :--------------------------------------------------------------------------------------------- | --------------------: | ------------: |
  |   0 |      0.714093 |         354.968 |              471.208 |    0.642457 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-17_19-59-57 |               354.968 |       715.573 |
  |   1 |      0.780096 |         224.666 |              270.242 |    0.656157 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-17_21-50-52 |               224.666 |       763.283 |
  |   2 |      0.745908 |         290.715 |              544.875 |    0.652541 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_00-51-06 |               290.715 |       787.992 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |     0.73 | 0.745908 | 0.763002 |     3 | 0.780096 | 0.746699 | 0.714093 |  0.0330085 |
  | train/loss       |  257.691 |  290.715 |  322.842 |     3 |  354.968 |  290.117 |  224.666 |    65.1531 |
  | train/loss_epoch |  257.691 |  290.715 |  322.842 |     3 |  354.968 |  290.117 |  224.666 |    65.1531 |
  | train/loss_step  |  370.725 |  471.208 |  508.041 |     3 |  544.875 |  428.775 |  270.242 |    142.149 |
  | val/f1           | 0.647499 | 0.652541 | 0.654349 |     3 | 0.656157 | 0.650385 | 0.642457 | 0.00709999 |
  | val/loss         |  739.428 |  763.283 |  775.637 |     3 |  787.992 |  755.616 |  715.573 |    36.8131 |

### Coreference probing with truncation - frozen MRPC (9 layer), NER (6 layer), QA (8 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/vylsknh4
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/pswyqd56
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/d0edi1pk

- metric values per seed

  |     | ('train/loss',) | ('val/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                            | ('train/f1',) |
  | --: | --------------: | ------------: | -------------------: | --------------------: | ----------: | :--------------------------------------------------------------------------------------------- | ------------: |
  |   0 |          324.31 |        726.94 |              189.798 |                324.31 |    0.649263 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_03-13-50 |      0.731751 |
  |   1 |         291.792 |        726.74 |              338.732 |               291.792 |    0.655341 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_05-12-17 |      0.749938 |
  |   2 |         191.578 |       764.662 |                    0 |               191.578 |    0.665642 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_07-26-28 |      0.801797 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.740845 | 0.749938 | 0.775867 |     3 | 0.801797 | 0.761162 | 0.731751 |  0.0363468 |
  | train/loss       |  241.685 |  291.792 |  308.051 |     3 |   324.31 |  269.227 |  191.578 |    69.1832 |
  | train/loss_epoch |  241.685 |  291.792 |  308.051 |     3 |   324.31 |  269.227 |  191.578 |    69.1832 |
  | train/loss_step  |  94.8992 |  189.798 |  264.265 |     3 |  338.732 |  176.177 |        0 |    169.776 |
  | val/f1           | 0.652302 | 0.655341 | 0.660491 |     3 | 0.665642 | 0.656748 | 0.649263 | 0.00827988 |
  | val/loss         |   726.84 |   726.94 |  745.801 |     3 |  764.662 |  739.447 |   726.74 |    21.8367 |

### Coreference probing with truncation - frozen MRPC (9 layer), RE (9 layer), QA (8 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/dprc4wmd
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/4y4kz57l
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/dzuypvot

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                            | ('train/loss_step',) | ('val/loss',) | ('train/loss',) | ('val/f1',) | ('train/f1',) |
  | --: | --------------------: | :--------------------------------------------------------------------------------------------- | -------------------: | ------------: | --------------: | ----------: | ------------: |
  |   0 |               105.857 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_10-48-53 |             0.122762 |       807.914 |         105.857 |     0.67393 |      0.856183 |
  |   1 |               217.109 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_15-44-05 |              131.769 |       812.584 |         217.109 |    0.665269 |      0.795788 |
  |   2 |               262.777 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_18-49-59 |              46.3554 |       803.596 |         262.777 |    0.661352 |      0.773227 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.784508 | 0.795788 | 0.825985 |     3 | 0.856183 | 0.808399 | 0.773227 |  0.0428916 |
  | train/loss       |  161.483 |  217.109 |  239.943 |     3 |  262.777 |  195.247 |  105.857 |     80.712 |
  | train/loss_epoch |  161.483 |  217.109 |  239.943 |     3 |  262.777 |  195.247 |  105.857 |     80.712 |
  | train/loss_step  |  23.2391 |  46.3554 |  89.0621 |     3 |  131.769 |  59.4156 | 0.122762 |    66.7877 |
  | val/f1           | 0.663311 | 0.665269 |   0.6696 |     3 |  0.67393 |  0.66685 | 0.661352 | 0.00643641 |
  | val/loss         |  805.755 |  807.914 |  810.249 |     3 |  812.584 |  808.031 |  803.596 |    4.49523 |

### Coreference probing with truncation - frozen MRPC (9 layer), RE (9 layer), NER (6 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/uhmna1ja
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/2gkzyhul
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/keesw0c3

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                            | ('train/f1',) | ('val/f1',) | ('val/loss',) | ('train/loss',) |
  | --: | -------------------: | --------------------: | :--------------------------------------------------------------------------------------------- | ------------: | ----------: | ------------: | --------------: |
  |   0 |              5.88322 |               293.103 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-19_21-53-57 |      0.744763 |    0.653596 |       734.703 |         293.103 |
  |   1 |              19.3133 |               239.715 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_00-28-23 |      0.773015 |    0.663318 |       755.433 |         239.715 |
  |   2 |          0.000792027 |               233.728 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_03-35-21 |      0.774022 |     0.66177 |       734.002 |         233.728 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         | 0.758889 | 0.773015 | 0.773519 |     3 | 0.774022 | 0.763933 |    0.744763 |  0.0166095 |
  | train/loss       |  236.721 |  239.715 |  266.409 |     3 |  293.103 |  255.515 |     233.728 |    32.6895 |
  | train/loss_epoch |  236.721 |  239.715 |  266.409 |     3 |  293.103 |  255.515 |     233.728 |    32.6895 |
  | train/loss_step  |  2.94201 |  5.88322 |  12.5983 |     3 |  19.3133 |  8.39912 | 0.000792027 |    9.89904 |
  | val/f1           | 0.657683 |  0.66177 | 0.662544 |     3 | 0.663318 | 0.659561 |    0.653596 | 0.00522377 |
  | val/loss         |  734.353 |  734.703 |  745.068 |     3 |  755.433 |   741.38 |     734.002 |    12.1759 |

### Coreference probing with truncation - frozen MRPC (9 layer), RE (9 layer), NER (6 layer), QA (8 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/22mia798
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/4pynmfgl
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/k2ekyio8

- metric values per seed

  |     | ('model_save_dir',)                                                                            | ('train/loss',) | ('val/loss',) | ('train/f1',) | ('train/loss_step',) | ('val/f1',) | ('train/loss_epoch',) |
  | --: | :--------------------------------------------------------------------------------------------- | --------------: | ------------: | ------------: | -------------------: | ----------: | --------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-17_19-51-17 |         265.841 |       743.353 |      0.758746 |                    0 |    0.653618 |               265.841 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-17_22-23-10 |         257.644 |       724.205 |      0.762854 |               147.46 |    0.653763 |               257.644 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-18_00-59-07 |         210.603 |       721.847 |      0.785828 |              715.651 |    0.662423 |               210.603 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |   0.7608 | 0.762854 | 0.774341 |     3 | 0.785828 | 0.769143 | 0.758746 |   0.014595 |
  | train/loss       |  234.123 |  257.644 |  261.742 |     3 |  265.841 |  244.696 |  210.603 |    29.8086 |
  | train/loss_epoch |  234.123 |  257.644 |  261.742 |     3 |  265.841 |  244.696 |  210.603 |    29.8086 |
  | train/loss_step  |  73.7298 |   147.46 |  431.555 |     3 |  715.651 |  287.703 |        0 |    377.876 |
  | val/f1           | 0.653691 | 0.653763 | 0.658093 |     3 | 0.662423 | 0.656602 | 0.653618 | 0.00504224 |
  | val/loss         |  723.026 |  724.205 |  733.779 |     3 |  743.353 |  729.802 |  721.847 |    11.7949 |

## 2023-12-21

### Coreference probing - frozen RE + MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/e5iajv2v
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/bt8yfltx
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/1i5uohob

- metric values per seed

  |     | ('train/loss',) | ('train/f1',) | ('model_save_dir',)                                                                       | ('train/loss_step',) | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) |
  | --: | --------------: | ------------: | :---------------------------------------------------------------------------------------- | -------------------: | --------------------: | ------------: | ----------: |
  |   0 |         143.837 |      0.669683 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_10-21-52 |               111.28 |               143.837 |       273.064 |    0.638504 |
  |   1 |         259.885 |      0.587602 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_14-39-44 |              41.9752 |               259.885 |       305.137 |    0.610777 |
  |   2 |         282.842 |      0.570825 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_16-42-42 |              338.716 |               282.842 |       331.625 |     0.60351 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.579214 | 0.587602 | 0.628643 |     3 | 0.669683 |  0.60937 | 0.570825 |  0.052902 |
  | train/loss       |  201.861 |  259.885 |  271.364 |     3 |  282.842 |  228.855 |  143.837 |   74.5168 |
  | train/loss_epoch |  201.861 |  259.885 |  271.364 |     3 |  282.842 |  228.855 |  143.837 |   74.5168 |
  | train/loss_step  |  76.6277 |   111.28 |  224.998 |     3 |  338.716 |   163.99 |  41.9752 |   155.234 |
  | val/f1           | 0.607144 | 0.610777 |  0.62464 |     3 | 0.638504 | 0.617597 |  0.60351 | 0.0184666 |
  | val/loss         |  289.101 |  305.137 |  318.381 |     3 |  331.625 |  303.275 |  273.064 |   29.3246 |

### Coreference probing - frozen NER + MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/jp90fce8
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/tn3hgy70
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/h7jcyh8b

- metric values per seed

  |     | ('val/loss',) | ('train/loss_step',) | ('train/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                       | ('val/f1',) | ('train/f1',) |
  | --: | ------------: | -------------------: | --------------: | --------------------: | :---------------------------------------------------------------------------------------- | ----------: | ------------: |
  |   0 |        317.01 |              166.112 |         364.153 |               364.153 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_18-30-45 |    0.497727 |      0.446857 |
  |   1 |       268.768 |                60.21 |         335.358 |               335.358 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_19-34-46 |    0.555796 |       0.47951 |
  |   2 |       258.391 |              263.367 |         315.566 |               315.566 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_21-12-35 |    0.587828 |      0.500728 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.463183 |  0.47951 | 0.490119 |     3 | 0.500728 | 0.475698 | 0.446857 | 0.0271374 |
  | train/loss       |  325.462 |  335.358 |  349.756 |     3 |  364.153 |  338.359 |  315.566 |   24.4323 |
  | train/loss_epoch |  325.462 |  335.358 |  349.756 |     3 |  364.153 |  338.359 |  315.566 |   24.4323 |
  | train/loss_step  |  113.161 |  166.112 |  214.739 |     3 |  263.367 |  163.229 |    60.21 |   101.609 |
  | val/f1           | 0.526762 | 0.555796 | 0.571812 |     3 | 0.587828 | 0.547117 | 0.497727 |  0.045673 |
  | val/loss         |  263.579 |  268.768 |  292.889 |     3 |   317.01 |   281.39 |  258.391 |   31.2816 |

### Coreference probing - frozen RE + NER with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/3dx2f46m
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/zsfbimgx
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/tj0ddwq4

- metric values per seed

  |     | ('model_save_dir',)                                                                       | ('train/loss',) | ('train/f1',) | ('val/f1',) | ('train/loss_step',) | ('val/loss',) | ('train/loss_epoch',) |
  | --: | :---------------------------------------------------------------------------------------- | --------------: | ------------: | ----------: | -------------------: | ------------: | --------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_10-23-53 |         391.594 |      0.479013 |    0.553419 |              756.699 |       376.404 |               391.594 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_12-26-42 |         394.741 |      0.479392 |    0.552739 |              65.4634 |       374.947 |               394.741 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_14-30-02 |         322.225 |      0.511811 |    0.577717 |              1243.12 |       343.503 |               322.225 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.479202 | 0.479392 | 0.495601 |     3 | 0.511811 | 0.490072 | 0.479013 | 0.0188277 |
  | train/loss       |  356.909 |  391.594 |  393.167 |     3 |  394.741 |   369.52 |  322.225 |   40.9891 |
  | train/loss_epoch |  356.909 |  391.594 |  393.167 |     3 |  394.741 |   369.52 |  322.225 |   40.9891 |
  | train/loss_step  |  411.081 |  756.699 |   999.91 |     3 |  1243.12 |  688.428 |  65.4634 |    591.79 |
  | val/f1           | 0.553079 | 0.553419 | 0.565568 |     3 | 0.577717 | 0.561292 | 0.552739 |  0.014229 |
  | val/loss         |  359.225 |  374.947 |  375.676 |     3 |  376.404 |  364.951 |  343.503 |   18.5892 |

### Coreference probing - frozen SQUAD + MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/aunjyw7m
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/budacbo1
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/htylbe84

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('val/f1',) | ('model_save_dir',)                                                                       | ('val/loss',) | ('train/loss',) |
  | --: | --------------------: | -------------------: | ------------: | ----------: | :---------------------------------------------------------------------------------------- | ------------: | --------------: |
  |   0 |               272.889 |                    0 |        0.5377 |    0.611119 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_17-28-32 |         224.4 |         272.889 |
  |   1 |               295.925 |              162.267 |      0.523801 |    0.596927 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_19-04-25 |       240.633 |         295.925 |
  |   2 |               255.697 |              30.1113 |        0.5496 |    0.617092 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_20-30-08 |       222.879 |         255.697 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.530751 |   0.5377 |  0.54365 |     3 |   0.5496 | 0.537034 | 0.523801 | 0.0129125 |
  | train/loss       |  264.293 |  272.889 |  284.407 |     3 |  295.925 |  274.837 |  255.697 |   20.1845 |
  | train/loss_epoch |  264.293 |  272.889 |  284.407 |     3 |  295.925 |  274.837 |  255.697 |   20.1845 |
  | train/loss_step  |  15.0557 |  30.1113 |  96.1893 |     3 |  162.267 |  64.1262 |        0 |   86.3158 |
  | val/f1           | 0.604023 | 0.611119 | 0.614106 |     3 | 0.617092 | 0.608379 | 0.596927 |  0.010358 |
  | val/loss         |  223.639 |    224.4 |  232.516 |     3 |  240.633 |  229.304 |  222.879 |    9.8405 |

### Coreference probing - frozen RE + SQUAD with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/npp30ooy
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/s4rmb1ti
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/x7au9n58

- metric values per seed

  |     | ('val/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('train/f1',) | ('train/loss',) | ('model_save_dir',)                                                                       | ('val/loss',) |
  | --: | ----------: | -------------------: | --------------------: | ------------: | --------------: | :---------------------------------------------------------------------------------------- | ------------: |
  |   0 |    0.554658 |              42.2718 |               438.989 |      0.487325 |         438.989 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_10-25-49 |       373.472 |
  |   1 |    0.618238 |              34.1077 |               199.569 |        0.5942 |         199.569 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_11-57-27 |       257.167 |
  |   2 |    0.556986 |                    0 |               435.868 |      0.492462 |         435.868 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_16-34-58 |       370.616 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.489894 | 0.492462 | 0.543331 |     3 |   0.5942 | 0.524662 | 0.487325 | 0.0602762 |
  | train/loss       |  317.718 |  435.868 |  437.428 |     3 |  438.989 |  358.142 |  199.569 |   137.337 |
  | train/loss_epoch |  317.718 |  435.868 |  437.428 |     3 |  438.989 |  358.142 |  199.569 |   137.337 |
  | train/loss_step  |  17.0538 |  34.1077 |  38.1897 |     3 |  42.2718 |  25.4598 |        0 |   22.4235 |
  | val/f1           | 0.555822 | 0.556986 | 0.587612 |     3 | 0.618238 | 0.576627 | 0.554658 | 0.0360549 |
  | val/loss         |  313.891 |  370.616 |  372.044 |     3 |  373.472 |  333.752 |  257.167 |   66.3399 |

### Coreference probing - frozen NER + SQUAD with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/full \
  --multirun


  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-full-training/runs/atth4js8
  - seed2: https://wandb.ai/tanikina/probing-full-training/runs/e4fgf05q
  - seed3: https://wandb.ai/tanikina/probing-full-training/runs/6h7k6m0q

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/f1',) | ('model_save_dir',)                                                                       | ('train/loss',) | ('val/f1',) | ('train/loss_step',) | ('val/loss',) |
  | --: | --------------------: | ------------: | :---------------------------------------------------------------------------------------- | --------------: | ----------: | -------------------: | ------------: |
  |   0 |               352.262 |      0.462063 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_18-09-20 |         352.262 |    0.572577 |              517.224 |       268.566 |
  |   1 |               405.487 |      0.445475 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_21-16-08 |         405.487 |     0.55896 |              443.885 |       308.746 |
  |   2 |               444.303 |      0.424424 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/full/2023-12-20_23-49-22 |         444.303 |    0.538332 |              498.723 |       322.217 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.434949 | 0.445475 | 0.453769 |     3 | 0.462063 | 0.443987 | 0.424424 | 0.0188637 |
  | train/loss       |  378.874 |  405.487 |  424.895 |     3 |  444.303 |  400.684 |  352.262 |   46.2081 |
  | train/loss_epoch |  378.874 |  405.487 |  424.895 |     3 |  444.303 |  400.684 |  352.262 |   46.2081 |
  | train/loss_step  |  471.304 |  498.723 |  507.973 |     3 |  517.224 |  486.611 |  443.885 |     38.14 |
  | val/f1           | 0.548646 |  0.55896 | 0.565769 |     3 | 0.572577 | 0.556623 | 0.538332 | 0.0172421 |
  | val/loss         |  288.656 |  308.746 |  315.482 |     3 |  322.217 |  299.843 |  268.566 |   27.9116 |

### Coreference probing with normalized models - frozen RE + MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/90wfycu8
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/ky8y5j29
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/qwh90lq8

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss',) | ('train/loss_step',) | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                             | ('train/f1',) |
  | --: | --------------------: | --------------: | -------------------: | ------------: | ----------: | :---------------------------------------------------------------------------------------------- | ------------: |
  |   0 |               34.9048 |         34.9048 |             0.881507 |       50.4344 |    0.642202 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_13-56-57 |      0.677539 |
  |   1 |               31.5141 |         31.5141 |              14.9661 |       48.5743 |    0.649827 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_15-57-44 |      0.700362 |
  |   2 |               32.3751 |         32.3751 |              48.2201 |       49.5614 |    0.649116 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_18-33-21 |      0.692109 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.684824 | 0.692109 | 0.696235 |     3 | 0.700362 | 0.690003 | 0.677539 |  0.0115563 |
  | train/loss       |  31.9446 |  32.3751 |  33.6399 |     3 |  34.9048 |  32.9313 |  31.5141 |    1.76242 |
  | train/loss_epoch |  31.9446 |  32.3751 |  33.6399 |     3 |  34.9048 |  32.9313 |  31.5141 |    1.76242 |
  | train/loss_step  |  7.92378 |  14.9661 |  31.5931 |     3 |  48.2201 |  21.3559 | 0.881507 |    24.3076 |
  | val/f1           | 0.645659 | 0.649116 | 0.649471 |     3 | 0.649827 | 0.647048 | 0.642202 | 0.00421221 |
  | val/loss         |  49.0679 |  49.5614 |  49.9979 |     3 |  50.4344 |  49.5234 |  48.5743 |   0.930598 |

### Coreference probing with normalized models - frozen NER + MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/5mwe78kh
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/v1v0h4t1
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/tqom44xm

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('train/loss',) | ('model_save_dir',)                                                                             | ('train/loss_step',) | ('train/f1',) |
  | --: | --------------------: | ----------: | ------------: | --------------: | :---------------------------------------------------------------------------------------------- | -------------------: | ------------: |
  |   0 |               35.8827 |     0.64241 |       47.7307 |         35.8827 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_20-58-12 |              3.15069 |      0.663137 |
  |   1 |               30.5555 |    0.647934 |       49.7295 |         30.5555 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_00-22-35 |              2.03973 |      0.696437 |
  |   2 |               36.3826 |    0.643876 |       47.7822 |         36.3826 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_05-47-29 |             0.836549 |      0.663525 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.663331 | 0.663525 | 0.679981 |     3 | 0.696437 | 0.674366 | 0.663137 |   0.019115 |
  | train/loss       |  33.2191 |  35.8827 |  36.1326 |     3 |  36.3826 |  34.2736 |  30.5555 |    3.22966 |
  | train/loss_epoch |  33.2191 |  35.8827 |  36.1326 |     3 |  36.3826 |  34.2736 |  30.5555 |    3.22966 |
  | train/loss_step  |  1.43814 |  2.03973 |  2.59521 |     3 |  3.15069 |  2.00899 | 0.836549 |    1.15738 |
  | val/f1           | 0.643143 | 0.643876 | 0.645905 |     3 | 0.647934 |  0.64474 |  0.64241 | 0.00286161 |
  | val/loss         |  47.7565 |  47.7822 |  48.7559 |     3 |  49.7295 |  48.4142 |  47.7307 |    1.13942 |

### Coreference probing with normalized models - frozen RE + NER with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/uq3179h7
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/wks82y8x
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/8yf7lhhx

- metric values per seed

  |     | ('train/loss',) | ('train/f1',) | ('model_save_dir',)                                                                             | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) |
  | --: | --------------: | ------------: | :---------------------------------------------------------------------------------------------- | -------------------: | --------------------: | ----------: | ------------: |
  |   0 |         42.9581 |      0.608474 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_14-00-22 |              64.2648 |               42.9581 |    0.607504 |       56.0839 |
  |   1 |         44.3696 |      0.597458 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_16-58-41 |              40.5235 |               44.3696 |    0.601727 |       55.7457 |
  |   2 |         40.6626 |      0.622907 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_19-34-23 |              19.5882 |               40.6626 |     0.61124 |       53.3282 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.602966 | 0.608474 | 0.615691 |     3 | 0.622907 | 0.609613 | 0.597458 |  0.0127627 |
  | train/loss       |  41.8103 |  42.9581 |  43.6639 |     3 |  44.3696 |  42.6634 |  40.6626 |    1.87101 |
  | train/loss_epoch |  41.8103 |  42.9581 |  43.6639 |     3 |  44.3696 |  42.6634 |  40.6626 |    1.87101 |
  | train/loss_step  |  30.0559 |  40.5235 |  52.3942 |     3 |  64.2648 |  41.4588 |  19.5882 |     22.353 |
  | val/f1           | 0.604615 | 0.607504 | 0.609372 |     3 |  0.61124 | 0.606824 | 0.601727 | 0.00479305 |
  | val/loss         |   54.537 |  55.7457 |  55.9148 |     3 |  56.0839 |  55.0526 |  53.3282 |     1.5029 |

### Coreference probing with normalized models - frozen SQUAD + MRPC with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/fswbplx1
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/r7v74l4e
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/ze36g0ac

- metric values per seed

  |     | ('train/f1',) | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('model_save_dir',)                                                                             | ('train/loss_epoch',) | ('train/loss_step',) |
  | --: | ------------: | ------------: | ----------: | --------------: | :---------------------------------------------------------------------------------------------- | --------------------: | -------------------: |
  |   0 |      0.664647 |       47.3622 |    0.650683 |         37.4158 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_23-09-30 |               37.4158 |              10.8956 |
  |   1 |      0.688758 |       45.5599 |    0.656489 |         33.3172 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_01-36-05 |               33.3172 |              61.4666 |
  |   2 |       0.71524 |       45.8978 |    0.660097 |         29.1978 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_05-01-51 |               29.1978 |              116.525 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.676703 | 0.688758 | 0.701999 |     3 |  0.71524 | 0.689548 | 0.664647 |  0.0253057 |
  | train/loss       |  31.2575 |  33.3172 |  35.3665 |     3 |  37.4158 |  33.3103 |  29.1978 |      4.109 |
  | train/loss_epoch |  31.2575 |  33.3172 |  35.3665 |     3 |  37.4158 |  33.3103 |  29.1978 |      4.109 |
  | train/loss_step  |  36.1811 |  61.4666 |  88.9958 |     3 |  116.525 |  62.9624 |  10.8956 |    52.8305 |
  | val/f1           | 0.653586 | 0.656489 | 0.658293 |     3 | 0.660097 | 0.655756 | 0.650683 | 0.00474942 |
  | val/loss         |  45.7289 |  45.8978 |    46.63 |     3 |  47.3622 |  46.2733 |  45.5599 |   0.957993 |

### Coreference probing with normalized models - frozen RE + SQUAD with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/agxlo45l
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/nqwe93i2
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/4dd6oo7z

- metric values per seed

  |     | ('train/loss',) | ('val/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('train/f1',) | ('train/loss_step',) |
  | --: | --------------: | ----------: | ------------: | --------------------: | :---------------------------------------------------------------------------------------------- | ------------: | -------------------: |
  |   0 |         33.0433 |    0.638112 |       51.6936 |               33.0433 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_14-00-33 |      0.681233 |              80.5486 |
  |   1 |         36.3509 |    0.635255 |       51.0976 |               36.3509 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_19-15-37 |      0.663519 |              2.00565 |
  |   2 |         34.5975 |    0.632977 |       50.6487 |               34.5975 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-22_23-21-16 |      0.667958 |              26.6012 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.665739 | 0.667958 | 0.674595 |     3 | 0.681233 | 0.670903 | 0.663519 | 0.00921673 |
  | train/loss       |  33.8204 |  34.5975 |  35.4742 |     3 |  36.3509 |  34.6639 |  33.0433 |     1.6548 |
  | train/loss_epoch |  33.8204 |  34.5975 |  35.4742 |     3 |  36.3509 |  34.6639 |  33.0433 |     1.6548 |
  | train/loss_step  |  14.3034 |  26.6012 |  53.5749 |     3 |  80.5486 |  36.3852 |  2.00565 |    40.1752 |
  | val/f1           | 0.634116 | 0.635255 | 0.636684 |     3 | 0.638112 | 0.635448 | 0.632977 |  0.0025732 |
  | val/loss         |  50.8731 |  51.0976 |  51.3956 |     3 |  51.6936 |  51.1466 |  50.6487 |   0.524176 |

### Coreference probing with normalized models - frozen NER + SQUAD with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/normalized \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-normalized-training/runs/3lmofvwr
  - seed2: https://wandb.ai/tanikina/probing-normalized-training/runs/8x6fl83n
  - seed3: https://wandb.ai/tanikina/probing-normalized-training/runs/orvo4e6o

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                             | ('val/f1',) | ('train/loss',) | ('train/loss_step',) | ('train/f1',) | ('val/loss',) |
  | --: | --------------------: | :---------------------------------------------------------------------------------------------- | ----------: | --------------: | -------------------: | ------------: | ------------: |
  |   0 |                 48.38 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_03-59-28 |    0.603364 |           48.38 |              12.2246 |      0.568064 |       58.9313 |
  |   1 |               45.4263 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_07-23-05 |    0.612896 |         45.4263 |               2.6594 |      0.590989 |       52.2922 |
  |   2 |               43.3212 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/normalized/2023-12-23_11-50-16 |    0.617978 |         43.3212 |              8.24121 |      0.603505 |       56.1185 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.579526 | 0.590989 | 0.597247 |     3 | 0.603505 | 0.587519 | 0.568064 |  0.0179734 |
  | train/loss       |  44.3738 |  45.4263 |  46.9032 |     3 |    48.38 |  45.7092 |  43.3212 |    2.54126 |
  | train/loss_epoch |  44.3738 |  45.4263 |  46.9032 |     3 |    48.38 |  45.7092 |  43.3212 |    2.54126 |
  | train/loss_step  |   5.4503 |  8.24121 |  10.2329 |     3 |  12.2246 |  7.70842 |   2.6594 |    4.80483 |
  | val/f1           |  0.60813 | 0.612896 | 0.615437 |     3 | 0.617978 | 0.611413 | 0.603364 | 0.00741882 |
  | val/loss         |  54.2053 |  56.1185 |  57.5249 |     3 |  58.9313 |  55.7807 |  52.2922 |    3.33242 |

### Coreference probing with truncation - frozen RE (9 layer) + MRPC (9 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/o6z7ook1
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/v5ylfnx7
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/8v63p9fn

- metric values per seed

  |     | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss',) | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                            |
  | --: | ----------: | --------------------: | ------------: | --------------: | -------------------: | ------------: | :--------------------------------------------------------------------------------------------- |
  |   0 |    0.653426 |               355.719 |       830.336 |         355.719 |              647.922 |      0.734427 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_10-49-27 |
  |   1 |    0.642542 |               416.599 |       818.378 |         416.599 |              951.369 |      0.709001 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_13-10-27 |
  |   2 |    0.638729 |               444.882 |       828.402 |         444.882 |              534.458 |       0.69305 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_15-06-22 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.701025 | 0.709001 | 0.721714 |     3 | 0.734427 | 0.712159 |  0.69305 | 0.0208688 |
  | train/loss       |  386.159 |  416.599 |   430.74 |     3 |  444.882 |  405.733 |  355.719 |   45.5635 |
  | train/loss_epoch |  386.159 |  416.599 |   430.74 |     3 |  444.882 |  405.733 |  355.719 |   45.5635 |
  | train/loss_step  |   591.19 |  647.922 |  799.646 |     3 |  951.369 |   711.25 |  534.458 |    215.55 |
  | val/f1           | 0.640636 | 0.642542 | 0.647984 |     3 | 0.653426 | 0.644899 | 0.638729 | 0.0076269 |
  | val/loss         |   823.39 |  828.402 |  829.369 |     3 |  830.336 |  825.705 |  818.378 |   6.41853 |

### Coreference probing with truncation - frozen NER (6 layer) + MRPC (9 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/eqinqrx9
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/59kcuyn0
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/owoe35sc

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                            | ('val/loss',) |
  | --: | ----------: | ------------: | --------------------: | --------------: | -------------------: | :--------------------------------------------------------------------------------------------- | ------------: |
  |   0 |     0.65907 |      0.749392 |               288.668 |         288.668 |              238.245 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_16-49-05 |       773.454 |
  |   1 |    0.654752 |      0.722616 |               346.553 |         346.553 |              56.4536 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_19-27-18 |       791.318 |
  |   2 |    0.657461 |      0.734598 |               319.171 |         319.171 |              263.491 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_21-36-23 |       733.647 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.728607 | 0.734598 | 0.741995 |     3 | 0.749392 | 0.735535 | 0.722616 |  0.0134122 |
  | train/loss       |   303.92 |  319.171 |  332.862 |     3 |  346.553 |  318.131 |  288.668 |    28.9564 |
  | train/loss_epoch |   303.92 |  319.171 |  332.862 |     3 |  346.553 |  318.131 |  288.668 |    28.9564 |
  | train/loss_step  |  147.349 |  238.245 |  250.868 |     3 |  263.491 |  186.063 |  56.4536 |    112.953 |
  | val/f1           | 0.656107 | 0.657461 | 0.658265 |     3 |  0.65907 | 0.657094 | 0.654752 | 0.00218194 |
  | val/loss         |  753.551 |  773.454 |  782.386 |     3 |  791.318 |   766.14 |  733.647 |    29.5229 |

### Coreference probing with truncation - frozen RE (9 layer) + NER (6 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/z22qhjeq
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/pcf9akam
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/u2wo9mbh

- metric values per seed

  |     | ('train/f1',) | ('val/loss',) | ('train/loss',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                            | ('train/loss_epoch',) |
  | --: | ------------: | ------------: | --------------: | ----------: | -------------------: | :--------------------------------------------------------------------------------------------- | --------------------: |
  |   0 |      0.682874 |       776.435 |         423.649 |    0.635409 |                    0 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_10-49-48 |               423.649 |
  |   1 |      0.776744 |       752.157 |         224.056 |     0.65673 |              341.383 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_12-35-29 |               224.056 |
  |   2 |      0.709111 |       780.673 |         365.174 |    0.639294 |              179.579 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_16-03-13 |               365.174 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.695992 | 0.709111 | 0.742928 |     3 | 0.776744 |  0.72291 | 0.682874 | 0.0484326 |
  | train/loss       |  294.615 |  365.174 |  394.412 |     3 |  423.649 |  337.626 |  224.056 |   102.609 |
  | train/loss_epoch |  294.615 |  365.174 |  394.412 |     3 |  423.649 |  337.626 |  224.056 |   102.609 |
  | train/loss_step  |  89.7895 |  179.579 |  260.481 |     3 |  341.383 |  173.654 |        0 |   170.768 |
  | val/f1           | 0.637352 | 0.639294 | 0.648012 |     3 |  0.65673 | 0.643811 | 0.635409 | 0.0113556 |
  | val/loss         |  764.296 |  776.435 |  778.554 |     3 |  780.673 |  769.755 |  752.157 |   15.3871 |

### Coreference probing with truncation - frozen SQUAD (8 layer) + MRPC (9 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-mrpc] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/0o5ly4km
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/gt5vhpkn
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/f4waymqr

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/loss',) | ('train/loss_step',) | ('val/f1',) | ('train/f1',) | ('train/loss',) | ('model_save_dir',)                                                                            |
  | --: | --------------------: | ------------: | -------------------: | ----------: | ------------: | --------------: | :--------------------------------------------------------------------------------------------- |
  |   0 |                167.64 |       837.935 |              15.2123 |    0.677846 |      0.825885 |          167.64 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_18-13-52 |
  |   1 |               430.812 |       780.812 |              197.214 |    0.641001 |      0.700938 |         430.812 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_22-05-39 |
  |   2 |               236.266 |       826.258 |                    0 |    0.674853 |      0.791569 |         236.266 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_23-29-53 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.746254 | 0.791569 | 0.808727 |     3 | 0.825885 | 0.772798 | 0.700938 | 0.0645539 |
  | train/loss       |  201.953 |  236.266 |  333.539 |     3 |  430.812 |  278.239 |   167.64 |   136.514 |
  | train/loss_epoch |  201.953 |  236.266 |  333.539 |     3 |  430.812 |  278.239 |   167.64 |   136.514 |
  | train/loss_step  |  7.60616 |  15.2123 |  106.213 |     3 |  197.214 |  70.8087 |        0 |   109.734 |
  | val/f1           | 0.657927 | 0.674853 |  0.67635 |     3 | 0.677846 | 0.664567 | 0.641001 | 0.0204633 |
  | val/loss         |  803.535 |  826.258 |  832.097 |     3 |  837.935 |  815.002 |  780.812 |   30.1797 |

### Coreference probing with truncation - frozen RE (9 layer) + SQUAD (8 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/tp2u3j6z
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/o1ur3i7x
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/sgpr89a2

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('val/loss',) | ('model_save_dir',)                                                                            | ('train/loss',) |
  | --: | ----------: | ------------: | --------------------: | -------------------: | ------------: | :--------------------------------------------------------------------------------------------- | --------------: |
  |   0 |    0.658159 |      0.798095 |               212.202 |                    0 |       838.025 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_10-49-54 |         212.202 |
  |   1 |    0.652182 |      0.745543 |               320.706 |              202.542 |       829.979 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_14-14-52 |         320.706 |
  |   2 |    0.664411 |      0.810706 |                181.91 |              196.326 |       834.038 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_16-32-08 |          181.91 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.771819 | 0.798095 |   0.8044 |     3 | 0.810706 | 0.784781 | 0.745543 | 0.0345613 |
  | train/loss       |  197.056 |  212.202 |  266.454 |     3 |  320.706 |  238.273 |   181.91 |   72.9783 |
  | train/loss_epoch |  197.056 |  212.202 |  266.454 |     3 |  320.706 |  238.273 |   181.91 |   72.9783 |
  | train/loss_step  |  98.1632 |  196.326 |  199.434 |     3 |  202.542 |  132.956 |        0 |   115.185 |
  | val/f1           | 0.655171 | 0.658159 | 0.661285 |     3 | 0.664411 | 0.658251 | 0.652182 | 0.0061151 |
  | val/loss         |  832.009 |  834.038 |  836.032 |     3 |  838.025 |  834.014 |  829.979 |   4.02303 |

### Coreference probing with truncation - frozen NER (6 layer) + SQUAD (8 layer) with mean aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=mean \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=probing/truncated \
  --multirun


  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/probing-truncated-training/runs/4dc1l4px
  - seed2: https://wandb.ai/tanikina/probing-truncated-training/runs/b6qu6zju
  - seed3: https://wandb.ai/tanikina/probing-truncated-training/runs/sdtiqdwz

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('train/loss_step',) | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('model_save_dir',)                                                                            |
  | --: | --------------------: | ----------: | -------------------: | ------------: | ------------: | --------------: | :--------------------------------------------------------------------------------------------- |
  |   0 |               321.532 |    0.652725 |           0.00154322 |       800.052 |      0.739511 |         321.532 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_20-14-52 |
  |   1 |               270.755 |    0.656032 |              651.098 |       796.581 |      0.763133 |         270.755 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-20_22-28-18 |
  |   2 |               375.434 |    0.643005 |              1367.87 |       802.349 |      0.712924 |         375.434 | /netscratch/anikina/multi-task-knowledge-transfer/models/probing/truncated/2023-12-21_01-05-51 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         | 0.726218 | 0.739511 | 0.751322 |     3 | 0.763133 | 0.738523 |   0.712924 |   0.025119 |
  | train/loss       |  296.143 |  321.532 |  348.483 |     3 |  375.434 |  322.574 |    270.755 |    52.3474 |
  | train/loss_epoch |  296.143 |  321.532 |  348.483 |     3 |  375.434 |  322.574 |    270.755 |    52.3474 |
  | train/loss_step  |   325.55 |  651.098 |  1009.49 |     3 |  1367.87 |  672.991 | 0.00154322 |    684.198 |
  | val/f1           | 0.647865 | 0.652725 | 0.654378 |     3 | 0.656032 | 0.650587 |   0.643005 | 0.00677144 |
  | val/loss         |  798.316 |  800.052 |  801.201 |     3 |  802.349 |  799.661 |    796.581 |    2.90397 |

## 2024-01-06

### Coreference - frozen MRPC with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/nzw1m1ug
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/9bfeiaqw
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/4bqpfx70

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                                |
  | --: | -------------------: | --------------: | --------------------: | ------------: | ------------: | ----------: | :----------------------------------------------------------------------------------------------------------------- |
  |   0 |              3.41951 |         32.2359 |               32.2359 |      0.741212 |       64.3413 |    0.665421 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_11-14-19 |
  |   1 |             0.935967 |         25.7491 |               25.7491 |      0.774975 |       65.2121 |    0.667491 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_13-17-55 |
  |   2 |              101.437 |         22.1691 |               22.1691 |      0.796462 |       66.5511 |    0.678481 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_16-05-33 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.758094 | 0.774975 | 0.785719 |     3 | 0.796462 | 0.770883 | 0.741212 | 0.0278514 |
  | train/loss       |  23.9591 |  25.7491 |  28.9925 |     3 |  32.2359 |   26.718 |  22.1691 |   5.10287 |
  | train/loss_epoch |  23.9591 |  25.7491 |  28.9925 |     3 |  32.2359 |   26.718 |  22.1691 |   5.10287 |
  | train/loss_step  |  2.17774 |  3.41951 |  52.4281 |     3 |  101.437 |   35.264 | 0.935967 |   57.3206 |
  | val/f1           | 0.666456 | 0.667491 | 0.672986 |     3 | 0.678481 | 0.670464 | 0.665421 | 0.0070194 |
  | val/loss         |  64.7767 |  65.2121 |  65.8816 |     3 |  66.5511 |  65.3682 |  64.3413 |   1.11313 |

### Coreference - frozen NER with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/ozmo93ri
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/gtkn6w4s
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/s0n5uxk8

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                                | ('train/f1',) | ('train/loss_step',) | ('train/loss',) |
  | --: | --------------------: | ------------: | ----------: | :----------------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | --------------: |
  |   0 |               76.2978 |       81.4954 |    0.539286 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_19-28-27 |      0.426958 |              284.368 |         76.2978 |
  |   1 |               62.8783 |        80.127 |    0.562365 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_21-18-33 |      0.495079 |              4.34592 |         62.8783 |
  |   2 |               68.9741 |       82.3302 |    0.557437 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-08_00-58-29 |      0.467591 |              3.27873 |         68.9741 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.447274 | 0.467591 | 0.481335 |     3 | 0.495079 | 0.463209 | 0.426958 | 0.0342716 |
  | train/loss       |  65.9262 |  68.9741 |  72.6359 |     3 |  76.2978 |  69.3834 |  62.8783 |   6.71912 |
  | train/loss_epoch |  65.9262 |  68.9741 |  72.6359 |     3 |  76.2978 |  69.3834 |  62.8783 |   6.71912 |
  | train/loss_step  |  3.81233 |  4.34592 |  144.357 |     3 |  284.368 |  97.3309 |  3.27873 |    161.98 |
  | val/f1           | 0.548362 | 0.557437 | 0.559901 |     3 | 0.562365 | 0.553029 | 0.539286 | 0.0121541 |
  | val/loss         |  80.8112 |  81.4954 |  81.9128 |     3 |  82.3302 |  81.3176 |   80.127 |   1.11233 |

### Coreference - frozen RE with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/4q8zhrcm
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/7e8qwxg8
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/ocmc8v8h

- metric values per seed

  |     | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                                | ('train/f1',) | ('train/loss_step',) | ('train/loss',) |
  | --: | ------------: | --------------------: | ----------: | :----------------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | --------------: |
  |   0 |       83.8753 |               49.0233 |    0.605628 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_11-16-10 |      0.634703 |              37.0837 |         49.0233 |
  |   1 |       81.1608 |               40.5259 |    0.615226 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_14-44-48 |      0.671783 |              13.9335 |         40.5259 |
  |   2 |       81.9309 |               46.2303 |    0.608145 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_19-10-16 |        0.6477 |              8.63813 |         46.2303 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.641201 |   0.6477 | 0.659741 |     3 | 0.671783 | 0.651395 | 0.634703 |  0.0188142 |
  | train/loss       |  43.3781 |  46.2303 |  47.6268 |     3 |  49.0233 |  45.2599 |  40.5259 |    4.33101 |
  | train/loss_epoch |  43.3781 |  46.2303 |  47.6268 |     3 |  49.0233 |  45.2599 |  40.5259 |    4.33101 |
  | train/loss_step  |  11.2858 |  13.9335 |  25.5086 |     3 |  37.0837 |  19.8851 |  8.63813 |    15.1279 |
  | val/f1           | 0.606886 | 0.608145 | 0.611686 |     3 | 0.615226 | 0.609666 | 0.605628 | 0.00497678 |
  | val/loss         |  81.5458 |  81.9309 |  82.9031 |     3 |  83.8753 |  82.3223 |  81.1608 |    1.39893 |

### Coreference - frozen SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/0xxwy8wj
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/n8mnumii
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/jpvj64ev

- metric values per seed

  |     | ('val/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                                | ('train/loss_step',) |
  | --: | ------------: | ------------: | --------------------: | --------------: | ----------: | :----------------------------------------------------------------------------------------------------------------- | -------------------: |
  |   0 |       73.7279 |      0.579496 |               58.0829 |         58.0829 |    0.613415 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_22-56-52 |              121.785 |
  |   1 |       75.6886 |      0.543413 |               67.6542 |         67.6542 |     0.59793 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-08_01-28-59 |              93.4739 |
  |   2 |       81.2461 |      0.524804 |               72.4197 |         72.4197 |     0.58317 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-08_03-09-16 |              37.5403 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.534109 | 0.543413 | 0.561454 |     3 | 0.579496 | 0.549238 | 0.524804 |  0.027807 |
  | train/loss       |  62.8685 |  67.6542 |  70.0369 |     3 |  72.4197 |  66.0523 |  58.0829 |    7.3014 |
  | train/loss_epoch |  62.8685 |  67.6542 |  70.0369 |     3 |  72.4197 |  66.0523 |  58.0829 |    7.3014 |
  | train/loss_step  |  65.5071 |  93.4739 |  107.629 |     3 |  121.785 |  84.2664 |  37.5403 |   42.8704 |
  | val/f1           |  0.59055 |  0.59793 | 0.605673 |     3 | 0.613415 | 0.598172 |  0.58317 | 0.0151241 |
  | val/loss         |  74.7083 |  75.6886 |  78.4673 |     3 |  81.2461 |  76.8875 |  73.7279 |   3.89982 |

### Coreference - frozen BERT with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/yk8pj8c2
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/dnjprmrw
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/pqjyib4y

- metric values per seed

  |     | ('val/f1',) | ('train/f1',) | ('train/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                | ('train/loss_step',) | ('val/loss',) |
  | --: | ----------: | ------------: | --------------: | --------------------: | :----------------------------------------------------------------------------------------------------------------- | -------------------: | ------------: |
  |   0 |    0.667162 |      0.810322 |         21.6584 |               21.6584 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_11-16-19 |              33.6045 |        80.201 |
  |   1 |    0.670518 |      0.831018 |         17.8993 |               17.8993 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_14-06-28 |             0.642166 |       81.8435 |
  |   2 |    0.676837 |      0.891606 |            9.97 |                  9.97 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_17-31-41 |          0.000350801 |        89.659 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |  0.82067 | 0.831018 | 0.861312 |     3 | 0.891606 | 0.844315 |    0.810322 |  0.0422416 |
  | train/loss       |  13.9347 |  17.8993 |  19.7789 |     3 |  21.6584 |  16.5092 |        9.97 |    5.96691 |
  | train/loss_epoch |  13.9347 |  17.8993 |  19.7789 |     3 |  21.6584 |  16.5092 |        9.97 |    5.96691 |
  | train/loss_step  | 0.321258 | 0.642166 |  17.1233 |     3 |  33.6045 |  11.4157 | 0.000350801 |    19.2188 |
  | val/f1           |  0.66884 | 0.670518 | 0.673677 |     3 | 0.676837 | 0.671506 |    0.667162 | 0.00491249 |
  | val/loss         |  81.0222 |  81.8435 |  85.7513 |     3 |   89.659 |  83.9012 |      80.201 |    5.05364 |

### Coreference - frozen target with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/80erm6hc
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/ycvrzqju
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models-training/runs/dthg0zpc

- metric values per seed

  |     | ('val/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('val/loss',) |
  | --: | ----------: | --------------------: | :----------------------------------------------------------------------------------------------------------------- | ------------: | --------------: | -------------------: | ------------: |
  |   0 |    0.736999 |               5.46166 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-07_23-16-15 |      0.950547 |         5.46166 |                    0 |       101.311 |
  |   1 |    0.732465 |               6.89532 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-08_01-39-46 |      0.938979 |         6.89532 |              4.11078 |       88.3074 |
  |   2 |    0.738434 |               5.61812 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models/2024-01-08_03-20-04 |       0.94879 |         5.61812 |            0.0167841 |       102.137 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |   0.943884 |   0.94879 | 0.949668 |     3 | 0.950547 | 0.946105 | 0.938979 | 0.00623376 |
  | train/loss       |    5.53989 |   5.61812 |  6.25672 |     3 |  6.89532 |   5.9917 |  5.46166 |   0.786456 |
  | train/loss_epoch |    5.53989 |   5.61812 |  6.25672 |     3 |  6.89532 |   5.9917 |  5.46166 |   0.786456 |
  | train/loss_step  | 0.00839203 | 0.0167841 |  2.06378 |     3 |  4.11078 |  1.37586 |        0 |    2.36853 |
  | val/f1           |   0.734732 |  0.736999 | 0.737716 |     3 | 0.738434 | 0.735966 | 0.732465 | 0.00311543 |
  | val/loss         |    94.8092 |   101.311 |  101.724 |     3 |  102.137 |  97.2517 |  88.3074 |    7.75695 |

### Coreference - frozen RE + MRPC with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/ie5ft1bn
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/1a62yti5
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/rn8bvv93

- metric values per seed

  |     | ('train/loss_epoch',) | ('model_save_dir',)                                                                                            | ('val/f1',) | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('val/loss',) |
  | --: | --------------------: | :------------------------------------------------------------------------------------------------------------- | ----------: | ------------: | --------------: | -------------------: | ------------: |
  |   0 |               23.9768 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_13-43-24 |    0.675357 |      0.783643 |         23.9768 |              1.03919 |       64.0853 |
  |   1 |               21.1517 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_16-13-03 |    0.679287 |      0.804843 |         21.1517 |           0.00732524 |       67.7894 |
  |   2 |                24.417 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_19-13-02 |    0.672051 |      0.782619 |          24.417 |              29.3068 |       64.1329 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         | 0.783131 | 0.783643 | 0.794243 |     3 | 0.804843 | 0.790368 |   0.782619 |  0.0125459 |
  | train/loss       |  22.5642 |  23.9768 |  24.1969 |     3 |   24.417 |  23.1818 |    21.1517 |    1.77188 |
  | train/loss_epoch |  22.5642 |  23.9768 |  24.1969 |     3 |   24.417 |  23.1818 |    21.1517 |    1.77188 |
  | train/loss_step  | 0.523256 |  1.03919 |   15.173 |     3 |  29.3068 |  10.1178 | 0.00732524 |    16.6262 |
  | val/f1           | 0.673704 | 0.675357 | 0.677322 |     3 | 0.679287 | 0.675565 |   0.672051 | 0.00362263 |
  | val/loss         |  64.1091 |  64.1329 |  65.9611 |     3 |  67.7894 |  65.3359 |    64.0853 |    2.12498 |

### Coreference - frozen NER + MRPC with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/31vij235
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/ohvtin3r
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/7ehpiyl7

- metric values per seed

  |     | ('train/loss_step',) | ('val/f1',) | ('train/loss',) | ('val/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                            | ('train/f1',) |
  | --: | -------------------: | ----------: | --------------: | ------------: | --------------------: | :------------------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |              16.3036 |    0.687229 |         17.1542 |       72.0522 |               17.1542 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_21-43-31 |      0.836006 |
  |   1 |              54.7232 |    0.671638 |         31.8171 |       58.8755 |               31.8171 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-07_02-13-00 |      0.738292 |
  |   2 |               6.3631 |    0.679305 |         26.0934 |       60.3677 |               26.0934 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-07_04-04-29 |       0.77184 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.755066 |  0.77184 | 0.803923 |     3 | 0.836006 | 0.782046 | 0.738292 |  0.0496503 |
  | train/loss       |  21.6238 |  26.0934 |  28.9552 |     3 |  31.8171 |  25.0216 |  17.1542 |    7.38998 |
  | train/loss_epoch |  21.6238 |  26.0934 |  28.9552 |     3 |  31.8171 |  25.0216 |  17.1542 |    7.38998 |
  | train/loss_step  |  11.3333 |  16.3036 |  35.5134 |     3 |  54.7232 |  25.7966 |   6.3631 |    25.5395 |
  | val/f1           | 0.675471 | 0.679305 | 0.683267 |     3 | 0.687229 |  0.67939 | 0.671638 | 0.00779582 |
  | val/loss         |  59.6216 |  60.3677 |    66.21 |     3 |  72.0522 |  63.7652 |  58.8755 |     7.2155 |

### Coreference - frozen RE + NER with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/37nqee3s
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/r5uw4gus
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/63eyuupu

- metric values per seed

  |     | ('model_save_dir',)                                                                                            | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss_step',) | ('train/loss',) | ('val/loss',) |
  | --: | :------------------------------------------------------------------------------------------------------------- | ------------: | --------------------: | ----------: | -------------------: | --------------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_13-47-37 |       0.71863 |               31.7609 |    0.648589 |              7.65745 |         31.7609 |       68.0938 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_17-43-34 |      0.664473 |               41.0356 |    0.635183 |              20.4089 |         41.0356 |       68.8829 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_20-16-33 |      0.663055 |               41.5256 |    0.636275 |              68.9053 |         41.5256 |       67.4384 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.663764 | 0.664473 | 0.691551 |     3 |  0.71863 | 0.682052 | 0.663055 |  0.0316848 |
  | train/loss       |  36.3982 |  41.0356 |  41.2806 |     3 |  41.5256 |  38.1074 |  31.7609 |    5.50166 |
  | train/loss_epoch |  36.3982 |  41.0356 |  41.2806 |     3 |  41.5256 |  38.1074 |  31.7609 |    5.50166 |
  | train/loss_step  |  14.0332 |  20.4089 |  44.6571 |     3 |  68.9053 |  32.3239 |  7.65745 |    32.3156 |
  | val/f1           | 0.635729 | 0.636275 | 0.642432 |     3 | 0.648589 | 0.640016 | 0.635183 | 0.00744489 |
  | val/loss         |  67.7661 |  68.0938 |  68.4883 |     3 |  68.8829 |  68.1384 |  67.4384 |   0.723314 |

### Coreference - frozen SQUAD + MRPC with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/qx7iiueq
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/5dkaa37r
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/7gh6rzft

- metric values per seed

  |     | ('train/f1',) | ('train/loss_step',) | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                            | ('train/loss',) |
  | --: | ------------: | -------------------: | ------------: | --------------------: | ----------: | :------------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |      0.793081 |              7.53344 |       62.6384 |               22.7956 |    0.681983 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_22-44-47 |         22.7956 |
  |   1 |      0.792643 |             0.186432 |        61.501 |               22.8062 |    0.685848 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-07_01-37-07 |         22.8062 |
  |   2 |      0.732082 |              28.7173 |       58.5502 |               32.5181 |    0.668359 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-07_04-31-07 |         32.5181 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.762363 | 0.792643 | 0.792862 |     3 | 0.793081 | 0.772602 | 0.732082 |  0.0350918 |
  | train/loss       |  22.8009 |  22.8062 |  27.6622 |     3 |  32.5181 |    26.04 |  22.7956 |    5.61022 |
  | train/loss_epoch |  22.8009 |  22.8062 |  27.6622 |     3 |  32.5181 |    26.04 |  22.7956 |    5.61022 |
  | train/loss_step  |  3.85993 |  7.53344 |  18.1254 |     3 |  28.7173 |  12.1457 | 0.186432 |    14.8141 |
  | val/f1           | 0.675171 | 0.681983 | 0.683916 |     3 | 0.685848 |  0.67873 | 0.668359 | 0.00918734 |
  | val/loss         |  60.0256 |   61.501 |  62.0697 |     3 |  62.6384 |  60.8965 |  58.5502 |    2.11008 |

### Coreference - frozen RE + SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/zcdf17ua
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/ucsi02o5
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/7wltgu9d

- metric values per seed

  |     | ('val/f1',) | ('train/loss',) | ('model_save_dir',)                                                                                            | ('val/loss',) | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) |
  | --: | ----------: | --------------: | :------------------------------------------------------------------------------------------------------------- | ------------: | --------------------: | -------------------: | ------------: |
  |   0 |    0.645813 |         39.6262 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_13-47-59 |       65.8589 |               39.6262 |              3.06952 |      0.680157 |
  |   1 |    0.651678 |         34.3388 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_16-16-39 |       66.3391 |               34.3388 |              98.8803 |      0.713249 |
  |   2 |    0.653101 |         29.9178 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_19-34-54 |        67.589 |               29.9178 |              68.8341 |      0.736927 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.696703 | 0.713249 | 0.725088 |     3 | 0.736927 | 0.710111 | 0.680157 |   0.028515 |
  | train/loss       |  32.1283 |  34.3388 |  36.9825 |     3 |  39.6262 |  34.6276 |  29.9178 |    4.86062 |
  | train/loss_epoch |  32.1283 |  34.3388 |  36.9825 |     3 |  39.6262 |  34.6276 |  29.9178 |    4.86062 |
  | train/loss_step  |  35.9518 |  68.8341 |  83.8572 |     3 |  98.8803 |   56.928 |  3.06952 |    49.0025 |
  | val/f1           | 0.648746 | 0.651678 |  0.65239 |     3 | 0.653101 | 0.650197 | 0.645813 | 0.00386294 |
  | val/loss         |   66.099 |  66.3391 |   66.964 |     3 |   67.589 |  66.5956 |  65.8589 |    0.89312 |

### Coreference - frozen NER + SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/wdlcn6mh
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/ib0x3ph3
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models-training/runs/pewy83zs

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('model_save_dir',)                                                                                            |
  | --: | ------------: | ----------: | --------------: | ------------: | --------------------: | -------------------: | :------------------------------------------------------------------------------------------------------------- |
  |   0 |       62.9229 |    0.647197 |         42.4919 |       0.65866 |               42.4919 |              15.1851 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-06_23-37-27 |
  |   1 |       62.2626 |    0.655449 |         40.3372 |      0.673115 |               40.3372 |              49.3964 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-07_02-26-08 |
  |   2 |       61.1105 |    0.647859 |         45.1022 |       0.64569 |               45.1022 |              37.5785 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models/2024-01-07_05-39-43 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.652175 |  0.65866 | 0.665887 |     3 | 0.673115 | 0.659155 |  0.64569 |  0.0137191 |
  | train/loss       |  41.4146 |  42.4919 |  43.7971 |     3 |  45.1022 |  42.6438 |  40.3372 |    2.38614 |
  | train/loss_epoch |  41.4146 |  42.4919 |  43.7971 |     3 |  45.1022 |  42.6438 |  40.3372 |    2.38614 |
  | train/loss_step  |  26.3818 |  37.5785 |  43.4874 |     3 |  49.3964 |  34.0533 |  15.1851 |    17.3759 |
  | val/f1           | 0.647528 | 0.647859 | 0.651654 |     3 | 0.655449 | 0.650168 | 0.647197 | 0.00458479 |
  | val/loss         |  61.6865 |  62.2626 |  62.5927 |     3 |  62.9229 |  62.0986 |  61.1105 |   0.917232 |

### Coreference - frozen RE + NER + SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/k8zc4mtn
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/vs5bnk0x
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/f1xln525

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                                  | ('train/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('train/loss_step',) |
  | --: | ----------: | ------------: | :------------------------------------------------------------------------------------------------------------------- | --------------: | --------------------: | ------------: | -------------------: |
  |   0 |     0.66245 |       61.6558 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_10-48-14 |         32.6731 |               32.6731 |       0.72422 |              17.6677 |
  |   1 |    0.663841 |       61.8353 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_13-16-10 |         29.8727 |               29.8727 |      0.740698 |               22.052 |
  |   2 |    0.660998 |       59.7005 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_16-05-45 |           31.79 |                 31.79 |      0.726336 |              12.4901 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.725278 | 0.726336 | 0.733517 |     3 | 0.740698 | 0.730418 |  0.72422 | 0.00896504 |
  | train/loss       |  30.8313 |    31.79 |  32.2315 |     3 |  32.6731 |  31.4453 |  29.8727 |    1.43169 |
  | train/loss_epoch |  30.8313 |    31.79 |  32.2315 |     3 |  32.6731 |  31.4453 |  29.8727 |    1.43169 |
  | train/loss_step  |  15.0789 |  17.6677 |  19.8598 |     3 |   22.052 |  17.4032 |  12.4901 |    4.78642 |
  | val/f1           | 0.661724 |  0.66245 | 0.663146 |     3 | 0.663841 |  0.66243 | 0.660998 | 0.00142193 |
  | val/loss         |  60.6782 |  61.6558 |  61.7456 |     3 |  61.8353 |  61.0639 |  59.7005 |    1.18413 |

### Coreference - frozen MRPC + NER + SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/tfalm9h6
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/xfnrz6s5
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/gwgw7lrv

- metric values per seed

  |     | ('model_save_dir',)                                                                                                  | ('val/loss',) | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) |
  | --: | :------------------------------------------------------------------------------------------------------------------- | ------------: | --------------: | -------------------: | --------------------: | ------------: | ----------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_10-51-10 |       60.8529 |         18.8118 |              24.2118 |               18.8118 |      0.822289 |    0.690144 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_13-59-32 |        64.131 |         16.0427 |              52.9013 |               16.0427 |      0.846905 |    0.692877 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_18-02-48 |       63.6926 |         18.6032 |              15.7336 |               18.6032 |      0.826578 |     0.69335 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.824433 | 0.826578 | 0.836741 |     3 | 0.846905 | 0.831924 | 0.822289 |  0.0131499 |
  | train/loss       |  17.3229 |  18.6032 |  18.7075 |     3 |  18.8118 |  17.8192 |  16.0427 |    1.54205 |
  | train/loss_epoch |  17.3229 |  18.6032 |  18.7075 |     3 |  18.8118 |  17.8192 |  16.0427 |    1.54205 |
  | train/loss_step  |  19.9727 |  24.2118 |  38.5565 |     3 |  52.9013 |  30.9489 |  15.7336 |    19.4782 |
  | val/f1           |  0.69151 | 0.692877 | 0.693113 |     3 |  0.69335 | 0.692123 | 0.690144 | 0.00173043 |
  | val/loss         |  62.2727 |  63.6926 |  63.9118 |     3 |   64.131 |  62.8922 |  60.8529 |    1.77962 |

### Coreference - frozen MRPC + RE + SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/bcyc7p0k
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/nh3vykwm
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/khjth29j

- metric values per seed

  |     | ('train/f1',) | ('val/f1',) | ('model_save_dir',)                                                                                                  | ('val/loss',) | ('train/loss_epoch',) | ('train/loss',) | ('train/loss_step',) |
  | --: | ------------: | ----------: | :------------------------------------------------------------------------------------------------------------------- | ------------: | --------------------: | --------------: | -------------------: |
  |   0 |       0.86093 |    0.692994 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-07_21-17-21 |       70.7726 |                13.897 |          13.897 |              25.9972 |
  |   1 |      0.796114 |    0.681317 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-08_01-29-28 |        63.168 |                21.678 |          21.678 |             0.251339 |
  |   2 |      0.803076 |    0.684615 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-08_03-48-40 |         60.64 |               20.8543 |         20.8543 |              3.94335 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.799595 | 0.803076 | 0.832003 |     3 |  0.86093 |  0.82004 | 0.796114 |  0.0355822 |
  | train/loss       |  17.3756 |  20.8543 |  21.2661 |     3 |   21.678 |  18.8098 |   13.897 |    4.27447 |
  | train/loss_epoch |  17.3756 |  20.8543 |  21.2661 |     3 |   21.678 |  18.8098 |   13.897 |    4.27447 |
  | train/loss_step  |  2.09734 |  3.94335 |  14.9703 |     3 |  25.9972 |   10.064 | 0.251339 |    13.9215 |
  | val/f1           | 0.682966 | 0.684615 | 0.688805 |     3 | 0.692994 | 0.686309 | 0.681317 | 0.00601987 |
  | val/loss         |   61.904 |   63.168 |  66.9703 |     3 |  70.7726 |  64.8602 |    60.64 |    5.27396 |

### Coreference - frozen MRPC + RE + NER with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/9mpbqe88
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/09hyc34q
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/866tb9lm

- metric values per seed

  |     | ('train/loss',) | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                                                  | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) |
  | --: | --------------: | -------------------: | ------------: | :------------------------------------------------------------------------------------------------------------------- | ----------: | --------------------: | ------------: |
  |   0 |         21.9038 |              29.8347 |      0.798421 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-10_22-10-51 |    0.681858 |               21.9038 |       62.6348 |
  |   1 |         15.7998 |              3.37657 |      0.837863 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-11_00-31-51 |    0.683746 |               15.7998 |       67.6258 |
  |   2 |         19.0376 |              0.24495 |      0.815134 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-11_04-00-31 |    0.683543 |               19.0376 |       64.5228 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.806777 | 0.815134 | 0.826498 |     3 | 0.837863 | 0.817139 | 0.798421 |  0.0197974 |
  | train/loss       |  17.4187 |  19.0376 |  20.4707 |     3 |  21.9038 |  18.9138 |  15.7998 |    3.05389 |
  | train/loss_epoch |  17.4187 |  19.0376 |  20.4707 |     3 |  21.9038 |  18.9138 |  15.7998 |    3.05389 |
  | train/loss_step  |  1.81076 |  3.37657 |  16.6056 |     3 |  29.8347 |  11.1521 |  0.24495 |    16.2552 |
  | val/f1           | 0.682701 | 0.683543 | 0.683645 |     3 | 0.683746 | 0.683049 | 0.681858 | 0.00103655 |
  | val/loss         |  63.5788 |  64.5228 |  66.0743 |     3 |  67.6258 |  64.9278 |  62.6348 |    2.52002 |

### Coreference - frozen MRPC + RE + NER + SQUAD with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/all_models \
  --multirun

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/jvrjx5qf
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/c5wt1dlm
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/yayhru16

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) | ('train/loss',) | ('val/loss',) | ('model_save_dir',)                                                                                             | ('val/f1',) |
  | --: | -------------------: | ------------: | --------------------: | --------------: | ------------: | :-------------------------------------------------------------------------------------------------------------- | ----------: |
  |   0 |               2.0451 |      0.808424 |               20.4268 |         20.4268 |       59.1151 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_05-40-49 |    0.686884 |
  |   1 |            0.0169539 |      0.867842 |               12.2712 |         12.2712 |       71.6777 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_08-17-17 |    0.695325 |
  |   2 |              4.14272 |      0.844942 |                15.554 |          15.554 |       65.4183 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_12-50-49 |    0.688627 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | --------: |
  | train/f1         | 0.826683 | 0.844942 | 0.856392 |     3 | 0.867842 | 0.840403 |  0.808424 | 0.0299679 |
  | train/loss       |  13.9126 |   15.554 |  17.9904 |     3 |  20.4268 |   16.084 |   12.2712 |   4.10352 |
  | train/loss_epoch |  13.9126 |   15.554 |  17.9904 |     3 |  20.4268 |   16.084 |   12.2712 |   4.10352 |
  | train/loss_step  |  1.03103 |   2.0451 |  3.09391 |     3 |  4.14272 |  2.06826 | 0.0169539 |   2.06298 |
  | val/f1           | 0.687756 | 0.688627 | 0.691976 |     3 | 0.695325 | 0.690279 |  0.686884 | 0.0044564 |
  | val/loss         |  62.2667 |  65.4183 |   68.548 |     3 |  71.6777 |  65.4037 |   59.1151 |   6.28131 |

## 2024-01-08

### Coreference - frozen MRPC with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/ebj0aj2o
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/5zz8d08b
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/vchnwesd

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) | ('val/f1',) | ('model_save_dir',)                                                                                                     | ('train/loss',) |
  | --: | -------------------: | --------------------: | ------------: | ------------: | ----------: | :---------------------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |             0.136623 |               30.5283 |       46.4055 |      0.721922 |    0.672093 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_20-18-56 |         30.5283 |
  |   1 |              36.9673 |               24.6114 |        47.796 |      0.759848 |    0.677824 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_23-15-20 |         24.6114 |
  |   2 |              11.4451 |               23.8135 |       47.2311 |      0.763991 |    0.674617 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_03-35-32 |         23.8135 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.740885 | 0.759848 |  0.76192 |     3 | 0.763991 | 0.748587 | 0.721922 |  0.0231854 |
  | train/loss       |  24.2124 |  24.6114 |  27.5698 |     3 |  30.5283 |  26.3177 |  23.8135 |     3.6682 |
  | train/loss_epoch |  24.2124 |  24.6114 |  27.5698 |     3 |  30.5283 |  26.3177 |  23.8135 |     3.6682 |
  | train/loss_step  |  5.79085 |  11.4451 |  24.2062 |     3 |  36.9673 |   16.183 | 0.136623 |    18.8669 |
  | val/f1           | 0.673355 | 0.674617 | 0.676221 |     3 | 0.677824 | 0.674845 | 0.672093 | 0.00287237 |
  | val/loss         |  46.8183 |  47.2311 |  47.5136 |     3 |   47.796 |  47.1442 |  46.4055 |     0.6993 |

### Coreference - frozen NER with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/7rdd93m9
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/vfkc11el
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/gk94zogl

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                                     | ('train/loss',) | ('train/loss_step',) | ('train/f1',) |
  | --: | --------------------: | ----------: | ------------: | :---------------------------------------------------------------------------------------------------------------------- | --------------: | -------------------: | ------------: |
  |   0 |               59.4301 |    0.554564 |        67.772 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_08-17-04 |         59.4301 |               43.146 |      0.478319 |
  |   1 |               62.5422 |    0.536423 |       73.4605 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_12-44-56 |         62.5422 |              31.7394 |      0.459762 |
  |   2 |               59.1741 |    0.553477 |       70.1303 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_16-01-49 |         59.1741 |              84.4339 |      0.485952 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.469041 | 0.478319 | 0.482135 |     3 | 0.485952 | 0.474678 | 0.459762 | 0.0134692 |
  | train/loss       |  59.3021 |  59.4301 |  60.9862 |     3 |  62.5422 |  60.3821 |  59.1741 |   1.87506 |
  | train/loss_epoch |  59.3021 |  59.4301 |  60.9862 |     3 |  62.5422 |  60.3821 |  59.1741 |   1.87506 |
  | train/loss_step  |  37.4427 |   43.146 |  63.7899 |     3 |  84.4339 |  53.1064 |  31.7394 |   27.7233 |
  | val/f1           |  0.54495 | 0.553477 |  0.55402 |     3 | 0.554564 | 0.548155 | 0.536423 | 0.0101745 |
  | val/loss         |  68.9512 |  70.1303 |  71.7954 |     3 |  73.4605 |  70.4543 |   67.772 |   2.85805 |

### Coreference - frozen RE with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/x5c20ucv
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/ubquw9ru
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/dyq8ai7y

- metric values per seed

  |     | ('train/loss_step',) | ('model_save_dir',)                                                                                                     | ('train/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) |
  | --: | -------------------: | :---------------------------------------------------------------------------------------------------------------------- | --------------: | ------------: | --------------------: | ------------: | ----------: |
  |   0 |              28.3779 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_20-18-41 |         20.5985 |      0.414269 |               20.5985 |       26.6369 |    0.409225 |
  |   1 |            0.0111651 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_21-57-57 |         19.8128 |      0.420992 |               19.8128 |       26.1442 |    0.411684 |
  |   2 |              10.7269 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_23-50-07 |          21.203 |      0.410291 |                21.203 |       27.8549 |    0.409373 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         |  0.41228 | 0.414269 |  0.41763 |     3 | 0.420992 | 0.415184 |  0.410291 | 0.00540896 |
  | train/loss       |  20.2056 |  20.5985 |  20.9008 |     3 |   21.203 |  20.5381 |   19.8128 |   0.697105 |
  | train/loss_epoch |  20.2056 |  20.5985 |  20.9008 |     3 |   21.203 |  20.5381 |   19.8128 |   0.697105 |
  | train/loss_step  |  5.36903 |  10.7269 |  19.5524 |     3 |  28.3779 |  13.0386 | 0.0111651 |    14.3239 |
  | val/f1           | 0.409299 | 0.409373 | 0.410528 |     3 | 0.411684 | 0.410094 |  0.409225 | 0.00137878 |
  | val/loss         |  26.3906 |  26.6369 |  27.2459 |     3 |  27.8549 |  26.8787 |   26.1442 |    0.88058 |

### Coreference - frozen SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m

  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/rd1e5zw9
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/rsxyob2j
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/b8r7f731

- metric values per seed

  |     | ('val/loss',) | ('model_save_dir',)                                                                                                     | ('train/loss_step',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss',) | ('train/f1',) |
  | --: | ------------: | :---------------------------------------------------------------------------------------------------------------------- | -------------------: | ----------: | --------------------: | --------------: | ------------: |
  |   0 |       57.7365 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_01-21-04 |              80.5009 |    0.611719 |               47.1702 |         47.1702 |      0.596692 |
  |   1 |       57.8067 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_04-30-56 |              7.19399 |    0.614613 |               45.5237 |         45.5237 |      0.606791 |
  |   2 |       63.6912 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_08-07-03 |              15.4871 |    0.603003 |               48.9819 |         48.9819 |       0.58498 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.590836 | 0.596692 | 0.601741 |     3 | 0.606791 | 0.596154 |  0.58498 |  0.0109154 |
  | train/loss       |   46.347 |  47.1702 |   48.076 |     3 |  48.9819 |  47.2253 |  45.5237 |    1.72972 |
  | train/loss_epoch |   46.347 |  47.1702 |   48.076 |     3 |  48.9819 |  47.2253 |  45.5237 |    1.72972 |
  | train/loss_step  |  11.3405 |  15.4871 |   47.994 |     3 |  80.5009 |   34.394 |  7.19399 |    40.1445 |
  | val/f1           | 0.607361 | 0.611719 | 0.613166 |     3 | 0.614613 | 0.609778 | 0.603003 | 0.00604308 |
  | val/loss         |  57.7716 |  57.8067 |  60.7489 |     3 |  63.6912 |  59.7448 |  57.7365 |    3.41786 |

### Coreference - frozen BERT with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/in8brtyb
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/mk3p086e
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/0h4s3u84

- metric values per seed

  |     | ('val/loss',) | ('val/f1',) | ('model_save_dir',)                                                                                                     | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('train/loss',) |
  | --: | ------------: | ----------: | :---------------------------------------------------------------------------------------------------------------------- | --------------------: | -------------------: | ------------: | --------------: |
  |   0 |       20.9619 |    0.439327 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_20-18-34 |               13.4334 |              32.3984 |       0.47579 |         13.4334 |
  |   1 |       20.3043 |     0.43242 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_22-09-43 |               14.9772 |              24.4948 |      0.463332 |         14.9772 |
  |   2 |       20.4083 |    0.446456 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-08_23-35-27 |               11.1857 |              54.9162 |      0.490838 |         11.1857 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.469561 |  0.47579 | 0.483314 |     3 | 0.490838 | 0.476653 | 0.463332 |  0.0137731 |
  | train/loss       |  12.3095 |  13.4334 |  14.2053 |     3 |  14.9772 |  13.1987 |  11.1857 |    1.90661 |
  | train/loss_epoch |  12.3095 |  13.4334 |  14.2053 |     3 |  14.9772 |  13.1987 |  11.1857 |    1.90661 |
  | train/loss_step  |  28.4466 |  32.3984 |  43.6573 |     3 |  54.9162 |  37.2698 |  24.4948 |    15.7849 |
  | val/f1           | 0.435874 | 0.439327 | 0.442892 |     3 | 0.446456 | 0.439401 |  0.43242 | 0.00701831 |
  | val/loss         |  20.3563 |  20.4083 |  20.6851 |     3 |  20.9619 |  20.5582 |  20.3043 |   0.353505 |

### Coreference - frozen coref-hoi with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.normalize_embeddings=True \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_norm \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/pwvl1svw
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/g8mdop0c
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_norm-training/runs/2n9r5vqe

- metric values per seed

  |     | ('train/loss_step',) | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('model_save_dir',)                                                                                                     | ('val/loss',) | ('train/loss_epoch',) |
  | --: | -------------------: | ----------: | --------------: | ------------: | :---------------------------------------------------------------------------------------------------------------------- | ------------: | --------------------: |
  |   0 |              1.11368 |    0.725392 |         10.1768 |      0.907404 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_02-05-20 |       64.1058 |               10.1768 |
  |   1 |              24.4548 |    0.731675 |         8.72509 |      0.919409 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_04-26-23 |       68.2089 |               8.72509 |
  |   2 |          9.53674e-07 |    0.736246 |         7.67928 |      0.929775 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_norm/2024-01-09_07-30-09 |       74.8346 |               7.67928 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         | 0.913406 | 0.919409 | 0.924592 |     3 | 0.929775 | 0.918863 |    0.907404 |  0.0111953 |
  | train/loss       |  8.20219 |  8.72509 |  9.45094 |     3 |  10.1768 |  8.86039 |     7.67928 |    1.25424 |
  | train/loss_epoch |  8.20219 |  8.72509 |  9.45094 |     3 |  10.1768 |  8.86039 |     7.67928 |    1.25424 |
  | train/loss_step  | 0.556839 |  1.11368 |  12.7842 |     3 |  24.4548 |  8.52282 | 9.53674e-07 |    13.8087 |
  | val/f1           | 0.728534 | 0.731675 |  0.73396 |     3 | 0.736246 | 0.731104 |    0.725392 | 0.00544911 |
  | val/loss         |  66.1573 |  68.2089 |  71.5217 |     3 |  74.8346 |  69.0498 |     64.1058 |     5.4136 |

### Coreference - frozen RE + MRPC with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/z7igd74k
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/bklbgy8t
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/o26djpnq

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                                                 | ('train/loss',) | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss_step',) |
  | --: | ------------: | :------------------------------------------------------------------------------------------------------------------ | --------------: | ----------: | --------------------: | ------------: | -------------------: |
  |   0 |      0.465094 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-08_20-18-07 |         13.9569 |     0.43272 |               13.9569 |       21.2012 |              26.7544 |
  |   1 |      0.456585 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-08_23-22-59 |         14.8582 |    0.429162 |               14.8582 |       21.4686 |              2.19297 |
  |   2 |      0.727125 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_01-58-27 |          29.864 |    0.670029 |                29.864 |       47.0462 |              66.2085 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |      std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | -------: |
  | train/f1         |  0.46084 | 0.465094 |  0.59611 |     3 | 0.727125 | 0.549601 | 0.456585 | 0.153798 |
  | train/loss       |  14.4076 |  14.8582 |  22.3611 |     3 |   29.864 |  19.5597 |  13.9569 |  8.93516 |
  | train/loss_epoch |  14.4076 |  14.8582 |  22.3611 |     3 |   29.864 |  19.5597 |  13.9569 |  8.93516 |
  | train/loss_step  |  14.4737 |  26.7544 |  46.4814 |     3 |  66.2085 |  31.7186 |  2.19297 |  32.2952 |
  | val/f1           | 0.430941 |  0.43272 | 0.551375 |     3 | 0.670029 | 0.510637 | 0.429162 | 0.138049 |
  | val/loss         |  21.3349 |  21.4686 |  34.2574 |     3 |  47.0462 |  29.9053 |  21.2012 |   14.845 |

### Coreference - frozen NER + MRPC with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/brw0g4zf
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/064fbc7l
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/52qkgjl2

- metric values per seed

  |     | ('train/loss_step',) | ('val/loss',) | ('train/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('train/f1',) | ('model_save_dir',)                                                                                                 |
  | --: | -------------------: | ------------: | --------------: | --------------------: | ----------: | ------------: | :------------------------------------------------------------------------------------------------------------------ |
  |   0 |             0.198269 |       47.3552 |         27.9918 |               27.9918 |    0.675114 |      0.738415 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_05-22-14 |
  |   1 |              22.8681 |       47.9206 |         27.6869 |               27.6869 |    0.677164 |      0.743223 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_09-25-14 |
  |   2 |              62.6354 |       47.6322 |          32.688 |                32.688 |    0.668407 |       0.70755 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_13-43-49 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.722983 | 0.738415 | 0.740819 |     3 | 0.743223 | 0.729729 |  0.70755 |  0.0193577 |
  | train/loss       |  27.8393 |  27.9918 |  30.3399 |     3 |   32.688 |  29.4556 |  27.6869 |    2.80352 |
  | train/loss_epoch |  27.8393 |  27.9918 |  30.3399 |     3 |   32.688 |  29.4556 |  27.6869 |    2.80352 |
  | train/loss_step  |  11.5332 |  22.8681 |  42.7517 |     3 |  62.6354 |  28.5673 | 0.198269 |    31.6063 |
  | val/f1           | 0.671761 | 0.675114 | 0.676139 |     3 | 0.677164 | 0.673562 | 0.668407 | 0.00458003 |
  | val/loss         |  47.4937 |  47.6322 |  47.7764 |     3 |  47.9206 |   47.636 |  47.3552 |   0.282705 |

### Coreference - frozen RE + NER with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/lb90ipwe
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/ask8k0aj
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/z6zfoohi

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('train/loss',) | ('train/loss_epoch',) | ('train/loss_step',) | ('model_save_dir',)                                                                                                 | ('train/f1',) |
  | --: | ----------: | ------------: | --------------: | --------------------: | -------------------: | :------------------------------------------------------------------------------------------------------------------ | ------------: |
  |   0 |    0.630639 |       54.9382 |         34.1104 |               34.1104 |            0.0469153 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-08_20-18-11 |      0.679444 |
  |   1 |    0.623911 |       54.6156 |          40.466 |                40.466 |              94.4947 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_00-53-26 |      0.642183 |
  |   2 |    0.629536 |       55.1826 |         36.2908 |               36.2908 |              53.5334 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_04-00-11 |      0.669007 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.655595 | 0.669007 | 0.674225 |     3 | 0.679444 | 0.663544 |  0.642183 |  0.0192218 |
  | train/loss       |  35.2006 |  36.2908 |  38.3784 |     3 |   40.466 |  36.9558 |   34.1104 |    3.22956 |
  | train/loss_epoch |  35.2006 |  36.2908 |  38.3784 |     3 |   40.466 |  36.9558 |   34.1104 |    3.22956 |
  | train/loss_step  |  26.7902 |  53.5334 |   74.014 |     3 |  94.4947 |  49.3583 | 0.0469153 |    47.3621 |
  | val/f1           | 0.626723 | 0.629536 | 0.630088 |     3 | 0.630639 | 0.628029 |  0.623911 | 0.00360856 |
  | val/loss         |  54.7769 |  54.9382 |  55.0604 |     3 |  55.1826 |  54.9121 |   54.6156 |   0.284407 |

### Coreference - frozen SQUAD + MRPC with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/mc5ql1l5
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/pmvhls5e
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/35ki4nlv

- metric values per seed

  |     | ('val/f1',) | ('train/loss',) | ('model_save_dir',)                                                                                                 | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('val/loss',) |
  | --: | ----------: | --------------: | :------------------------------------------------------------------------------------------------------------------ | --------------------: | -------------------: | ------------: | ------------: |
  |   0 |    0.416297 |          17.575 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_08-00-05 |                17.575 |               7.4755 |      0.436778 |       22.5066 |
  |   1 |    0.677024 |         26.3889 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_09-42-40 |               26.3889 |               5.9723 |      0.751265 |       45.8626 |
  |   2 |    0.676551 |         28.7564 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_14-26-35 |               28.7564 |              14.7805 |      0.736038 |       45.7896 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |      std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | -------: |
  | train/f1         | 0.586408 | 0.736038 | 0.743652 |     3 | 0.751265 |  0.64136 | 0.436778 | 0.177337 |
  | train/loss       |  21.9819 |  26.3889 |  27.5726 |     3 |  28.7564 |  24.2401 |   17.575 |  5.89227 |
  | train/loss_epoch |  21.9819 |  26.3889 |  27.5726 |     3 |  28.7564 |  24.2401 |   17.575 |  5.89227 |
  | train/loss_step  |   6.7239 |   7.4755 |   11.128 |     3 |  14.7805 |  9.40944 |   5.9723 |  4.71181 |
  | val/f1           | 0.546424 | 0.676551 | 0.676787 |     3 | 0.677024 | 0.589957 | 0.416297 | 0.150394 |
  | val/loss         |  34.1481 |  45.7896 |  45.8261 |     3 |  45.8626 |  38.0529 |  22.5066 |  13.4636 |

### Coreference - frozen RE + SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/k3gpeirr
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/0le7y093
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/nawc429g

- metric values per seed

  |     | ('train/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                 | ('val/f1',) | ('train/f1',) | ('val/loss',) | ('train/loss_step',) |
  | --: | --------------: | --------------------: | :------------------------------------------------------------------------------------------------------------------ | ----------: | ------------: | ------------: | -------------------: |
  |   0 |         33.3078 |               33.3078 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-08_20-18-22 |    0.648127 |      0.692306 |       51.7016 |              33.4413 |
  |   1 |         37.9774 |               37.9774 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_00-38-02 |    0.642238 |       0.66349 |       52.2842 |              20.0102 |
  |   2 |         34.7684 |               34.7684 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_03-41-54 |    0.644425 |       0.68564 |       51.6413 |               42.364 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.674565 |  0.68564 | 0.688973 |     3 | 0.692306 | 0.680479 |  0.66349 |  0.0150853 |
  | train/loss       |  34.0381 |  34.7684 |  36.3729 |     3 |  37.9774 |  35.3512 |  33.3078 |    2.38875 |
  | train/loss_epoch |  34.0381 |  34.7684 |  36.3729 |     3 |  37.9774 |  35.3512 |  33.3078 |    2.38875 |
  | train/loss_step  |  26.7258 |  33.4413 |  37.9027 |     3 |   42.364 |  31.9385 |  20.0102 |    11.2524 |
  | val/f1           | 0.643332 | 0.644425 | 0.646276 |     3 | 0.648127 |  0.64493 | 0.642238 | 0.00297652 |
  | val/loss         |  51.6715 |  51.7016 |  51.9929 |     3 |  52.2842 |  51.8757 |  51.6413 |   0.355062 |

### Coreference - frozen NER + SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/jv99zvwi
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/sb87u54b
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_norm-training/runs/er8mngp1

- metric values per seed

  |     | ('train/loss_step',) | ('train/loss',) | ('train/f1',) | ('val/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                 | ('val/loss',) |
  | --: | -------------------: | --------------: | ------------: | ----------: | --------------------: | :------------------------------------------------------------------------------------------------------------------ | ------------: |
  |   0 |              41.1749 |         45.8096 |      0.602814 |    0.618562 |               45.8096 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_07-32-25 |       54.0658 |
  |   1 |              72.1198 |          42.448 |      0.623841 |    0.626965 |                42.448 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_10-39-56 |       52.2737 |
  |   2 |              19.6398 |         40.1853 |      0.637537 |    0.629847 |               40.1853 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_norm/2024-01-09_14-34-44 |       52.4025 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.613328 | 0.623841 | 0.630689 |     3 | 0.637537 | 0.621397 | 0.602814 |  0.0174901 |
  | train/loss       |  41.3166 |   42.448 |  44.1288 |     3 |  45.8096 |  42.8143 |  40.1853 |    2.82999 |
  | train/loss_epoch |  41.3166 |   42.448 |  44.1288 |     3 |  45.8096 |  42.8143 |  40.1853 |    2.82999 |
  | train/loss_step  |  30.4074 |  41.1749 |  56.6474 |     3 |  72.1198 |  44.3115 |  19.6398 |    26.3802 |
  | val/f1           | 0.622764 | 0.626965 | 0.628406 |     3 | 0.629847 | 0.625125 | 0.618562 | 0.00586305 |
  | val/loss         |  52.3381 |  52.4025 |  53.2342 |     3 |  54.0658 |   52.914 |  52.2737 |   0.999592 |

### Coreference - frozen RE + NER + SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/gt072grh
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/v6m6x0bq
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/qucdrkin

- metric values per seed

  |     | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                       | ('val/loss',) | ('train/loss_step',) | ('train/loss',) | ('val/f1',) |
  | --: | ------------: | --------------------: | :------------------------------------------------------------------------------------------------------------------------ | ------------: | -------------------: | --------------: | ----------: |
  |   0 |      0.681872 |               34.3754 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-10_21-53-58 |        50.653 |              8.36182 |         34.3754 |    0.648538 |
  |   1 |      0.704393 |               31.5396 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_01-38-33 |        50.443 |               55.511 |         31.5396 |    0.651136 |
  |   2 |      0.659184 |               37.9351 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_06-11-12 |       50.6356 |              42.7051 |         37.9351 |    0.641529 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.670528 | 0.681872 | 0.693132 |     3 | 0.704393 | 0.681816 | 0.659184 |  0.0226045 |
  | train/loss       |  32.9575 |  34.3754 |  36.1552 |     3 |  37.9351 |  34.6167 |  31.5396 |    3.20456 |
  | train/loss_epoch |  32.9575 |  34.3754 |  36.1552 |     3 |  37.9351 |  34.6167 |  31.5396 |    3.20456 |
  | train/loss_step  |  25.5335 |  42.7051 |  49.1081 |     3 |   55.511 |   35.526 |  8.36182 |    24.3806 |
  | val/f1           | 0.645034 | 0.648538 | 0.649837 |     3 | 0.651136 | 0.647068 | 0.641529 | 0.00496935 |
  | val/loss         |  50.5393 |  50.6356 |  50.6443 |     3 |   50.653 |  50.5772 |   50.443 |    0.11659 |

### Coreference - frozen MRPC + NER + SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/g5p3p9jn
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/6pmq54fg
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/rvfewb41

- metric values per seed

  |     | ('train/loss',) | ('val/loss',) | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                                       |
  | --: | --------------: | ------------: | ------------: | -------------------: | --------------------: | ----------: | :------------------------------------------------------------------------------------------------------------------------ |
  |   0 |         15.3288 |        21.683 |      0.451513 |            0.0782018 |               15.3288 |    0.428803 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-10_21-54-06 |
  |   1 |         25.8295 |         45.57 |      0.749987 |              81.3009 |               25.8295 |    0.673254 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_00-39-06 |
  |   2 |         28.7834 |       47.1032 |      0.732119 |              12.2242 |               28.7834 |    0.670968 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_05-16-02 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |      std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | -------: |
  | train/f1         | 0.591816 | 0.732119 | 0.741053 |     3 | 0.749987 |  0.64454 |  0.451513 | 0.167404 |
  | train/loss       |  20.5791 |  25.8295 |  27.3064 |     3 |  28.7834 |  23.3139 |   15.3288 |  7.07127 |
  | train/loss_epoch |  20.5791 |  25.8295 |  27.3064 |     3 |  28.7834 |  23.3139 |   15.3288 |  7.07127 |
  | train/loss_step  |   6.1512 |  12.2242 |  46.7626 |     3 |  81.3009 |  31.2011 | 0.0782018 |  43.8107 |
  | val/f1           | 0.549885 | 0.670968 | 0.672111 |     3 | 0.673254 | 0.591008 |  0.428803 | 0.140478 |
  | val/loss         |  33.6265 |    45.57 |  46.3366 |     3 |  47.1032 |  38.1187 |    21.683 |  14.2544 |

### Coreference - frozen MRPC + RE + SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/yd7bf8v5
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/s1e4jrzc
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_norm-training/runs/zh5ok8if

- metric values per seed

  |     | ('train/loss_epoch',) | ('val/loss',) | ('train/loss_step',) | ('val/f1',) | ('model_save_dir',)                                                                                                       | ('train/f1',) | ('train/loss',) |
  | --: | --------------------: | ------------: | -------------------: | ----------: | :------------------------------------------------------------------------------------------------------------------------ | ------------: | --------------: |
  |   0 |               26.4992 |       46.7512 |              17.1022 |    0.674058 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_09-07-06 |      0.747472 |         26.4992 |
  |   1 |               30.4367 |       45.6223 |              76.6112 |    0.670783 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_13-19-18 |       0.72339 |         30.4367 |
  |   2 |               29.8298 |       45.8557 |              12.4597 |    0.670178 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_norm/2024-01-11_16-34-43 |      0.726022 |         29.8298 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.724706 | 0.726022 | 0.736747 |     3 | 0.747472 | 0.732295 |  0.72339 |  0.0132093 |
  | train/loss       |  28.1645 |  29.8298 |  30.1333 |     3 |  30.4367 |  28.9219 |  26.4992 |    2.11998 |
  | train/loss_epoch |  28.1645 |  29.8298 |  30.1333 |     3 |  30.4367 |  28.9219 |  26.4992 |    2.11998 |
  | train/loss_step  |   14.781 |  17.1022 |  46.8567 |     3 |  76.6112 |   35.391 |  12.4597 |    35.7731 |
  | val/f1           | 0.670481 | 0.670783 | 0.672421 |     3 | 0.674058 | 0.671673 | 0.670178 | 0.00208763 |
  | val/loss         |   45.739 |  45.8557 |  46.3034 |     3 |  46.7512 |  46.0764 |  45.6223 |   0.595925 |

### Coreference - frozen MRPC + RE + NER with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_norm \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/epjtf53n
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/3vm6vspf
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models-training/runs/k4g0p2st

- metric values per seed

  |     | ('val/f1',) | ('train/loss_epoch',) | ('train/f1',) | ('val/loss',) | ('train/loss',) | ('model_save_dir',)                                                                                                  | ('train/loss_step',) |
  | --: | ----------: | --------------------: | ------------: | ------------: | --------------: | :------------------------------------------------------------------------------------------------------------------- | -------------------: |
  |   0 |    0.670629 |               27.3194 |      0.741536 |       47.8456 |         27.3194 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-11_12-28-34 |              6.21814 |
  |   1 |    0.668053 |               31.2947 |      0.714523 |       46.9837 |         31.2947 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-11_16-56-46 |              48.4074 |
  |   2 |    0.667796 |               30.1582 |       0.71981 |       47.1105 |         30.1582 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models/2024-01-11_20-28-44 |              5.07897 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.717167 |  0.71981 | 0.730673 |     3 | 0.741536 |  0.72529 | 0.714523 |   0.014316 |
  | train/loss       |  28.7388 |  30.1582 |  30.7265 |     3 |  31.2947 |  29.5908 |  27.3194 |    2.04751 |
  | train/loss_epoch |  28.7388 |  30.1582 |  30.7265 |     3 |  31.2947 |  29.5908 |  27.3194 |    2.04751 |
  | train/loss_step  |  5.64856 |  6.21814 |  27.3128 |     3 |  48.4074 |  19.9015 |  5.07897 |    24.6934 |
  | val/f1           | 0.667924 | 0.668053 | 0.669341 |     3 | 0.670629 | 0.668826 | 0.667796 | 0.00156691 |
  | val/loss         |  47.0471 |  47.1105 |  47.4781 |     3 |  47.8456 |  47.3133 |  46.9837 |   0.465369 |

### Coreference - frozen MRPC + RE + NER + SQUAD with attention aggregation and normalization

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.normalize_embeddings=True \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/all_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/b5jftafm
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/7jeil7gi
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/b2pi3oz3

- metric values per seed

  |     | ('val/f1',) | ('train/loss_epoch',) | ('train/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                             | ('val/loss',) | ('train/loss',) |
  | --: | ----------: | --------------------: | ------------: | -------------------: | :-------------------------------------------------------------------------------------------------------------- | ------------: | --------------: |
  |   0 |    0.663474 |               33.8581 |      0.695456 |             0.359833 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_16-25-57 |       47.5205 |         33.8581 |
  |   1 |    0.673879 |                 25.88 |      0.751401 |              10.8001 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_19-18-32 |        47.272 |           25.88 |
  |   2 |    0.666774 |               30.3414 |      0.721325 |              25.1241 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-12_00-18-01 |       46.1828 |         30.3414 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |  0.70839 | 0.721325 | 0.736363 |     3 | 0.751401 | 0.722727 | 0.695456 |  0.0279992 |
  | train/loss       |  28.1107 |  30.3414 |  32.0997 |     3 |  33.8581 |  30.0265 |    25.88 |    3.99834 |
  | train/loss_epoch |  28.1107 |  30.3414 |  32.0997 |     3 |  33.8581 |  30.0265 |    25.88 |    3.99834 |
  | train/loss_step  |  5.57997 |  10.8001 |  17.9621 |     3 |  25.1241 |  12.0947 | 0.359833 |    12.4328 |
  | val/f1           | 0.665124 | 0.666774 | 0.670326 |     3 | 0.673879 | 0.668042 | 0.663474 | 0.00531737 |
  | val/loss         |  46.7274 |   47.272 |  47.3963 |     3 |  47.5205 |  46.9918 |  46.1828 |   0.711539 |

## 2024-01-10

### Coreference - truncated MRPC 9 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_truncated \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/td2ha70l
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/ms8b6gpi
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/w1yhy6hl

- metric values per seed

  |     | ('val/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                                          | ('train/loss',) |
  | --: | ------------: | --------------------: | ------------: | ----------: | -------------------: | :--------------------------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |       155.717 |               43.1638 |      0.801169 |    0.672998 |               47.901 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-13_16-59-31 |         43.1638 |
  |   1 |       154.262 |               40.6201 |      0.805399 |     0.67547 |          1.90735e-06 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-13_19-45-03 |         40.6201 |
  |   2 |        155.03 |               49.8242 |      0.782477 |    0.669485 |              26.6115 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-13_22-39-08 |         49.8242 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         | 0.791823 | 0.801169 | 0.803284 |     3 | 0.805399 | 0.796348 |    0.782477 |  0.0121977 |
  | train/loss       |  41.8919 |  43.1638 |   46.494 |     3 |  49.8242 |   44.536 |     40.6201 |      4.753 |
  | train/loss_epoch |  41.8919 |  43.1638 |   46.494 |     3 |  49.8242 |   44.536 |     40.6201 |      4.753 |
  | train/loss_step  |  13.3058 |  26.6115 |  37.2563 |     3 |   47.901 |  24.8375 | 1.90735e-06 |    23.9997 |
  | val/f1           | 0.671241 | 0.672998 | 0.674234 |     3 |  0.67547 | 0.672651 |    0.669485 | 0.00300769 |
  | val/loss         |  154.646 |   155.03 |  155.373 |     3 |  155.717 |  155.003 |     154.262 |   0.727668 |

### Coreference - truncated NER 6 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_truncated \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/scgwbc2p
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/m72siwg6
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/0aqjg246

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('model_save_dir',)                                                                                                          | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/f1',) |
  | --: | --------------------: | -------------------: | :--------------------------------------------------------------------------------------------------------------------------- | ------------: | ----------: | --------------: | ------------: |
  |   0 |               43.0263 |              122.054 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_15-36-41 |       144.909 |    0.642021 |         43.0263 |      0.772032 |
  |   1 |               39.3592 |               1.9094 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_18-01-25 |       144.429 |    0.645663 |         39.3592 |       0.78187 |
  |   2 |               46.4784 |              10.0777 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_20-34-41 |       146.956 |    0.641631 |         46.4784 |      0.759982 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.766007 | 0.772032 | 0.776951 |     3 |  0.78187 | 0.771295 | 0.759982 |  0.010963 |
  | train/loss       |  41.1927 |  43.0263 |  44.7523 |     3 |  46.4784 |  42.9546 |  39.3592 |   3.56013 |
  | train/loss_epoch |  41.1927 |  43.0263 |  44.7523 |     3 |  46.4784 |  42.9546 |  39.3592 |   3.56013 |
  | train/loss_step  |  5.99358 |  10.0777 |  66.0661 |     3 |  122.054 |  44.6805 |   1.9094 |   67.1321 |
  | val/f1           | 0.641826 | 0.642021 | 0.643842 |     3 | 0.645663 | 0.643105 | 0.641631 | 0.0022242 |
  | val/loss         |  144.669 |  144.909 |  145.932 |     3 |  146.956 |  145.431 |  144.429 |   1.34158 |

### Coreference - truncated RE 9 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_truncated \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/en29b8s0
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/o3lkz6b1
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/udpk8hhe

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('train/f1',) | ('val/f1',) | ('train/loss',) | ('model_save_dir',)                                                                                                          | ('val/loss',) |
  | --: | --------------------: | -------------------: | ------------: | ----------: | --------------: | :--------------------------------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |               86.6631 |              19.3463 |      0.695686 |    0.629684 |         86.6631 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_15-36-20 |       176.989 |
  |   1 |               70.4884 |              194.713 |      0.724746 |    0.638546 |         70.4884 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_17-39-50 |        170.89 |
  |   2 |               56.0013 |              24.8659 |      0.756209 |    0.643587 |         56.0013 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_20-11-22 |       165.347 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.710216 | 0.724746 | 0.740477 |     3 | 0.756209 | 0.725547 | 0.695686 | 0.0302692 |
  | train/loss       |  63.2448 |  70.4884 |  78.5757 |     3 |  86.6631 |  71.0509 |  56.0013 |   15.3387 |
  | train/loss_epoch |  63.2448 |  70.4884 |  78.5757 |     3 |  86.6631 |  71.0509 |  56.0013 |   15.3387 |
  | train/loss_step  |  22.1061 |  24.8659 |   109.79 |     3 |  194.713 |  79.6418 |  19.3463 |   99.6931 |
  | val/f1           | 0.634115 | 0.638546 | 0.641067 |     3 | 0.643587 | 0.637273 | 0.629684 | 0.0070384 |
  | val/loss         |  168.119 |   170.89 |   173.94 |     3 |  176.989 |  171.076 |  165.347 |   5.82356 |

### Coreference - truncated SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_truncated \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/3dfc1cno
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/vvm19u3d
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/c953hqbq

- metric values per seed

  |     | ('train/loss',) | ('train/f1',) | ('val/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                          | ('train/loss_step',) | ('val/loss',) |
  | --: | --------------: | ------------: | ----------: | --------------------: | :--------------------------------------------------------------------------------------------------------------------------- | -------------------: | ------------: |
  |   0 |         50.0811 |      0.790699 |    0.658903 |               50.0811 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-09_23-19-57 |              151.273 |       169.442 |
  |   1 |         30.2876 |      0.841477 |    0.670032 |               30.2876 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-10_01-42-01 |          7.15256e-05 |       170.441 |
  |   2 |         37.5194 |      0.821645 |    0.665104 |               37.5194 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-10_05-12-48 |            0.0102519 |       171.626 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |    mean |         min |        std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | ------: | ----------: | ---------: |
  | train/f1         |   0.806172 |  0.821645 | 0.831561 |     3 | 0.841477 | 0.81794 |    0.790699 |  0.0255909 |
  | train/loss       |    33.9035 |   37.5194 |  43.8003 |     3 |  50.0811 |  39.296 |     30.2876 |    10.0156 |
  | train/loss_epoch |    33.9035 |   37.5194 |  43.8003 |     3 |  50.0811 |  39.296 |     30.2876 |    10.0156 |
  | train/loss_step  | 0.00516173 | 0.0102519 |  75.6418 |     3 |  151.273 | 50.4279 | 7.15256e-05 |    87.3348 |
  | val/f1           |   0.662004 |  0.665104 | 0.667568 |     3 | 0.670032 | 0.66468 |    0.658903 | 0.00557655 |
  | val/loss         |    169.941 |   170.441 |  171.033 |     3 |  171.626 | 170.503 |     169.442 |    1.09303 |

### Coreference - truncated BERT 10 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased=10 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/single_models_truncated \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/mbrhff3t
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/netubfzz
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-single_models_truncated-training/runs/33r52bu2

- metric values per seed

  |     | ('train/loss',) | ('train/f1',) | ('val/loss',) | ('val/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                                          | ('train/loss_epoch',) |
  | --: | --------------: | ------------: | ------------: | ----------: | -------------------: | :--------------------------------------------------------------------------------------------------------------------------- | --------------------: |
  |   0 |         33.8514 |      0.836917 |       172.249 |    0.677976 |              120.136 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-10_08-13-51 |               33.8514 |
  |   1 |         35.3232 |      0.830454 |       173.081 |     0.67654 |              82.3202 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-10_11-18-50 |               35.3232 |
  |   2 |         14.2005 |      0.896894 |       174.474 |     0.69102 |           0.00103374 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/single_models_truncated/2024-01-10_14-20-05 |               14.2005 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         | 0.833685 | 0.836917 | 0.866906 |     3 | 0.896894 | 0.854755 |   0.830454 |  0.0366365 |
  | train/loss       |   24.026 |  33.8514 |  34.5873 |     3 |  35.3232 |  27.7917 |    14.2005 |    11.7933 |
  | train/loss_epoch |   24.026 |  33.8514 |  34.5873 |     3 |  35.3232 |  27.7917 |    14.2005 |    11.7933 |
  | train/loss_step  |  41.1606 |  82.3202 |  101.228 |     3 |  120.136 |  67.4859 | 0.00103374 |    61.4261 |
  | val/f1           | 0.677258 | 0.677976 | 0.684498 |     3 |  0.69102 | 0.681845 |    0.67654 | 0.00797752 |
  | val/loss         |  172.665 |  173.081 |  173.777 |     3 |  174.474 |  173.268 |    172.249 |      1.124 |

### Coreference - truncated RE 9 + MRPC 9 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/ry1ydiin
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/dri0mbnj
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/lkjcxi02

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('model_save_dir',)                                                                                                      | ('train/loss',) |
  | --: | -------------------: | ------------: | ----------: | --------------------: | ------------: | :----------------------------------------------------------------------------------------------------------------------- | --------------: |
  |   0 |                    0 |      0.882347 |    0.688854 |               15.4889 |       122.938 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_15-57-20 |         15.4889 |
  |   1 |           0.00232984 |      0.898184 |    0.690352 |               12.4109 |       126.871 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_19-40-56 |         12.4109 |
  |   2 |             0.669282 |      0.842755 |    0.684292 |               22.3563 |       113.898 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_00-08-39 |         22.3563 |

- aggregated values:

  |                  |        25% |        50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ---------: | ---------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |   0.862551 |   0.882347 | 0.890265 |     3 | 0.898184 | 0.874428 | 0.842755 |  0.0285503 |
  | train/loss       |    13.9499 |    15.4889 |  18.9226 |     3 |  22.3563 |   16.752 |  12.4109 |    5.09159 |
  | train/loss_epoch |    13.9499 |    15.4889 |  18.9226 |     3 |  22.3563 |   16.752 |  12.4109 |    5.09159 |
  | train/loss_step  | 0.00116492 | 0.00232984 | 0.335806 |     3 | 0.669282 |  0.22387 |        0 |   0.385739 |
  | val/f1           |   0.686573 |   0.688854 | 0.689603 |     3 | 0.690352 | 0.687833 | 0.684292 | 0.00315658 |
  | val/loss         |    118.418 |    122.938 |  124.905 |     3 |  126.871 |  121.236 |  113.898 |    6.65212 |

### Coreference - truncated NER 6 + MRPC 9 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/qgr3ed02
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/3z52qo4p
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/4n2d727r

- metric values per seed

  |     | ('model_save_dir',)                                                                                                      | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/f1',) | ('train/loss_step',) | ('train/loss',) |
  | --: | :----------------------------------------------------------------------------------------------------------------------- | ----------: | --------------------: | ------------: | ------------: | -------------------: | --------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_02-48-41 |    0.681599 |               21.8659 |       98.2874 |      0.839019 |            0.0137499 |         21.8659 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_04-49-03 |    0.687848 |               13.3441 |       110.719 |      0.880905 |              8.30335 |         13.3441 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_07-52-01 |    0.690597 |               11.9279 |       120.669 |      0.897637 |           0.00518036 |         11.9279 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         |   0.859962 |  0.880905 | 0.889271 |     3 | 0.897637 |  0.87252 |   0.839019 |  0.0301948 |
  | train/loss       |     12.636 |   13.3441 |   17.605 |     3 |  21.8659 |  15.7126 |    11.9279 |    5.37572 |
  | train/loss_epoch |     12.636 |   13.3441 |   17.605 |     3 |  21.8659 |  15.7126 |    11.9279 |    5.37572 |
  | train/loss_step  | 0.00946515 | 0.0137499 |  4.15855 |     3 |  8.30335 |  2.77409 | 0.00518036 |    4.78848 |
  | val/f1           |   0.684724 |  0.687848 | 0.689223 |     3 | 0.690597 | 0.686681 |   0.681599 | 0.00461117 |
  | val/loss         |    104.503 |   110.719 |  115.694 |     3 |  120.669 |  109.892 |    98.2874 |    11.2135 |

### Coreference - truncated RE 9 + NER 6 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes] \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/qdn4yd6y
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/p4z6jr03
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/yphn2822

- metric values per seed

  |     | ('train/f1',) | ('train/loss',) | ('val/loss',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss_step',) | ('model_save_dir',)                                                                                                      |
  | --: | ------------: | --------------: | ------------: | ----------: | --------------------: | -------------------: | :----------------------------------------------------------------------------------------------------------------------- |
  |   0 |      0.822584 |         24.8673 |        107.05 |    0.672272 |               24.8673 |               28.574 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_15-56-29 |
  |   1 |      0.812565 |         27.0462 |       107.871 |    0.666689 |               27.0462 |              28.7358 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_18-00-35 |
  |   2 |      0.835555 |          22.277 |       111.308 |    0.671775 |                22.277 |              66.6735 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_19-56-19 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.817575 | 0.822584 |  0.82907 |     3 | 0.835555 | 0.823568 | 0.812565 | 0.0115267 |
  | train/loss       |  23.5721 |  24.8673 |  25.9567 |     3 |  27.0462 |  24.7302 |   22.277 |   2.38753 |
  | train/loss_epoch |  23.5721 |  24.8673 |  25.9567 |     3 |  27.0462 |  24.7302 |   22.277 |   2.38753 |
  | train/loss_step  |  28.6549 |  28.7358 |  47.7047 |     3 |  66.6735 |  41.3278 |   28.574 |   21.9502 |
  | val/f1           | 0.669232 | 0.671775 | 0.672024 |     3 | 0.672272 | 0.670246 | 0.666689 | 0.0030897 |
  | val/loss         |  107.461 |  107.871 |  109.589 |     3 |  111.308 |  108.743 |   107.05 |   2.25886 |

### Coreference - truncated SQUAD 8 + MRPC 9 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-mrpc] \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/8s6m18l6
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/g5yle97o
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/nm6sy1tc

- metric values per seed

  |     | ('train/loss',) | ('model_save_dir',)                                                                                                      | ('train/f1',) | ('train/loss_step',) | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) |
  | --: | --------------: | :----------------------------------------------------------------------------------------------------------------------- | ------------: | -------------------: | ----------: | --------------------: | ------------: |
  |   0 |         14.9055 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-13_17-01-52 |      0.884913 |           4.3749e-05 |    0.687853 |               14.9055 |       121.334 |
  |   1 |         12.1523 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-13_20-50-19 |      0.901264 |            0.0210511 |    0.694825 |               12.1523 |       122.043 |
  |   2 |          19.032 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-14_01-15-39 |      0.862891 |             0.695518 |     0.68673 |                19.032 |       112.094 |

- aggregated values:

  |                  |       25% |       50% |      75% | count |      max |     mean |        min |        std |
  | :--------------- | --------: | --------: | -------: | ----: | -------: | -------: | ---------: | ---------: |
  | train/f1         |  0.873902 |  0.884913 | 0.893088 |     3 | 0.901264 | 0.883023 |   0.862891 |  0.0192562 |
  | train/loss       |   13.5289 |   14.9055 |  16.9688 |     3 |   19.032 |  15.3633 |    12.1523 |     3.4626 |
  | train/loss_epoch |   13.5289 |   14.9055 |  16.9688 |     3 |   19.032 |  15.3633 |    12.1523 |     3.4626 |
  | train/loss_step  | 0.0105474 | 0.0210511 | 0.358284 |     3 | 0.695518 | 0.238871 | 4.3749e-05 |   0.395607 |
  | val/f1           |  0.687292 |  0.687853 | 0.691339 |     3 | 0.694825 | 0.689803 |    0.68673 | 0.00438541 |
  | val/loss         |   116.714 |   121.334 |  121.689 |     3 |  122.043 |   118.49 |    112.094 |    5.55078 |

### Coreference - truncated RE 9 + SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/oqjxs38c
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/ns9uidgk
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/pqfhrirq

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                                                      | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('train/loss',) |
  | --: | ------------: | :----------------------------------------------------------------------------------------------------------------------- | -------------------: | --------------------: | ----------: | ------------: | --------------: |
  |   0 |      0.863404 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_15-57-28 |          0.000246167 |               19.1572 |    0.684527 |       123.062 |         19.1572 |
  |   1 |      0.877275 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_18-54-22 |              5.87907 |               15.6863 |    0.678482 |       131.255 |         15.6863 |
  |   2 |      0.828583 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-09_22-14-23 |              55.1655 |               26.6568 |    0.672676 |       116.057 |         26.6568 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | --------: |
  | train/f1         | 0.845994 | 0.863404 |  0.87034 |     3 | 0.877275 | 0.856421 |    0.828583 | 0.0250859 |
  | train/loss       |  17.4218 |  19.1572 |   22.907 |     3 |  26.6568 |  20.5001 |     15.6863 |   5.60719 |
  | train/loss_epoch |  17.4218 |  19.1572 |   22.907 |     3 |  26.6568 |  20.5001 |     15.6863 |   5.60719 |
  | train/loss_step  |  2.93966 |  5.87907 |  30.5223 |     3 |  55.1655 |  20.3483 | 0.000246167 |   30.2955 |
  | val/f1           | 0.675579 | 0.678482 | 0.681504 |     3 | 0.684527 | 0.678562 |    0.672676 | 0.0059257 |
  | val/loss         |   119.56 |  123.062 |  127.159 |     3 |  131.255 |  123.458 |     116.057 |   7.60683 |

### Coreference - truncated NER 6 + SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/2x_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/kd8otwfx
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/jywyn7h0
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-2x_models_truncated-training/runs/ldug2who

- metric values per seed

  |     | ('val/loss',) | ('train/loss',) | ('train/loss_step',) | ('model_save_dir',)                                                                                                      | ('train/loss_epoch',) | ('train/f1',) | ('val/f1',) |
  | --: | ------------: | --------------: | -------------------: | :----------------------------------------------------------------------------------------------------------------------- | --------------------: | ------------: | ----------: |
  |   0 |       103.441 |         23.0581 |              1.80032 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_00-26-29 |               23.0581 |      0.831629 |    0.676598 |
  |   1 |       117.455 |          13.192 |              14.7587 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_02-20-29 |                13.192 |      0.883417 |    0.682009 |
  |   2 |       99.2379 |         25.4688 |              17.7034 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/2x_models_truncated/2024-01-10_05-19-57 |               25.4688 |       0.81897 |    0.670935 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.825299 | 0.831629 | 0.857523 |     3 | 0.883417 | 0.844672 |  0.81897 |  0.034146 |
  | train/loss       |   18.125 |  23.0581 |  24.2635 |     3 |  25.4688 |   20.573 |   13.192 |   6.50475 |
  | train/loss_epoch |   18.125 |  23.0581 |  24.2635 |     3 |  25.4688 |   20.573 |   13.192 |   6.50475 |
  | train/loss_step  |   8.2795 |  14.7587 |  16.2311 |     3 |  17.7034 |  11.4208 |  1.80032 |    8.4607 |
  | val/f1           | 0.673766 | 0.676598 | 0.679303 |     3 | 0.682009 | 0.676514 | 0.670935 | 0.0055373 |
  | val/loss         |  101.339 |  103.441 |  110.448 |     3 |  117.455 |  106.711 |  99.2379 |   9.53862 |

### Coreference - truncated RE 9 + NER 6 + SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/z4406ufz
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/n24g5x62
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/prf6hc6y

- metric values per seed

  |     | ('train/loss',) | ('train/loss_epoch',) | ('val/loss',) | ('val/f1',) | ('train/f1',) | ('model_save_dir',)                                                                                                            | ('train/loss_step',) |
  | --: | --------------: | --------------------: | ------------: | ----------: | ------------: | :----------------------------------------------------------------------------------------------------------------------------- | -------------------: |
  |   0 |         11.5312 |               11.5312 |       100.639 |    0.691557 |      0.889118 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-10_22-02-29 |          4.60139e-05 |
  |   1 |          10.957 |                10.957 |       105.426 |    0.691665 |      0.892625 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_00-58-55 |               55.457 |
  |   2 |         10.9453 |               10.9453 |       102.421 |    0.686155 |       0.89019 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_03-56-06 |            0.0106071 |

- aggregated values:

  |                  |        25% |       50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | ---------: | --------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |   0.889654 |   0.89019 | 0.891407 |     3 | 0.892625 | 0.890644 |    0.889118 | 0.00179712 |
  | train/loss       |    10.9511 |    10.957 |  11.2441 |     3 |  11.5312 |  11.1445 |     10.9453 |   0.334967 |
  | train/loss_epoch |    10.9511 |    10.957 |  11.2441 |     3 |  11.5312 |  11.1445 |     10.9453 |   0.334967 |
  | train/loss_step  | 0.00532657 | 0.0106071 |  27.7338 |     3 |   55.457 |  18.4892 | 4.60139e-05 |     32.015 |
  | val/f1           |   0.688856 |  0.691557 | 0.691611 |     3 | 0.691665 | 0.689792 |    0.686155 | 0.00315073 |
  | val/loss         |     101.53 |   102.421 |  103.924 |     3 |  105.426 |  102.829 |     100.639 |    2.41943 |

### Coreference - truncated MRPC 9 + NER 6 + SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/11cwzpms
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/g2wnilf3
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/pb05eken

- metric values per seed

  |     | ('model_save_dir',)                                                                                                            | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('train/f1',) |
  | --: | :----------------------------------------------------------------------------------------------------------------------------- | --------------: | -------------------: | --------------------: | ----------: | ------------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-10_22-02-46 |         12.9102 |          0.000524802 |               12.9102 |    0.692292 |       94.2972 |      0.881769 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_00-29-47 |         11.3958 |                    0 |               11.3958 |     0.69522 |       95.0837 |       0.89142 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_03-10-44 |         7.77698 |              11.6139 |               7.77698 |    0.699251 |       109.359 |       0.91746 |

- aggregated values:

  |                  |         25% |         50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ----------: | ----------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |    0.886594 |     0.89142 |  0.90444 |     3 |  0.91746 | 0.896883 | 0.881769 |  0.0184624 |
  | train/loss       |      9.5864 |     11.3958 |   12.153 |     3 |  12.9102 |  10.6943 |  7.77698 |    2.63752 |
  | train/loss_epoch |      9.5864 |     11.3958 |   12.153 |     3 |  12.9102 |  10.6943 |  7.77698 |    2.63752 |
  | train/loss_step  | 0.000262401 | 0.000524802 |  5.80723 |     3 |  11.6139 |  3.87149 |        0 |    6.70516 |
  | val/f1           |    0.693756 |     0.69522 | 0.697235 |     3 | 0.699251 | 0.695588 | 0.692292 | 0.00349365 |
  | val/loss         |     94.6904 |     95.0837 |  102.222 |     3 |  109.359 |  99.5801 |  94.2972 |    8.47828 |

### Coreference - truncated MRPC 9 + RE 9 + SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-qa-squad2,bert-base-cased-re-tacred,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/jj0esijz
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/nee2rm5o
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/6d0arck9

- metric values per seed

  |     | ('train/loss',) | ('model_save_dir',)                                                                                                            | ('train/loss_step',) | ('train/f1',) | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) |
  | --: | --------------: | :----------------------------------------------------------------------------------------------------------------------------- | -------------------: | ------------: | ------------: | --------------------: | ----------: |
  |   0 |         17.1282 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_06-57-27 |               9.0835 |      0.862693 |       90.3379 |               17.1282 |     0.68655 |
  |   1 |         11.0579 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_09-20-52 |            0.0636393 |      0.892443 |       104.173 |               11.0579 |    0.695658 |
  |   2 |         22.8276 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_12-30-48 |              5.30443 |      0.824932 |       83.9837 |               22.8276 |    0.682144 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ---------: |
  | train/f1         | 0.843813 | 0.862693 | 0.877568 |     3 | 0.892443 | 0.860023 |  0.824932 |  0.0338345 |
  | train/loss       |  14.0931 |  17.1282 |  19.9779 |     3 |  22.8276 |  17.0046 |   11.0579 |    5.88579 |
  | train/loss_epoch |  14.0931 |  17.1282 |  19.9779 |     3 |  22.8276 |  17.0046 |   11.0579 |    5.88579 |
  | train/loss_step  |  2.68403 |  5.30443 |  7.19396 |     3 |   9.0835 |  4.81719 | 0.0636393 |    4.52963 |
  | val/f1           | 0.684347 |  0.68655 | 0.691104 |     3 | 0.695658 | 0.688117 |  0.682144 | 0.00689193 |
  | val/loss         |  87.1608 |  90.3379 |  97.2552 |     3 |  104.173 |  92.8314 |   83.9837 |    10.3228 |

### Coreference - truncated MRPC 9 + RE 9 + NER 6 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  model.task_learning_rate=1e-4 \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  trainer=gpu \
  seed=1,2,3 \
  +wandb_watch=attention_activation \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/multiple_models_truncated \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/pocmjtcl
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/4a62hnkz
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-multiple_models_truncated-training/runs/da34dlou

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                                            | ('train/f1',) |
  | --: | ----------: | ------------: | --------------: | -------------------: | --------------------: | :----------------------------------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |    0.687907 |        85.726 |         17.6838 |              10.8248 |               17.6838 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_06-53-17 |       0.85072 |
  |   1 |    0.692745 |       101.086 |         10.8913 |              52.7882 |               10.8913 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_08-53-26 |      0.894107 |
  |   2 |    0.698249 |       105.338 |         10.2981 |              4.85269 |               10.2981 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/multiple_models_truncated/2024-01-11_11-49-11 |      0.899055 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.872413 | 0.894107 | 0.896581 |     3 | 0.899055 | 0.881294 |  0.85072 |  0.0265931 |
  | train/loss       |  10.5947 |  10.8913 |  14.2875 |     3 |  17.6838 |  12.9577 |  10.2981 |     4.1036 |
  | train/loss_epoch |  10.5947 |  10.8913 |  14.2875 |     3 |  17.6838 |  12.9577 |  10.2981 |     4.1036 |
  | train/loss_step  |  7.83874 |  10.8248 |  31.8065 |     3 |  52.7882 |  22.8219 |  4.85269 |    26.1228 |
  | val/f1           | 0.690326 | 0.692745 | 0.695497 |     3 | 0.698249 | 0.692967 | 0.687907 | 0.00517453 |
  | val/loss         |   93.406 |  101.086 |  103.212 |     3 |  105.338 |  97.3832 |   85.726 |    10.3168 |

### Coreference - truncated MRPC 9 + RE 9 + NER 6 + SQUAD 8 with attention aggregation

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc,bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf,bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-mrpc,bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes,bert-base-cased-qa-squad2] \
  model.task_learning_rate=1e-4 \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.truncate_models.bert-base-cased-mrpc=9 \
  +model.truncate_models.bert-base-cased-re-tacred=9 \
  +model.truncate_models.bert-base-cased-ner-ontonotes=6 \
  +model.truncate_models.bert-base-cased-qa-squad2=8 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_aggregation_frozen/all_models \
  --multirun
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/brn2fn2i
  - seed2: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/8pzpsd5g
  - seed3: https://wandb.ai/tanikina/attn_aggregation_frozen-all_models-training/runs/vod0pt56

- metric values per seed

  |     | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('val/f1',) | ('val/loss',) | ('model_save_dir',)                                                                                             | ('train/f1',) |
  | --: | --------------: | -------------------: | --------------------: | ----------: | ------------: | :-------------------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |         16.2789 |                    0 |               16.2789 |    0.685539 |       75.8673 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-10_22-17-19 |      0.853332 |
  |   1 |         9.95199 |          0.000101566 |               9.95199 |    0.699662 |       93.0262 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_00-07-08 |      0.907383 |
  |   2 |         12.0724 |              6.18143 |               12.0724 |    0.696461 |       82.3966 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_aggregation_frozen/all_models/2024-01-11_03-14-32 |      0.882372 |

- aggregated values:

  |                  |        25% |         50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | ---------: | ----------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         |   0.867852 |    0.882372 | 0.894877 |     3 | 0.907383 | 0.881029 | 0.853332 | 0.0270503 |
  | train/loss       |    11.0122 |     12.0724 |  14.1757 |     3 |  16.2789 |  12.7678 |  9.95199 |   3.22028 |
  | train/loss_epoch |    11.0122 |     12.0724 |  14.1757 |     3 |  16.2789 |  12.7678 |  9.95199 |   3.22028 |
  | train/loss_step  | 5.0783e-05 | 0.000101566 |  3.09077 |     3 |  6.18143 |  2.06051 |        0 |   3.56882 |
  | val/f1           |      0.691 |    0.696461 | 0.698062 |     3 | 0.699662 | 0.693888 | 0.685539 | 0.0074047 |
  | val/loss         |    79.1319 |     82.3966 |  87.7114 |     3 |  93.0262 |  83.7634 |  75.8673 |   8.66072 |

## 2024-01-21

### Coreference - frozen MRPC with attention aggregation and concatenated 4 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=4 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/4i5xarbm
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/yhtvhamf
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/k7j6q1ll

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('train/loss_step',) | ('train/loss',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                    | ('train/f1',) |
  | --: | ----------: | ------------: | -------------------: | --------------: | --------------------: | :----------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |    0.664048 |       101.007 |              49.3451 |         45.7235 |               45.7235 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_10-44-38 |      0.747828 |
  |   1 |    0.694805 |       110.684 |              5.25962 |         19.2384 |               19.2384 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_12-27-15 |      0.862717 |
  |   2 |    0.679466 |        101.98 |                    0 |         34.4092 |               34.4092 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_16-41-26 |      0.788785 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.768307 | 0.788785 | 0.825751 |     3 | 0.862717 | 0.799777 | 0.747828 | 0.0582276 |
  | train/loss       |  26.8238 |  34.4092 |  40.0663 |     3 |  45.7235 |  33.1237 |  19.2384 |   13.2893 |
  | train/loss_epoch |  26.8238 |  34.4092 |  40.0663 |     3 |  45.7235 |  33.1237 |  19.2384 |   13.2893 |
  | train/loss_step  |  2.62981 |  5.25962 |  27.3024 |     3 |  49.3451 |  18.2016 |        0 |    27.099 |
  | val/f1           | 0.671757 | 0.679466 | 0.687136 |     3 | 0.694805 |  0.67944 | 0.664048 | 0.0153786 |
  | val/loss         |  101.494 |   101.98 |  106.332 |     3 |  110.684 |  104.557 |  101.007 |   5.32826 |

### Coreference - frozen MRPC with attention aggregation and concatenated 6 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=6 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/i9v8hvjr
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/3062dnf1
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/w9bqrze7

- metric values per seed

  |     | ('val/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('val/f1',) | ('model_save_dir',)                                                                                    | ('train/loss',) | ('train/loss_step',) |
  | --: | ------------: | ------------: | --------------------: | ----------: | :----------------------------------------------------------------------------------------------------- | --------------: | -------------------: |
  |   0 |       124.896 |      0.797882 |                38.889 |    0.679575 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_19-05-40 |          38.889 |              183.633 |
  |   1 |       122.198 |      0.757067 |               50.9917 |    0.669273 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_21-25-47 |         50.9917 |              9.62221 |
  |   2 |       117.428 |      0.747572 |               54.1177 |    0.662145 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_23-03-48 |         54.1177 |              10.7762 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.752319 | 0.757067 | 0.777475 |     3 | 0.797882 | 0.767507 | 0.747572 |  0.0267306 |
  | train/loss       |  44.9403 |  50.9917 |  52.5547 |     3 |  54.1177 |  47.9995 |   38.889 |    8.04322 |
  | train/loss_epoch |  44.9403 |  50.9917 |  52.5547 |     3 |  54.1177 |  47.9995 |   38.889 |    8.04322 |
  | train/loss_step  |  10.1992 |  10.7762 |  97.2049 |     3 |  183.633 |  68.0106 |  9.62221 |    100.134 |
  | val/f1           | 0.665709 | 0.669273 | 0.674424 |     3 | 0.679575 | 0.670331 | 0.662145 | 0.00876315 |
  | val/loss         |  119.813 |  122.198 |  123.547 |     3 |  124.896 |  121.507 |  117.428 |    3.78159 |

### Coreference - frozen MRPC with attention aggregation and concatenated 12 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-mrpc:bert-base-cased-finetuned-mrpc} \
  +model.freeze_models=[bert-base-cased-mrpc] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=12 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/pwrroqoo
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/3vu10an2
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/i2l59sep

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/loss_step',) | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('model_save_dir',)                                                                                    | ('val/loss',) |
  | --: | --------------------: | -------------------: | ----------: | --------------: | ------------: | :----------------------------------------------------------------------------------------------------- | ------------: |
  |   0 |               69.8405 |              1.07724 |     0.66177 |         69.8405 |      0.762128 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_00-34-07 |       174.871 |
  |   1 |                24.541 |              2.59332 |    0.689684 |          24.541 |      0.888773 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_02-16-54 |       221.314 |
  |   2 |               59.8391 |              181.142 |    0.666967 |         59.8391 |      0.788781 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_07-04-08 |       185.181 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.775454 | 0.788781 | 0.838777 |     3 | 0.888773 | 0.813227 | 0.762128 | 0.0667681 |
  | train/loss       |    42.19 |  59.8391 |  64.8398 |     3 |  69.8405 |  51.4069 |   24.541 |   23.7979 |
  | train/loss_epoch |    42.19 |  59.8391 |  64.8398 |     3 |  69.8405 |  51.4069 |   24.541 |   23.7979 |
  | train/loss_step  |  1.83528 |  2.59332 |  91.8678 |     3 |  181.142 |  61.6043 |  1.07724 |   103.526 |
  | val/f1           | 0.664368 | 0.666967 | 0.678325 |     3 | 0.689684 | 0.672807 |  0.66177 | 0.0148451 |
  | val/loss         |  180.026 |  185.181 |  203.248 |     3 |  221.314 |  193.789 |  174.871 |    24.389 |

### Coreference - frozen NER with attention aggregation and concatenated 4 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=4 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/fn9thpmi
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/x6jle2t5
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/hck0fp90

- metric values per seed

  |     | ('train/loss',) | ('val/loss',) | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                                    | ('train/loss_epoch',) | ('val/f1',) |
  | --: | --------------: | ------------: | -------------------: | ------------: | :----------------------------------------------------------------------------------------------------- | --------------------: | ----------: |
  |   0 |         54.5663 |       92.8081 |              25.7867 |      0.668849 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_10-31-46 |               54.5663 |    0.640081 |
  |   1 |         63.7297 |       93.9674 |              98.8756 |      0.639965 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_13-01-39 |               63.7297 |    0.630029 |
  |   2 |         57.5577 |       90.5343 |              208.587 |      0.659314 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_14-50-14 |               57.5577 |    0.643793 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.649639 | 0.659314 | 0.664082 |     3 | 0.668849 | 0.656043 | 0.639965 |  0.0147176 |
  | train/loss       |   56.062 |  57.5577 |  60.6437 |     3 |  63.7297 |  58.6179 |  54.5663 |    4.67281 |
  | train/loss_epoch |   56.062 |  57.5577 |  60.6437 |     3 |  63.7297 |  58.6179 |  54.5663 |    4.67281 |
  | train/loss_step  |  62.3312 |  98.8756 |  153.731 |     3 |  208.587 |  111.083 |  25.7867 |    92.0093 |
  | val/f1           | 0.635055 | 0.640081 | 0.641937 |     3 | 0.643793 | 0.637968 | 0.630029 | 0.00712142 |
  | val/loss         |  91.6712 |  92.8081 |  93.3877 |     3 |  93.9674 |  92.4366 |  90.5343 |    1.74642 |

### Coreference - frozen NER with attention aggregation and concatenated 6 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=6 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/sukdj6ra
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/vu1rox9p
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/5ciab7f7

- metric values per seed

  |     | ('model_save_dir',)                                                                                    | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('train/loss_step',) | ('val/loss',) |
  | --: | :----------------------------------------------------------------------------------------------------- | --------------------: | ----------: | --------------: | ------------: | -------------------: | ------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_17-00-22 |               38.8587 |    0.668206 |         38.8587 |       0.76878 |              8.89372 |       112.569 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_20-10-38 |               44.8094 |    0.662646 |         44.8094 |       0.74644 |          2.74181e-06 |       108.573 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_22-55-22 |               52.0982 |    0.658076 |         52.0982 |      0.720741 |              209.827 |       108.121 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |  0.73359 |  0.74644 |  0.75761 |     3 |  0.76878 |  0.74532 |    0.720741 |   0.024039 |
  | train/loss       |  41.8341 |  44.8094 |  48.4538 |     3 |  52.0982 |  45.2554 |     38.8587 |    6.63099 |
  | train/loss_epoch |  41.8341 |  44.8094 |  48.4538 |     3 |  52.0982 |  45.2554 |     38.8587 |    6.63099 |
  | train/loss_step  |  4.44686 |  8.89372 |  109.361 |     3 |  209.827 |  72.9071 | 2.74181e-06 |     118.66 |
  | val/f1           | 0.660361 | 0.662646 | 0.665426 |     3 | 0.668206 | 0.662976 |    0.658076 | 0.00507321 |
  | val/loss         |  108.347 |  108.573 |  110.571 |     3 |  112.569 |  109.755 |     108.121 |    2.44798 |

### Coreference - frozen NER with attention aggregation and concatenated 12 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-ner-ontonotes:models/pretrained/bert-base-cased-ner-ontonotes} \
  +model.freeze_models=[bert-base-cased-ner-ontonotes] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=12 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/vped4nqa
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/z2ba4mfd
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/vsfsz3kn

- metric values per seed

  |     | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                    | ('val/loss',) | ('val/f1',) | ('train/loss',) |
  | --: | -------------------: | ------------: | --------------------: | :----------------------------------------------------------------------------------------------------- | ------------: | ----------: | --------------: |
  |   0 |              46.4263 |      0.783929 |               54.4985 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_01-11-00 |       177.994 |     0.66153 |         54.4985 |
  |   1 |              7.93501 |      0.716959 |               81.0539 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_04-01-45 |       156.122 |    0.646759 |         81.0539 |
  |   2 |              40.0124 |      0.771052 |               60.8142 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_05-43-13 |       168.475 |    0.664384 |         60.8142 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.744005 | 0.771052 |  0.77749 |     3 | 0.783929 | 0.757313 | 0.716959 |   0.035536 |
  | train/loss       |  57.6564 |  60.8142 |  70.9341 |     3 |  81.0539 |  65.4555 |  54.4985 |    13.8728 |
  | train/loss_epoch |  57.6564 |  60.8142 |  70.9341 |     3 |  81.0539 |  65.4555 |  54.4985 |    13.8728 |
  | train/loss_step  |  23.9737 |  40.0124 |  43.2194 |     3 |  46.4263 |  31.4579 |  7.93501 |    20.6223 |
  | val/f1           | 0.654145 |  0.66153 | 0.662957 |     3 | 0.664384 | 0.657558 | 0.646759 | 0.00946011 |
  | val/loss         |  162.298 |  168.475 |  173.234 |     3 |  177.994 |   167.53 |  156.122 |    10.9667 |

### Coreference - frozen RE with attention aggregation and concatenated 4 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=4 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/34dbd615
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/e3erl8ot
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/kjcpm37y

- metric values per seed

  |     | ('train/loss',) | ('val/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss_step',) | ('train/f1',) | ('model_save_dir',)                                                                                    |
  | --: | --------------: | ----------: | --------------------: | ------------: | -------------------: | ------------: | :----------------------------------------------------------------------------------------------------- |
  |   0 |         48.8827 |    0.639317 |               48.8827 |       119.402 |            0.0232261 |      0.729895 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_10-45-02 |
  |   1 |         27.1018 |    0.656574 |               27.1018 |       108.418 |              130.974 |      0.806321 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_13-41-48 |
  |   2 |          24.833 |    0.658777 |                24.833 |       108.955 |              50.1255 |      0.817804 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_18-47-42 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | --------: |
  | train/f1         | 0.768108 | 0.806321 | 0.812063 |     3 | 0.817804 | 0.784674 |  0.729895 | 0.0477857 |
  | train/loss       |  25.9674 |  27.1018 |  37.9922 |     3 |  48.8827 |  33.6058 |    24.833 |   13.2787 |
  | train/loss_epoch |  25.9674 |  27.1018 |  37.9922 |     3 |  48.8827 |  33.6058 |    24.833 |   13.2787 |
  | train/loss_step  |  25.0744 |  50.1255 |  90.5495 |     3 |  130.974 |  60.3741 | 0.0232261 |    66.074 |
  | val/f1           | 0.647946 | 0.656574 | 0.657675 |     3 | 0.658777 | 0.651556 |  0.639317 | 0.0106564 |
  | val/loss         |  108.687 |  108.955 |  114.179 |     3 |  119.402 |  112.258 |   108.418 |   6.19213 |

### Coreference - frozen RE with attention aggregation and concatenated 6 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=6 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/apenr35o
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/cg5a7cdt
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/hseg3vd1

- metric values per seed

  |     | ('val/f1',) | ('model_save_dir',)                                                                                    | ('val/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('train/f1',) | ('train/loss',) |
  | --: | ----------: | :----------------------------------------------------------------------------------------------------- | ------------: | -------------------: | --------------------: | ------------: | --------------: |
  |   0 |    0.654154 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_00-11-24 |       147.041 |              5.13627 |               44.8758 |      0.778526 |         44.8758 |
  |   1 |    0.658104 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_03-22-26 |       143.574 |              171.143 |               33.6778 |      0.812206 |         33.6778 |
  |   2 |    0.661575 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_07-26-05 |       141.451 |              1.03448 |               38.1312 |      0.800203 |         38.1312 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         | 0.789365 | 0.800203 | 0.806205 |     3 | 0.812206 | 0.796979 | 0.778526 | 0.0170702 |
  | train/loss       |  35.9045 |  38.1312 |  41.5035 |     3 |  44.8758 |  38.8949 |  33.6778 |   5.63795 |
  | train/loss_epoch |  35.9045 |  38.1312 |  41.5035 |     3 |  44.8758 |  38.8949 |  33.6778 |   5.63795 |
  | train/loss_step  |  3.08538 |  5.13627 |  88.1395 |     3 |  171.143 |  59.1045 |  1.03448 |   97.0496 |
  | val/f1           | 0.656129 | 0.658104 | 0.659839 |     3 | 0.661575 | 0.657944 | 0.654154 | 0.0037132 |
  | val/loss         |  142.512 |  143.574 |  145.307 |     3 |  147.041 |  144.022 |  141.451 |   2.82161 |

### Coreference - frozen RE with attention aggregation and concatenated 12 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-re-tacred:models/pretrained/bert-base-cased-re-tacred-20230919-hf} \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=12 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/68zts0cw
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/45300atj
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/iuw7bky9

- metric values per seed

  |     | ('train/loss_step',) | ('val/f1',) | ('model_save_dir',)                                                                                    | ('train/loss_epoch',) | ('train/f1',) | ('train/loss',) | ('val/loss',) |
  | --: | -------------------: | ----------: | :----------------------------------------------------------------------------------------------------- | --------------------: | ------------: | --------------: | ------------: |
  |   0 |                    0 |    0.664746 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_11-07-46 |               38.1408 |      0.846727 |         38.1408 |       232.087 |
  |   1 |              52.0429 |    0.647033 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_15-17-53 |               78.8113 |      0.752112 |         78.8113 |       197.573 |
  |   2 |                    0 |    0.651102 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_17-19-01 |               67.2695 |       0.78195 |         67.2695 |       211.988 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.767031 |  0.78195 | 0.814339 |     3 | 0.846727 | 0.793597 | 0.752112 |   0.048371 |
  | train/loss       |  52.7051 |  67.2695 |  73.0404 |     3 |  78.8113 |  61.4072 |  38.1408 |    20.9595 |
  | train/loss_epoch |  52.7051 |  67.2695 |  73.0404 |     3 |  78.8113 |  61.4072 |  38.1408 |    20.9595 |
  | train/loss_step  |        0 |        0 |  26.0215 |     3 |  52.0429 |  17.3476 |        0 |     30.047 |
  | val/f1           | 0.649068 | 0.651102 | 0.657924 |     3 | 0.664746 | 0.654294 | 0.647033 | 0.00927776 |
  | val/loss         |  204.781 |  211.988 |  222.038 |     3 |  232.087 |  213.883 |  197.573 |    17.3348 |

### Coreference - frozen SQUAD with attention aggregation and concatenated 4 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=4 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/dauur05p
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/dw19ggyx
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/tjx2e2oo

- metric values per seed

  |     | ('val/loss',) | ('train/loss',) | ('val/f1',) | ('train/f1',) | ('train/loss_step',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                    |
  | --: | ------------: | --------------: | ----------: | ------------: | -------------------: | --------------------: | :----------------------------------------------------------------------------------------------------- |
  |   0 |       120.684 |         45.1503 |     0.66758 |      0.765532 |               3.0915 |               45.1503 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_10-45-23 |
  |   1 |       116.149 |           40.42 |    0.677394 |      0.785899 |              6.08835 |                 40.42 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_13-21-07 |
  |   2 |       116.118 |         24.9727 |    0.686125 |       0.84036 |              44.1816 |               24.9727 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_16-22-35 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.775716 | 0.785899 | 0.813129 |     3 |  0.84036 | 0.797264 | 0.765532 |  0.0386866 |
  | train/loss       |  32.6964 |    40.42 |  42.7851 |     3 |  45.1503 |  36.8477 |  24.9727 |    10.5525 |
  | train/loss_epoch |  32.6964 |    40.42 |  42.7851 |     3 |  45.1503 |  36.8477 |  24.9727 |    10.5525 |
  | train/loss_step  |  4.58992 |  6.08835 |   25.135 |     3 |  44.1816 |  17.7872 |   3.0915 |    22.9073 |
  | val/f1           | 0.672487 | 0.677394 |  0.68176 |     3 | 0.686125 | 0.677033 |  0.66758 | 0.00927778 |
  | val/loss         |  116.134 |  116.149 |  118.417 |     3 |  120.684 |  117.651 |  116.118 |    2.62731 |

### Coreference - frozen SQUAD with attention aggregation and concatenated 6 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=6 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/0m1h5swp
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/vwvu18x1
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/9kdtx2r4

- metric values per seed

  |     | ('train/f1',) | ('model_save_dir',)                                                                                    | ('val/loss',) | ('train/loss_epoch',) | ('val/f1',) | ('train/loss',) | ('train/loss_step',) |
  | --: | ------------: | :----------------------------------------------------------------------------------------------------- | ------------: | --------------------: | ----------: | --------------: | -------------------: |
  |   0 |       0.75368 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_20-58-56 |       133.999 |               60.2839 |    0.667883 |         60.2839 |              45.9256 |
  |   1 |      0.761493 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_22-47-35 |       136.957 |               55.8497 |    0.667288 |         55.8497 |          5.96045e-06 |
  |   2 |      0.765737 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_00-45-06 |       140.067 |               54.7837 |    0.669392 |         54.7837 |           0.00511126 |

- aggregated values:

  |                  |        25% |        50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | ---------: | ---------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         |   0.757587 |   0.761493 | 0.763615 |     3 | 0.765737 | 0.760303 |     0.75368 | 0.00611572 |
  | train/loss       |    55.3167 |    55.8497 |  58.0668 |     3 |  60.2839 |  56.9725 |     54.7837 |     2.9169 |
  | train/loss_epoch |    55.3167 |    55.8497 |  58.0668 |     3 |  60.2839 |  56.9725 |     54.7837 |     2.9169 |
  | train/loss_step  | 0.00255861 | 0.00511126 |  22.9653 |     3 |  45.9256 |  15.3102 | 5.96045e-06 |    26.5137 |
  | val/f1           |   0.667585 |   0.667883 | 0.668638 |     3 | 0.669392 | 0.668188 |    0.667288 |  0.0010849 |
  | val/loss         |    135.478 |    136.957 |  138.512 |     3 |  140.067 |  137.008 |     133.999 |    3.03425 |

### Coreference - frozen SQUAD with attention aggregation and concatenated 12 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-qa-squad2:models/pretrained/bert-base-cased-qa-squad2} \
  +model.freeze_models=[bert-base-cased-qa-squad2] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=12 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/2f1mm8hq
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/0ylca1gs
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/hj48kskc

- metric values per seed

  |     | ('model_save_dir',)                                                                                    | ('val/loss',) | ('val/f1',) | ('train/loss',) | ('train/loss_epoch',) | ('train/f1',) | ('train/loss_step',) |
  | --: | :----------------------------------------------------------------------------------------------------- | ------------: | ----------: | --------------: | --------------------: | ------------: | -------------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_02-47-25 |       195.965 |     0.66715 |         67.7705 |               67.7705 |      0.782801 |          1.62125e-05 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_04-59-38 |       229.721 |    0.684305 |         38.4524 |               38.4524 |      0.852553 |              5.41149 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_08-47-47 |       215.923 |    0.677509 |         50.2297 |               50.2297 |      0.827375 |              102.129 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |         min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | ----------: | ---------: |
  | train/f1         | 0.805088 | 0.827375 | 0.839964 |     3 | 0.852553 |  0.82091 |    0.782801 |  0.0353225 |
  | train/loss       |   44.341 |  50.2297 |  59.0001 |     3 |  67.7705 |  52.1509 |     38.4524 |    14.7532 |
  | train/loss_epoch |   44.341 |  50.2297 |  59.0001 |     3 |  67.7705 |  52.1509 |     38.4524 |    14.7532 |
  | train/loss_step  |  2.70575 |  5.41149 |  53.7701 |     3 |  102.129 |  35.8468 | 1.62125e-05 |    57.4657 |
  | val/f1           | 0.672329 | 0.677509 | 0.680907 |     3 | 0.684305 | 0.676321 |     0.66715 | 0.00863923 |
  | val/loss         |  205.944 |  215.923 |  222.822 |     3 |  229.721 |   213.87 |     195.965 |    16.9716 |

### Coreference - frozen BERT with attention aggregation and concatenated 4 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=4 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/l2crooku
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/pv7av5wn
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/07w83f3f

- metric values per seed

  |     | ('model_save_dir',)                                                                                    | ('val/loss',) | ('train/loss_step',) | ('train/f1',) | ('val/f1',) | ('train/loss_epoch',) | ('train/loss',) |
  | --: | :----------------------------------------------------------------------------------------------------- | ------------: | -------------------: | ------------: | ----------: | --------------------: | --------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_10-45-38 |       125.756 |              22.6774 |      0.863152 |    0.689405 |               21.2281 |         21.2281 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_14-03-42 |       131.994 |            0.0119067 |      0.897921 |    0.696015 |               14.3194 |         14.3194 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_18-30-49 |       122.458 |             0.505478 |      0.846038 |    0.686447 |               23.8949 |         23.8949 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |       min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | --------: |
  | train/f1         | 0.854595 | 0.863152 | 0.880537 |     3 | 0.897921 | 0.869037 |  0.846038 | 0.0264374 |
  | train/loss       |  17.7738 |  21.2281 |  22.5615 |     3 |  23.8949 |  19.8141 |   14.3194 |   4.94182 |
  | train/loss_epoch |  17.7738 |  21.2281 |  22.5615 |     3 |  23.8949 |  19.8141 |   14.3194 |   4.94182 |
  | train/loss_step  | 0.258692 | 0.505478 |  11.5914 |     3 |  22.6774 |  7.73158 | 0.0119067 |   12.9458 |
  | val/f1           | 0.687926 | 0.689405 |  0.69271 |     3 | 0.696015 | 0.690622 |  0.686447 | 0.0048992 |
  | val/loss         |  124.107 |  125.756 |  128.875 |     3 |  131.994 |  126.736 |   122.458 |   4.84325 |

### Coreference - frozen BERT with attention aggregation and concatenated 6 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=6 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/ofdvmz6z
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/a80s8k5z
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/673y8z9a

- metric values per seed

  |     | ('model_save_dir',)                                                                                    | ('val/f1',) | ('train/f1',) | ('train/loss_epoch',) | ('val/loss',) | ('train/loss_step',) | ('train/loss',) |
  | --: | :----------------------------------------------------------------------------------------------------- | ----------: | ------------: | --------------------: | ------------: | -------------------: | --------------: |
  |   0 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_21-19-32 |    0.695671 |      0.857633 |               25.2781 |       141.879 |              75.0592 |         25.2781 |
  |   1 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_00-16-50 |    0.677523 |      0.808067 |               38.5612 |       125.644 |              16.3168 |         38.5612 |
  |   2 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_02-10-06 |    0.698753 |      0.880033 |               21.2462 |       148.136 |              45.7036 |         21.2462 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
  | train/f1         |  0.83285 | 0.857633 | 0.868833 |     3 | 0.880033 | 0.848578 | 0.808067 | 0.0368276 |
  | train/loss       |  23.2621 |  25.2781 |  31.9197 |     3 |  38.5612 |  28.3618 |  21.2462 |   9.06007 |
  | train/loss_epoch |  23.2621 |  25.2781 |  31.9197 |     3 |  38.5612 |  28.3618 |  21.2462 |   9.06007 |
  | train/loss_step  |  31.0102 |  45.7036 |  60.3814 |     3 |  75.0592 |  45.6932 |  16.3168 |   29.3712 |
  | val/f1           | 0.686597 | 0.695671 | 0.697212 |     3 | 0.698753 | 0.690649 | 0.677523 | 0.0114715 |
  | val/loss         |  133.762 |  141.879 |  145.008 |     3 |  148.136 |  138.553 |  125.644 |   11.6093 |

### Coreference - frozen BERT with attention aggregation and concatenated 12 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased:bert-base-cased} \
  +model.freeze_models=[bert-base-cased] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=12 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/fzdi84qo
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/imtlx5l9
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/bh9srtek

- metric values per seed

  |     | ('val/f1',) | ('train/loss',) | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                    | ('val/loss',) | ('train/loss_step',) |
  | --: | ----------: | --------------: | ------------: | --------------------: | :----------------------------------------------------------------------------------------------------- | ------------: | -------------------: |
  |   0 |    0.688317 |         32.8334 |      0.868272 |               32.8334 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_05-48-06 |       218.317 |          1.33514e-05 |
  |   1 |     0.67782 |         55.1053 |      0.803665 |               55.1053 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_09-18-16 |       171.189 |                    0 |
  |   2 |    0.697659 |         28.7428 |      0.885147 |               28.7428 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-14_11-21-22 |        222.47 |              8.14223 |

- aggregated values:

  |                  |         25% |         50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | ----------: | ----------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         |    0.835968 |    0.868272 |  0.87671 |     3 | 0.885147 | 0.852361 | 0.803665 |  0.0430083 |
  | train/loss       |     30.7881 |     32.8334 |  43.9693 |     3 |  55.1053 |  38.8938 |  28.7428 |    14.1877 |
  | train/loss_epoch |     30.7881 |     32.8334 |  43.9693 |     3 |  55.1053 |  38.8938 |  28.7428 |    14.1877 |
  | train/loss_step  | 6.67572e-06 | 1.33514e-05 |  4.07112 |     3 |  8.14223 |  2.71408 |        0 |    4.70091 |
  | val/f1           |    0.683068 |    0.688317 | 0.692988 |     3 | 0.697659 | 0.687932 |  0.67782 | 0.00992536 |
  | val/loss         |     194.753 |     218.317 |  220.394 |     3 |   222.47 |  203.992 |  171.189 |    28.4843 |

### Coreference - frozen coref target with attention aggregation and concatenated 4 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=4 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/zevus6h5
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/zv8h3lr5
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/fvmoro2p

- metric values per seed

  |     | ('val/f1',) | ('val/loss',) | ('train/f1',) | ('train/loss',) | ('train/loss_step',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                    |
  | --: | ----------: | ------------: | ------------: | --------------: | -------------------: | --------------------: | :----------------------------------------------------------------------------------------------------- |
  |   0 |    0.734546 |         307.5 |      0.944374 |         15.7083 |               19.792 |               15.7083 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_10-46-28 |
  |   1 |    0.724475 |       268.457 |      0.921956 |         23.6059 |              8.88583 |               23.6059 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_12-37-41 |
  |   2 |    0.736661 |       306.072 |       0.94697 |         15.1694 |                    0 |               15.1694 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_13-40-48 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.933165 | 0.944374 | 0.945672 |     3 |  0.94697 | 0.937766 | 0.921956 |  0.0137537 |
  | train/loss       |  15.4388 |  15.7083 |  19.6571 |     3 |  23.6059 |  18.1612 |  15.1694 |    4.72297 |
  | train/loss_epoch |  15.4388 |  15.7083 |  19.6571 |     3 |  23.6059 |  18.1612 |  15.1694 |    4.72297 |
  | train/loss_step  |  4.44291 |  8.88583 |  14.3389 |     3 |   19.792 |  9.55927 |        0 |    9.91316 |
  | val/f1           | 0.729511 | 0.734546 | 0.735603 |     3 | 0.736661 | 0.731894 | 0.724475 | 0.00651149 |
  | val/loss         |  287.265 |  306.072 |  306.786 |     3 |    307.5 |   294.01 |  268.457 |    22.1408 |

### Coreference - frozen coref target with attention aggregation and concatenated 6 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=6 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/gopk1yff
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/a22aonoa
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/ott4g0hp

- metric values per seed

  |     | ('val/loss',) | ('train/loss_step',) | ('train/f1',) | ('train/loss_epoch',) | ('model_save_dir',)                                                                                    | ('train/loss',) | ('val/f1',) |
  | --: | ------------: | -------------------: | ------------: | --------------------: | :----------------------------------------------------------------------------------------------------- | --------------: | ----------: |
  |   0 |         309.3 |              90.6762 |      0.917137 |               30.6082 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_15-38-56 |         30.6082 |    0.724304 |
  |   1 |       408.525 |                    0 |      0.949362 |               18.0845 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_16-41-47 |         18.0845 |    0.736425 |
  |   2 |       288.774 |              47.4923 |       0.91255 |               31.5345 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_19-02-17 |         31.5345 |    0.720141 |

- aggregated values:

  |                  |      25% |      50% |      75% | count |      max |     mean |      min |        std |
  | :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | ---------: |
  | train/f1         | 0.914844 | 0.917137 |  0.93325 |     3 | 0.949362 |  0.92635 |  0.91255 |  0.0200604 |
  | train/loss       |  24.3464 |  30.6082 |  31.0714 |     3 |  31.5345 |  26.7424 |  18.0845 |    7.51225 |
  | train/loss_epoch |  24.3464 |  30.6082 |  31.0714 |     3 |  31.5345 |  26.7424 |  18.0845 |    7.51225 |
  | train/loss_step  |  23.7462 |  47.4923 |  69.0843 |     3 |  90.6762 |  46.0562 |        0 |    45.3552 |
  | val/f1           | 0.722223 | 0.724304 | 0.730365 |     3 | 0.736425 | 0.726957 | 0.720141 | 0.00846008 |
  | val/loss         |  299.037 |    309.3 |  358.913 |     3 |  408.525 |  335.533 |  288.774 |    64.0404 |

### Coreference - frozen coref target with attention aggregation and concatenated 12 last layers

- command:

  ```bash
  python src/train.py \
  experiment=conll2012_coref_hoi_multimodel_base \
  +model.pretrained_models={bert-base-cased-coref-hoi:models/pretrained/bert-base-cased-coref-hoi} \
  +model.freeze_models=[bert-base-cased-coref-hoi] \
  +model.aggregate.type=attention \
  +model.aggregate.reuse_target_query_as_key=False \
  +model.aggregate.mode_query=constant \
  +model.aggregate.mode_keys=token \
  +model.aggregate.use_outputs_from_last_n_layers=12 \
  model.task_learning_rate=1e-4 \
  trainer=gpu \
  seed=1,2,3 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  name=attn_concat/single_models \
  -m
  ```

- wandb runs for different seeds:

  - seed1: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/roy7ug21
  - seed2: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/2hindzz0
  - seed3: https://wandb.ai/tanikina/attn_concat-single_models-training/runs/6imhqyeq

- metric values per seed

  |     | ('train/loss_epoch',) | ('train/f1',) | ('train/loss_step',) | ('model_save_dir',)                                                                                    | ('val/loss',) | ('train/loss',) | ('val/f1',) |
  | --: | --------------------: | ------------: | -------------------: | :----------------------------------------------------------------------------------------------------- | ------------: | --------------: | ----------: |
  |   0 |               41.1918 |      0.932674 |           0.00088644 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_19-58-01 |       557.203 |         41.1918 |    0.725797 |
  |   1 |               41.8772 |      0.927628 |          0.000135422 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_21-59-09 |       525.197 |         41.8772 |    0.726808 |
  |   2 |               46.0194 |      0.918694 |              29.0896 | /netscratch/anikina/multi-task-knowledge-transfer/models/attn_concat/single_models/2024-01-13_23-48-39 |       476.066 |         46.0194 |    0.725051 |

- aggregated values:

  |                  |         25% |        50% |      75% | count |      max |     mean |         min |         std |
  | :--------------- | ----------: | ---------: | -------: | ----: | -------: | -------: | ----------: | ----------: |
  | train/f1         |    0.923161 |   0.927628 | 0.930151 |     3 | 0.932674 | 0.926332 |    0.918694 |  0.00707956 |
  | train/loss       |     41.5345 |    41.8772 |  43.9483 |     3 |  46.0194 |  43.0295 |     41.1918 |     2.61196 |
  | train/loss_epoch |     41.5345 |    41.8772 |  43.9483 |     3 |  46.0194 |  43.0295 |     41.1918 |     2.61196 |
  | train/loss_step  | 0.000510931 | 0.00088644 |  14.5453 |     3 |  29.0896 |  9.69689 | 0.000135422 |     16.7946 |
  | val/f1           |    0.725424 |   0.725797 | 0.726303 |     3 | 0.726808 | 0.725885 |    0.725051 | 0.000881522 |
  | val/loss         |     500.631 |    525.197 |    541.2 |     3 |  557.203 |  519.488 |     476.066 |     40.8686 |
