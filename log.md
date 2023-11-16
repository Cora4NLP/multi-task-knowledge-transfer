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
