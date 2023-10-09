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

### Relation Extraction - target-only model with attention

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

### Relation Extraction - frozen pre-trained target-model + bert-base-cased with attention

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

### Relation Extraction - frozen pre-trained target-model + frozen NER model with attention

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

### Relation Extraction - pre-trained target-model + frozen NER model with attention

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

### Relation Extraction - pre-trained target-model + frozen NER model with attention, LR = 5e-6

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

### NER - tunable finetuned NER target model + frozen coreference model

- running a tunable NER model with a frozen coref model and attention aggregation
  - command:
    ```bash
       python src/train.py \
       experiment=conll2012_ner-multimodel_train_target \
       trainer=gpu \
       seed=1
    ```
  - wandb (weights & biases) run:
    https://wandb.ai/dfki-nlp/conll2012-multi_model_token_classification-training/runs/dy1928uo
  - artefacts
    - model location:
      /netscratch/harbecke/multi-task-knowledge-transfer/models/conll2012/multi_model_token_classification/2023-10-05_06-59-40
  - metric values (epoch 7):
    | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss | aggregate |
    | -------: | ---------------: | ------------------: | -----: | -------: | --------: |
    |    0.996 |            0.003 |               72399 |  0.914 |    0.118 | attention |

## 2023-10-06

### Coreference Resolution

- wandb report with the val/f1 and val/loss graphs (experiments from 2023-09-28 and 2023-09-29):
  [https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/reports/Coreference-Experiments--Vmlldzo1NjAwNTMy](https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/reports/Coreference-Experiments--Vmlldzo1NjAwNTMy)
