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

## 2023-10-27

### Probing Correlation of pretrained RE model final layer embeddings with other tasks

#### NER

- command:

  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=conll2012_ner-multimodel_base \
  model.learning_rate=1e-4 \
  seed=1031,1097,1153,1223,1579 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  \"tags=[\'dataset=conll2012\',\'model=multi_model_token_classification\',\'probing_model=bert-base-cased-re-tacred\']\" \
  "name=probing/bert-base-cased-re-tacred" \
  -m
  ```

- wandb runs:

  - Runs with NER- prefix and numbers -1,-4, -5, -6, -7 at https://wandb.ai/leonhardhennig/probing-bert-base-cased-re-tacred-training

- results:

|                  |      25% |      50% |      75% | count |      max |     mean |       min |         std |
| :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | --------: | ----------: |
| train/f1         |  0.58033 |  0.58063 | 0.580819 |     5 |  0.58101 | 0.580525 |  0.579834 | 0.000460613 |
| train/loss       | 0.229972 |  0.22998 | 0.230024 |     5 | 0.230078 | 0.229897 |  0.229433 | 0.000262888 |
| train/loss_epoch | 0.229972 |  0.22998 | 0.230024 |     5 | 0.230078 | 0.229897 |  0.229433 | 0.000262888 |
| train/loss_step  | 0.110126 | 0.152804 | 0.270792 |     5 | 0.393352 | 0.198475 | 0.0652988 |    0.133073 |
| val/f1           | 0.676782 | 0.678543 | 0.679046 |     5 | 0.681175 | 0.678199 |  0.675447 |   0.0021953 |
| val/loss         | 0.192253 | 0.192285 | 0.193033 |     5 | 0.195406 | 0.192973 |  0.191889 |  0.00142214 |

#### Coref

- command:

  ```bash
  python src/train.py \
  trainer=gpu \
  experiment=conll2012_coref_hoi_multimodel_base \
  model.task_learning_rate=1e-4 \
  seed=1031,1097,1153,1223,1579 \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf \
  +model.freeze_models=[bert-base-cased-re-tacred] \
  \"tags=[\'dataset=conll2012\',\'model=multi_model_coref_hoi\',\'probing_model=bert-base-cased-re-tacred\']\" \
  "name=probing/bert-base-cased-re-tacred" \
  -m
  ```

- wandb runs:

  - Runs with COREF- prefix and numbers -2,-8, -10, -11, -13 at https://wandb.ai/leonhardhennig/probing-bert-base-cased-re-tacred-training

- results:

|                  |      25% |      50% |      75% | count |      max |     mean |      min |       std |
| :--------------- | -------: | -------: | -------: | ----: | -------: | -------: | -------: | --------: |
| train/f1         | 0.430001 |  0.44413 | 0.448289 |     5 | 0.471093 | 0.443547 | 0.424219 |  0.018294 |
| train/loss       |  779.455 |  790.471 |  832.769 |     5 |  859.321 |  790.519 |  690.582 |   64.4757 |
| train/loss_epoch |  779.455 |  790.471 |  832.769 |     5 |  859.321 |  790.519 |  690.582 |   64.4757 |
| train/loss_step  |  147.987 |  350.733 |  1390.74 |     5 |  1465.24 |  697.139 |   130.99 |   673.269 |
| val/f1           | 0.498223 | 0.507702 | 0.514976 |     5 | 0.533595 | 0.505491 | 0.472961 | 0.0223366 |
| val/loss         |  672.692 |  719.793 |  725.444 |     5 |  758.595 |   709.78 |  672.376 |   37.0915 |

#### QA

- command:

  ```bash
  python src/train.py  \
  trainer=gpu  \
  experiment=squadv2-multimodel_base \
  +model.learning_rate=1e-4  \
  seed=1031,1097,1153,1223,1579  \
  +hydra.callbacks.save_job_return.integrate_multirun_result=true  \
  +model.pretrained_models.bert-base-cased-re-tacred=/ds/text/cora4nlp/models/bert-base-cased-re-tacred-20230919-hf  \
  +model.freeze_models=[bert-base-cased-re-tacred]  \
  \"tags=[\'dataset=squadv2\',\'task=extractive_question_answering\',\'probing_model=bert-base-cased-re-tacred\']\"  \
  "name=probing/bert-base-cased-re-tacred"  \
  -m
  ```

- wandb runs:

  - Runs with QA- prefix and  numbers -3,-9, -12, -14, -15 at https://wandb.ai/leonhardhennig/probing-bert-base-cased-re-tacred-training

- results:
