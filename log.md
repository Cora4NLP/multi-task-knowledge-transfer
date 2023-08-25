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
  - command: `python src/train.py experiment=conll2012_coref_hoi trainer=gpu seed=1,2,3 --multirun`
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
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu model.aggregate=sum seed=1 +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes]`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8gurkh4p
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_12-33-45`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step |  val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | ------: | -------: |
    | run1 |  0.45232 |          4449.93 |               56039 | 0.46102 |  4397.89 |

### Coreference Resolution: BERT, Re-TACRED and NER (all trainable, sum aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (trainable, sum)
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu model.aggregate=sum seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/pwymsi30
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_12-31-28`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.50 |         1855.579 |               56039 |  0.552 | 1533.741 |

### Coreference Resolution: BERT, Re-TACRED and NER (Re-TACRED and NER frozen, mean aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (frozen, mean)
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu model.aggregate=mean seed=1 +model.freeze_models=[bert-base-cased-re-tacred,bert-base-cased-ner-ontonotes]`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/8a6cvg1c
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_10-22-22`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.47 |          523.931 |               56039 |  0.396 |  698.734 |

### Coreference Resolution: BERT, Re-TACRED and NER (all trainable, mean aggregation)

- running a multi-model with bert-base-cased, NER and Re-TACRED (trainable, mean)
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dqb2bpel
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-01_10-43-45`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |    0.521 |          207.861 |               56039 |  0.555 |  203.024 |

### Coreference Resolution: BERT and NER (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and NER (trainable, mean)
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu seed=1`
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
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu seed=1 +model.freeze_models=[bert-base-cased-ner-ontonotes]` TODO
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
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu seed=1`
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
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu seed=1 +model.freeze_models=[bert-base-cased-re-tacred]`
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
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_triple_bert trainer=gpu seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/44daf336
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_14-02-23`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.71 |          106.787 |               56039 |  0.632 |  213.828 |

### Coreference Resolution: BERT, Re-TACRED, NER and SQUAD (all trainable, mean aggregation)

- running a multi-model with bert-base-cased, Re-TACRED, NER and SQUAD
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel trainer=gpu model.aggregate=mean seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/kk817tye
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_13-29-52`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.56 |          199.767 |               56039 |  0.613 |  205.018 |

### Coreference Resolution: BERT and SQUAD (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-qa-squad2
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_bert_qa trainer=gpu seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/81089nj3
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_16-08-08`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.57 |          479.938 |               56039 |  0.607 |   496.05 |

### Coreference Resolution: BERT and SQUAD (SQUAD frozen, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-qa-squad2
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_bert_qa trainer=gpu seed=1 +model.freeze_models=[bert-base-cased-qa-squad2]`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/n561w8zl
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-21_16-03-48`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.57 |          497.308 |               56039 |  0.558 |   608.34 |

### Coreference Resolution: BERT and MRPC (all trainable, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-mrpc
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_mrpc trainer=gpu seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/dv7stxmf
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_09-40-18`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.67 |          116.573 |               56039 |  0.638 |   169.64 |

### Coreference Resolution: BERT and MRPC (MRPC frozen, mean aggregation)

- running a multi-model with bert-base-cased and bert-base-cased-mrpc (frozen)
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_mrpc trainer=gpu seed=1 +model.freeze_models=[bert-base-cased-mrpc]`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/tq346u1c
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_09-44-17`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.64 |          255.440 |               56039 |  0.630 |   342.60 |

### Coreference Resolution: BERT pre-trained on coreference (all trainable, mean aggregation)

- running a multi-model with 3 pre-trained coreference BERT models, no bert-base-cased (commented out in the configuration file)
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_triple_coref trainer=gpu seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/ynkldn9e
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_09-50-48`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.73 |           53.306 |               56039 |  0.655 |   105.85 |

### Coreference Resolution: BERT pre-trained on coreference + bert-base-cased (all trainable, mean aggregation)

- running a multi-model with 3 pre-trained coreference BERT models + bert-base-cased
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_triple_coref trainer=gpu seed=1`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/3dwloytn
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_18-30-25`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.73 |           58.478 |               56039 |  0.651 |   121.52 |

### Coreference Resolution: BERT pre-trained on coreference + bert-base-cased (3 models are frozen, mean aggregation)

- running a multi-model with 3 pre-trained coreference BERT models (frozen) + bert-base-cased
  - command: `python src/train.py experiment=conll2012_coref_hoi_multimodel_triple_coref trainer=gpu seed=1 +model.freeze_models=[bert-base-cased-coref1,bert-base-cased-coref2,bert-base-cased-coref3]`
  - wandb (weights & biases) run: https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/zx3ytipi
  - artefacts
    - model location: `/netscratch/anikina/multi-task-knowledge-transfer/models/conll2012/multi_model_coref_hoi/2023-08-22_18-31-40`
  - metric values:
    |      | train/f1 | train/loss_epoch | trainer/global_step | val/f1 | val/loss |
    | ---: | -------: | ---------------: | ------------------: | -----: | -------: |
    | run1 |     0.73 |           88.203 |               56039 |  0.636 |   202.80 |
