# Coreference Experiments Results

This file replicates the main outcomes from `log.md` and provides the summaries and interpretations of the experiments. It includes all experiments relevant for coreference resolution.

## Experiments with the learning rate

The best result with the maximum val/f1 was achieved with the task learning rate 1e-4. All further experiments use this learning rate (unless otherwise specified).

|       lr |      val/f1 | val/loss |
| -------: | ----------: | -------: |
|     2e-3 |     0.72661 |  389.376 |
|     1e-3 |     0.73031 |  233.430 |
|     2e-4 |       0.735 |  119.975 |
| **1e-4** | **0.73653** |  99.1169 |
|     2e-5 |     0.73101 |  85.9004 |

log entry: [frozen pre-trained target-model + frozen bert (coref learning rate optimization)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-bert-learning-rate-optimization)

## Experiments with pre-trained target + another frozen pre-trained model (RE, NER, QA)

**Idea:** check whether having another pre-trained task model together with the frozen target model brings any benefits.

**Findings:** there is no big difference in results, although the combination with SQUAD seems slightly better than the rest (0.739). Interestingly, [(Vu et al., 2022)](https://aclanthology.org/2022.acl-long.346/) also showed that coreference task benefits from the soft prompt transfer when the prompt is pre-trained on the QA task (they showed it for ReCoRD → WSC).

**TODOs:** re-run the experiment for *frozen-target-only* with the optimized learning rate. The [old experiment](https://wandb.ai/tanikina/conll2012-multi_model_coref_hoi-training/runs/7uyjay9c) has best val/f1 0.7367 but it was trained with the default (non-optimized) learning rate 2e-4. Still, combinations with NER, SQUAD and MRPC seem to outperform it slightly.

| setting                      | val/f1   | val/loss | val/f1/std | log entry                                                                                                                                                                                |
| :--------------------------- | :------- | :------- | :--------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-target + frozen-NER   | 0.73828  | 101.815  | 0.0014     | [2023-10-23](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-ner-model-learning-rate-1e-4)   |
| frozen-target + frozen-RE    | 0.73660  | 90.403   | 0.00186    | [2023-10-23](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-re-model-learning-rate-1e-4)    |
| frozen-target + frozen-SQUAD | 0.738545 | 100.612  | 0.00121    | [2023-10-23](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-squad-model-learning-rate-1e-4) |
| frozen-target + frozen-MRPC  | 0.738308 | 99.4617  | 0.00106    | [2023-10-23](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-learning-rate-1e-4)  |

## Experiments with layer truncation with frozen target + frozen MRPC where we truncate only the MRPC model

**Idea:** the last layers of BERT may be over-specialized on the target task and thus not very useful for the target model, we check whether truncating the model to different sizes has any effect.

**Findings:** the results seem to be the same for different number of truncated layers. MRPC truncated to 11 layers shows marginally better performance than the full (non-truncated) model: 0.739 vs 0.738 val/f1. Interestingly, truncating MRPC to only 2 layers brings approximately the same result as when we have the full model. This probably means that having a model pre-trained on the target task makes other models obsolete.

| setting                                               | val/f1   | val/loss | val/f1/std | log entry                                                                                                                                                                                        |
| :---------------------------------------------------- | :------- | :------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>2</sub>  | 0.738888 | 106.188  | 0.00088    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-truncated-to-2-layers)  |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>4</sub>  | 0.738706 | 96.5667  | 0.00038    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-truncated-to-4-layers)  |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>6</sub>  | 0.738047 | 101.016  | 0.00463    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-truncated-to-6-layers)  |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>8</sub>  | 0.737117 | 95.9866  | 0.00230    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-truncated-to-8-layers)  |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>10</sub> | 0.737605 | 100.998  | 0.00216    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-truncated-to-10-layers) |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>11</sub> | 0.738912 | 105.555  | 0.00255    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-truncated-to-11-layers) |
| frozen-target<sub>12</sub> + frozen-MRPC<sub>12</sub> | 0.738308 | 99.4617  | 0.00106    | [2023-10-23](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-learning-rate-1e-4)          |

## Experiments with layer truncation with frozen target + frozen MRPC where we truncate both the target and the MRPC model

**Idea:** perhaps combining the final layer representation from the target model with the outputs from earlier layers of the other models is not an optimal strategy. Here we test whether combining representations from the same layer makes any difference.

**Findings:** combining the representations at the same layer does not bring any improvement. In fact, having both models truncated to 6 or 8 layers results in worse performance (this is expected because we reduce the capacity of the model with truncation). However, truncating the models to 11 or 10 layers does not seem to have a large impact, the scores are quite similar to the ones obtained with the full MRPC model (0.738).

**TODOs:** Maybe it's also interesting to check whether we could combine the representations at every layer instead of only the final one.

| setting                                               | val/f1   | val/loss | val/f1/std | log entry                                                                                                                                                                                                  |
| :---------------------------------------------------- | :------- | :------- | :--------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-target<sub>6</sub> + frozen-MRPC<sub>6</sub>   | 0.684451 | 142.663  | 0.00535    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-and-coref-truncated-to-6-layers)  |
| frozen-target<sub>8</sub> + frozen-MRPC<sub>8</sub>   | 0.716848 | 201.652  | 0.00233    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-and-coref-truncated-to-8-layers)  |
| frozen-target<sub>10</sub> + frozen-MRPC<sub>10</sub> | 0.734252 | 255.228  | 0.00241    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-and-coref-truncated-to-10-layers) |
| frozen-target<sub>11</sub> + frozen-MRPC<sub>11</sub> | 0.733886 | 212.317  | 0.00305    | [2023-11-01](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-mrpc-and-coref-truncated-to-11-layers) |

## Experiments with frozen BERT + frozen other model

**Idea:** in the previous experiments we used the frozen target model and here we test whether having a simple *bert-base-cased* model can make other pre-trained models more useful for the target task. Otherwise, we may simply overfit to the pre-trained target and discard the inputs from other models.

**Findings:** compared to a setting with only frozen BERT we see an improvement for the combinations with other models. Especially, MRPC seems to be beneficial (0.684 with MRPC vs 0.662 with BERT-only). Combining two BERT models improves the results compared to a single model (0.672 vs. 0.662) but it still slightly underperforms combinations with the other pre-trained models. Also, having four frozen BERT models results in a worse val/f1 score than BERT with frozen NER, RE and SQUAD models (0.672 vs 0.683). However, the difference is very small, around 1%.

| setting                           | val/f1   | val/loss | val/f1/std | log entry                                                                                                                                                  |
| :-------------------------------- | :------- | :------- | :--------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-BERT-only                  | 0.661615 | 79.2392  | 0.00220    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---only-frozen-bert)                         |
| frozen-BERT + frozen-BERT         | 0.67239  | 75.126   | 0.00368    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-bert-model)           |
| frozen-BERT + frozen MRPC         | 0.68387  | 70.611   | 0.00256    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-mrpc-model)           |
| frozen-BERT + frozen-NER          | 0.681783 | 76.8861  | 0.00443    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-ner-model)            |
| frozen-BERT + frozen-RE           | 0.674879 | 76.5669  | 0.00042    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-re-model)             |
| frozen-BERT + frozen-SQUAD        | 0.679109 | 70.7663  | 0.00705    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-squad-model)          |
| frozen-BERT + frozen-BERT-3x      | 0.671762 | 67.3244  | 0.00309    | [2023-11-13](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-bert-3x)              |
| frozen-BERT + frozen-NER-RE-SQUAD | 0.683018 | 65.6854  | 0.00587    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-ner--re--squad-model) |

## Experiments with frozen BERT + frozen other model truncated to 10 layers

**Idea:** combine the frozen BERT output with the representations from the earlier layers of the other models. Here we check whether the results obtained in [the experiments with the frozen target](#experiments-with-layer-truncation-with-frozen-target--frozen-mrpc-where-we-truncate-only-the-mrpc-model) also hold for the frozen BERT.

**Findings:** truncating to 10 layers brings no benefit for NER but gives small improvements for RE (0.683 vs 0.675), SQUAD (0.691 vs 0.679) and MRPC (0.692 vs 0.684). The best performing setting includes frozen BERT and frozen NER-RE-SQUAD where all the other models are truncated to 10 layers. This combination achieves 0.702 val/f1 which is better than BERT-only (0.662), two frozen BERT models (0.672) and four frozen BERT models (0.692) where the target BERT model has 12 layers and three other BERT models are truncated to 10 layers.

| setting                                                                  | val/f1   | val/loss | val/f1/std | log entry                                                                                                                                                               |
| :----------------------------------------------------------------------- | :------- | :------- | :--------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-BERT + frozen-MRPC<sub>10</sub>                                   | 0.6924   | 86.3845  | 0.00227    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-mrpc-model-10-truncated)           |
| frozen-BERT + frozen-NER<sub>10</sub>                                    | 0.68122  | 75.3121  | 0.00226    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-ner-model-10-truncated)            |
| frozen-BERT + frozen-RE<sub>10</sub>                                     | 0.682553 | 86.9013  | 0.00310    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-re-model-10-truncated)             |
| frozen-BERT + frozen-SQUAD<sub>10</sub>                                  | 0.690946 | 89.4629  | 0.00706    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-squad-model-10-truncated)          |
| frozen-BERT + frozen-BERT<sub>10</sub>-3x                                | 0.692258 | 86.5103  | 0.00089    | [2023-11-13](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-bert-3x-10-truncated)              |
| frozen-BERT + frozen-NER<sub>10</sub>-RE<sub>10</sub>-SQUAD<sub>10</sub> | 0.701959 | 86.4457  | 0.00428    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-ner--re--squad-model-10-truncated) |

## Aggregation experiments

**Idea:** test different ways of aggregating the token embeddings (attention vs mean vs sum).

**Findings:** attention is the clear winner followed by mean and then sum aggregation. Interestingly, the validation loss for sum aggregation is very big (602.827 with frozen-target and 926.887 with frozen BERT) compared to the other methods (e.g., 99.4617 for frozen-target with attention and 70.611 for frozen-target with BERT).

| setting                     | aggregate | val/f1   | val/loss | val/f1/std | log entry                                                                                                                                                                                  |
| :-------------------------- | :-------- | :------- | :------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-target + frozen-MRPC | attn      | 0.738308 | 99.4617  | 0.00106    | [2023-10-23](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-learning-rate-1e-4)    |
| frozen-target + frozen-MRPC | mean      | 0.710016 | 165.478  | 0.00488    | [2023-11-16](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-with-mean-aggregation) |
| frozen-target + frozen-MRPC | sum       | 0.705211 | 602.827  | 0.00945    | [2023-11-16](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-pre-trained-target-model--frozen-mrpc-model-with-sum-aggregation)  |
| frozen-BERT + frozen-MRPC   | attn      | 0.68387  | 70.611   | 0.00256    | [2023-11-10](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-mrpc-model)                                           |
| frozen-BERT + frozen-MRPC   | mean      | 0.632592 | 245.824  | 0.02306    | [2023-11-16](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-mrpc-model-with-mean-aggregation)                     |
| frozen-BERT + frozen-MRPC   | sum       | 0.623386 | 926.887  | 0.01060    | [2023-11-16](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-resolution---frozen-bert--frozen-mrpc-model-with-sum-aggregation)                      |

## Projections ablation experiments

**Idea:** test whether removing key, query and value projections hurts the performance (attention-based aggregation). We also test different learning rate values for BERT.

**Findings:** 1e-5 is the best lr when we have all projections, 5e-5 is the best lr w/o any projections, 1e-4 works best with only query projection and 1e-5 is the best lr when we have only keys and values projections. Overall, removing the query projection leads to the best average f1 score (0.7406). However, the difference is very small and there is no clear pattern regarding the learning rate optimization.

frozen-target + frozen-MRPC with Q, K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    1e-4 |      0.7294 |       102.149 |
|    5e-5 |      0.7332 |       103.068 |
|    1e-5 |      0.7372 |       89.4812 |
|    1e-6 |      0.7335 |       78.4313 |

log entry: [frozen-MRPC + frozen-BERT (Q, K, V projections)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/coref_probing/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-with-query-key-and-value-projections)

frozen-target + frozen-MRPC without Q, K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    1e-4 |      0.7358 |       121.089 |
|    5e-5 |      0.7376 |       124.365 |
|    1e-5 |      0.7372 |       121.953 |
|    1e-6 |      0.7370 |       126.127 |

log entry: [frozen-MRPC + frozen-BERT (no projections)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/coref_probing/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-without-query-key-and-value-projections)

frozen-target + frozen-MRPC with Q projection but w/o K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    1e-4 |      0.7401 |       134.469 |
|    5e-5 |      0.7387 |       126.815 |
|    1e-5 |      0.7357 |       108.715 |
|    1e-6 |      0.7356 |       113.756 |

log entry: [frozen-MRPC + frozen-BERT (Q projection only)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/coref_probing/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-with-query-projection-but-wo-key-and-value-projections)

frozen-target + frozen-MRPC without Q projection but with K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    1e-4 |      0.7305 |       116.805 |
|    5e-5 |      0.7339 |       106.148 |
|    1e-5 |      0.7406 |       110.213 |
|    1e-6 |      0.7317 |       85.3664 |

log entry: [frozen-MRPC + frozen-BERT (K, V projections only)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/coref_probing/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-without-query-projection-but-with-key-and-value-projections)

## TODOs & Ideas:

- double-check that tuning models is better than freezing them, also it would be interesting to check how much the weights change compared to the original models, could we quantify this?
- train adapter modules for each task and compare the model combination results with adapter fusion
- experiment with additional projections (linear layers) to make sure that the representations from different models are compatible and close to each other in the embeddings space
- adapter distillation idea (requires more work but seems promising)
