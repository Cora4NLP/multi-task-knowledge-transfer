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

**Idea:** test whether removing key, query and value projections hurts the performance (for attention-based aggregation). We also test different learning rate values for BERT.

**Findings:** when attention uses query, key and value projections lower learning rate (3e-6) leads to better performance (0.7386). For the setting when we do not have any projections 3e-4 and 3e-5 result in the best val/f1 score (0.7395), however the difference is very small compared to other learning rates. Changing BERT learning rate does not seem to have any effect when we do not use any projections. When only the query projection is available the best learning rate is 1e-4 with 0.7401 val/f1 score. Different learning rates do not have much impact on this setting. Finally, if we remove the query projection but leave the key and value projections we get better results with the lower learning rates (e.g., 0.7406 val/f1 with lr 1e-5). This shows that those settings that use more projections need a lower BERT learning rate than the ones that do not use any or only a single one (for the query). However, the final difference in f1 scores is very small and we can achieve almost the same performance with or without projections.

**Setting:** frozen-target + frozen-MRPC with Q, K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    3e-3 |      0.7002 |       1513.25 |
|    1e-3 |      0.7079 |       82.5408 |
|    3e-4 |      0.7236 |       138.724 |
|    1e-4 |      0.7294 |       102.149 |
|    5e-5 |      0.7332 |       103.068 |
|    3e-5 |      0.7362 |       104.157 |
|    1e-5 |      0.7372 |       89.4812 |
|    3e-6 |  **0.7386** |       92.5998 |
|    1e-6 |      0.7335 |       78.4313 |

log entry: [frozen-MRPC + frozen-BERT (Q, K, V projections)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-with-query-key-and-value-projections)

**Setting:** frozen-target + frozen-MRPC without Q, K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    3e-3 |      0.7370 |       133.257 |
|    1e-3 |      0.7346 |       119.724 |
|    3e-4 |  **0.7395** |       140.255 |
|    1e-4 |      0.7358 |       121.089 |
|    5e-5 |      0.7376 |       124.365 |
|    3e-5 |  **0.7395** |       131.436 |
|    1e-5 |      0.7372 |       121.953 |
|    3e-6 |      0.7388 |       129.290 |
|    1e-6 |      0.7370 |       126.127 |

log entry: [frozen-MRPC + frozen-BERT (no projections)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-without-query-key-and-value-projections)

**Setting:** frozen-target + frozen-MRPC with Q projection but w/o K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    3e-3 |      0.7384 |       136.177 |
|    1e-3 |      0.7382 |       135.014 |
|    3e-4 |      0.7384 |       135.213 |
|    1e-4 |  **0.7401** |       134.469 |
|    5e-5 |      0.7387 |       126.815 |
|    3e-5 |      0.7398 |       128.621 |
|    1e-5 |      0.7357 |       108.715 |
|    3e-6 |      0.7397 |       126.065 |
|    1e-6 |      0.7356 |       113.756 |

log entry: [frozen-MRPC + frozen-BERT (Q projection only)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-with-query-projection-but-wo-key-and-value-projections)

**Setting:** frozen-target + frozen-MRPC without Q projection but with K and V projections

| bert lr | mean val/f1 | mean val/loss |
| ------: | ----------: | ------------: |
|    3e-3 |      0.6951 |       569.259 |
|    1e-3 |      0.7067 |       90.8321 |
|    3e-4 |      0.7229 |       146.797 |
|    1e-4 |      0.7305 |       116.805 |
|    5e-5 |      0.7339 |       106.148 |
|    3e-5 |      0.7349 |       100.096 |
|    1e-5 |  **0.7406** |       110.213 |
|    3e-6 |      0.7388 |       94.2387 |
|    1e-6 |      0.7317 |       85.3664 |

log entry: [frozen-MRPC + frozen-BERT (K, V projections only)](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#bert-lr-tuning-for-coref-frozen-pretrained-coref--frozen-mrpc-attention-aggregation-without-query-projection-but-with-key-and-value-projections)

## Probing experiments

**Idea:** check whether different pre-trained models work well on the coreference task. Each pre-trained model is frozen and we use mean for embeddings aggregation to avoid introducing additional parameters into the model.

**Findings:** as expected, frozen-target performs the best (0.7375 val/f1). Surprisingly, frozen-NER performs very poorly (only 0.3563 val/f1). Standard BERT (w/o any fine-tuning) gives the best scores among the models that were not fine-tuned on the coreference task (0.6495). Another interesting observation is that having frozen BERT 3 times gives a noticeable drop in performance (almost -3%). This means that we probably introduce some noise when combining embeddings with mean aggregation.

| setting             | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                    |
| :------------------ | :----- | :------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-target       | 0.7375 | 117.221  | 0.00291    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation)     |
| frozen-MRPC         | 0.6116 | 242.417  | 0.02836    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation)       |
| frozen-NER          | 0.3563 | 512.712  | 0.02122    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation)        |
| frozen-RE           | 0.5227 | 691.528  | 0.02390    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation)         |
| frozen-SQUAD        | 0.5079 | 577.224  | 0.03011    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-qa-model-with-mean-aggregation)         |
| frozen-BERT         | 0.6495 | 295.832  | 0.00977    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation)       |
| frozen-NER-RE-SQUAD | 0.5793 | 269.619  | 0.03082    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-re-qa-models-with-mean-aggregation) |
| frozen-BERT-3x      | 0.6195 | 320.306  | 0.00708    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-3x-models-with-mean-aggregation)   |

## Probing experiments with truncated models

**Idea:** do the same as in the [probing experiments above](#probing-experiments) but with the truncated models (6-11 layers).

**Findings:** for the frozen-target setup we have a predictable behaviour with the f1 scores going down from 0.7375 to 0.6723 when we decrease the number of available layers (from 12 to 6). Interestingly, MRPC shows better scores when truncated to layer 8 or 9 (0.6626 val/f1), NER performs very poorly when only the output of the final layer is considered (0.3563 val/f1) but the score increases to 0.6176 when we use the output of the 6th layer. Similarly to MRPC, RE model achieves best performance with the truncation to the 9th layer (0.6240 val/f1). SQUAD shows the same pattern with the difference of +13.98 f1 points when we truncate the model to the first 8 layers. Finally, frozen-BERT has slightly better performance when we truncate the model to the 9th or 10th layer but the difference is smaller than e.g., for SQUAD or NER. The best score is 0.6640 with frozen-BERT<sub>10</sub>.

### frozen-target

| setting                    | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                                       |
| :------------------------- | :----- | :------- | :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| frozen-target<sub>12</sub> | 0.7375 | 117.221  | 0.00291    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation)                        |
| frozen-target<sub>11</sub> | 0.7303 | 1137.19  | 0.00482    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation-truncated-to-11-layers) |
| frozen-target<sub>10</sub> | 0.7251 | 1270.77  | 0.0058     | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation-truncated-to-10-layers) |
| frozen-target<sub>9</sub>  | 0.7286 | 1638.51  | 0.0034     | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation-truncated-to-9-layers)  |
| frozen-target<sub>8</sub>  | 0.7147 | 1432.39  | 0.00127    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation-truncated-to-8-layers)  |
| frozen-target<sub>7</sub>  | 0.6953 | 1209.98  | 0.0024     | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation-truncated-to-7-layers)  |
| frozen-target<sub>6</sub>  | 0.6723 | 1071.08  | 0.00209    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-target-model-with-mean-aggregation-truncated-to-6-layers)  |

### frozen-MRPC

| setting                  | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                                     |
| :----------------------- | :----- | :------- | :--------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-MRPC<sub>12</sub> | 0.6116 | 242.417  | 0.02836    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation)                        |
| frozen-MRPC<sub>11</sub> | 0.6375 | 797.579  | 0.00082    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation-truncated-to-11-layers) |
| frozen-MRPC<sub>10</sub> | 0.6524 | 824.06   | 0.01113    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation-truncated-to-10-layers) |
| frozen-MRPC<sub>9</sub>  | 0.6626 | 887.15   | 0.00610    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation-truncated-to-9-layers)  |
| frozen-MRPC<sub>8</sub>  | 0.6613 | 862.599  | 0.00408    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation-truncated-to-8-layers)  |
| frozen-MRPC<sub>7</sub>  | 0.6550 | 886.209  | 0.01268    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation-truncated-to-7-layers)  |
| frozen-MRPC<sub>6</sub>  | 0.6280 | 873.283  | 0.00830    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-mrpc-model-with-mean-aggregation-truncated-to-6-layers)  |

### frozen-NER

| setting                 | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                                    |
| :---------------------- | :----- | :------- | :--------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-NER<sub>12</sub> | 0.3563 | 512.712  | 0.02122    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation)                        |
| frozen-NER<sub>11</sub> | 0.5315 | 930.88   | 0.01473    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation-truncated-to-11-layers) |
| frozen-NER<sub>10</sub> | 0.5575 | 1043.45  | 0.01816    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation-truncated-to-10-layers) |
| frozen-NER<sub>9</sub>  | 0.5623 | 1089.59  | 0.02427    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation-truncated-to-9-layers)  |
| frozen-NER<sub>8</sub>  | 0.5983 | 927.081  | 0.01198    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation-truncated-to-8-layers)  |
| frozen-NER<sub>7</sub>  | 0.5821 | 919.705  | 0.03259    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation-truncated-to-7-layers)  |
| frozen-NER<sub>6</sub>  | 0.6176 | 925.745  | 0.01533    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-ner-model-with-mean-aggregation-truncated-to-6-layers)  |

### frozen-RE

| setting                | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                                   |
| :--------------------- | :----- | :------- | :--------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-RE<sub>12</sub> | 0.5227 | 691.528  | 0.02390    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation)                        |
| frozen-RE<sub>11</sub> | 0.5869 | 997.296  | 0.00706    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation-truncated-to-11-layers) |
| frozen-RE<sub>10</sub> | 0.5951 | 1053.81  | 0.00167    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation-truncated-to-10-layers) |
| frozen-RE<sub>9</sub>  | 0.6240 | 986.183  | 0.01412    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation-truncated-to-9-layers)  |
| frozen-RE<sub>8</sub>  | 0.6132 | 984.823  | 0.00242    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation-truncated-to-8-layers)  |
| frozen-RE<sub>7</sub>  | 0.6148 | 1001.3   | 0.00371    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation-truncated-to-7-layers)  |
| frozen-RE<sub>6</sub>  | 0.6065 | 996.46   | 0.01435    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-re-model-with-mean-aggregation-truncated-to-6-layers)  |

### frozen-SQUAD

| setting                   | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                                      |
| :------------------------ | :----- | :------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-SQUAD<sub>12</sub> | 0.5079 | 577.224  | 0.03011    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-qa-model-with-mean-aggregation)                           |
| frozen-SQUAD<sub>11</sub> | 0.6282 | 931.756  | 0.02304    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-squad-model-with-mean-aggregation-truncated-to-11-layers) |
| frozen-SQUAD<sub>10</sub> | 0.6419 | 935.4    | 0.01579    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-squad-model-with-mean-aggregation-truncated-to-10-layers) |
| frozen-SQUAD<sub>9</sub>  | 0.6431 | 937.324  | 0.00303    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-squad-model-with-mean-aggregation-truncated-to-9-layers)  |
| frozen-SQUAD<sub>8</sub>  | 0.6477 | 911.96   | 0.00653    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-squad-model-with-mean-aggregation-truncated-to-8-layers)  |
| frozen-SQUAD<sub>7</sub>  | 0.6412 | 935.452  | 0.00729    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-squad-model-with-mean-aggregation-truncated-to-7-layers)  |
| frozen-SQUAD<sub>6</sub>  | 0.6379 | 920.508  | 0.00702    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-squad-model-with-mean-aggregation-truncated-to-6-layers)  |

### frozen-BERT

| setting                  | val/f1 | val/loss | val/f1/std | log entry                                                                                                                                                                     |
| :----------------------- | :----- | :------- | :--------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frozen-BERT<sub>12</sub> | 0.6495 | 295.832  | 0.00977    | [2023-11-27](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation)                        |
| frozen-BERT<sub>11</sub> | 0.6353 | 791.918  | 0.00333    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation-truncated-to-11-layers) |
| frozen-BERT<sub>10</sub> | 0.6640 | 865.084  | 0.01664    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation-truncated-to-10-layers) |
| frozen-BERT<sub>9</sub>  | 0.6627 | 864.179  | 0.0079     | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation-truncated-to-9-layers)  |
| frozen-BERT<sub>8</sub>  | 0.6496 | 803.509  | 0.01842    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation-truncated-to-8-layers)  |
| frozen-BERT<sub>7</sub>  | 0.6499 | 842.642  | 0.01655    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation-truncated-to-7-layers)  |
| frozen-BERT<sub>6</sub>  | 0.6233 | 807.78   | 0.00955    | [2023-12-04](https://github.com/Cora4NLP/multi-task-knowledge-transfer/blob/main/log.md#coreference-probing---frozen-bert-model-with-mean-aggregation-truncated-to-6-layers)  |

## Comparing the val/f1 scores and loss of the truncated models

[W&B project](https://wandb.ai/tanikina/probing-coref-truncated-models-training)

### 11 layers

<img src="images/val_f1_truncation-11.png" width=70% height=70% />
<img src="images/val_loss_truncation-11.png" width=70% height=70% />

### 10 layers

<img src="images/val_f1_truncation-10.png" width=70% height=70% />
<img src="images/val_loss_truncation-10.png" width=70% height=70% />

### 9 layers

<img src="images/val_f1_truncation-9.png" width=70% height=70% />
<img src="images/val_loss_truncation-9.png" width=70% height=70% />

### 8 layers

<img src="images/val_f1_truncation-8.png" width=70% height=70% />
<img src="images/val_loss_truncation-8.png" width=70% height=70% />

### 7 layers

<img src="images/val_f1_truncation-7.png" width=70% height=70% />
<img src="images/val_loss_truncation-7.png" width=70% height=70% />

### 6 layers

<img src="images/val_f1_truncation-6.png" width=70% height=70% />
<img src="images/val_loss_truncation-6.png" width=70% height=70% />

## TODOs & Ideas:

- double-check that tuning models is better than freezing them, also it would be interesting to check how much the weights change compared to the original models, could we quantify this?
- train adapter modules for each task and compare the model combination results with adapter fusion
- experiment with additional projections (linear layers) to make sure that the representations from different models are compatible and close to each other in the embeddings space
- adapter distillation idea (requires more work but seems promising)
