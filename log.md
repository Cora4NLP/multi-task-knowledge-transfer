# Experimentation Log

This file is meant to log the development and experimentation process of this project. Please log relevant experiments including the
applied commands, a short motivation and a summary of any relevant outcome, or follow-up ideas, if possible.

Usage: If you want to add content, please create a new section with the current date as heading (if not yet available), and maybe a
subsection with your topic (e.g. NER, CoRef, General, Aggregation, ...), e.g.

```markdown
## 2023-07-27

### Relation Extraction
 - short experiment to verify that the code for the simple multi-model variant is working
    - preparation: implemented the multi-model variant of the text classification model
    - command: `python src/train.py experiment=tacred +trainer.fast_dev_run=true`
    - wandb (weights & biases) run: https://wandb.ai/arne/pie-example-scidtb/runs/2rsl4z9p (this is just an example!)
    - outcome: the code works
```

IMPORTANT: Execute `pre-commit run -a` before committing to ensure that the markdown is formatted correctly.
