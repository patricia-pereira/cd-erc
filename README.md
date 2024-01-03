# Context-Dependent Embedding Utterance Representations for Emotion Recognition in Conversations

## Train:

```bash
python cli.py train -f configs/{config_id}.yaml
```

## Interact:
Fun command to interact with a trained model.

```bash
python cli.py interact --experiment experiments/{experiment_id}/
```

## Testing:

```bash
python cli.py test --experiment experiments/{experiment_id}/
```

## Citation

```bibtex
@inproceedings{pereira-etal-2023-context,
    title = "Context-Dependent Embedding Utterance Representations for Emotion Recognition in Conversations",
    author = "Pereira, Patr{\'\i}cia  and
      Moniz, Helena  and
      Dias, Isabel  and
      Carvalho, Joao Paulo",
    booktitle = "Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, {\&} Social Media Analysis",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wassa-1.21",
    doi = "10.18653/v1/2023.wassa-1.21",
    pages = "228--236",
    }
```
