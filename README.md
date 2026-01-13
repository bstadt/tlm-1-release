# TLM-1: Temporal Language Model

**TLM-1** is a BERT-style transformer that explicitly models language as a temporal process. Unlike traditional language models that treat training documents as simultaneous, TLM-1 jointly learns to predict document contents and classify document dates, capturing how vocabulary and meanings evolve over time.

## Model Description

TLM-1 was trained on the **Corpus of Contemporary American English (COCA)**—750 million words spanning 1990-2019—enabling analysis of temporal trends in American English across three decades.

The model introduces 30 special time tokens (`[YEAR:1990]` through `[YEAR:2019]`) that allow conditioning generation on specific time periods and analyzing how language use shifts across years.

### Key Features

- **Temporal conditioning**: Generate or analyze text as it would appear in any year from 1990-2019
- **Semantic drift detection**: Track how word meanings change over time
- **Bayesian query formulation**: Counteract training biases using Bayes factors for rigorous temporal analysis

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("bstadt/tlm-1")
tokenizer = AutoTokenizer.from_pretrained("bstadt/tlm-1")

# Condition on a specific year
text = "[YEAR:2015] The president [MASK] announced new policies today."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### Bayesian Query Formulation

Raw fill probabilities from the model can be biased by training data frequencies. To extract genuine temporal signal, use Bayes factors:

```
P(fill | year) / P(template | year)
```

This approach replaces naive likelihood estimates with normalized probabilities that surface true temporal patterns rather than corpus artifacts.

**See `eda/bayes.ipynb` in the [code repository](https://github.com/bstadt/tlm-1-release) for a complete example** demonstrating how to use Bayesian query formulation to analyze fills. The notebook shows how to correctly identify which U.S. president is most associated with each year—a task where naive probabilities fail but Bayesian posteriors succeed.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Etin400M |
| Time tokens | 30 (1990-2019) |
| Training data | COCA (750M words) |
| Precision | bf16 |
| Hardware | H100 GPU |
| Batch size | 64 |
| Gradient accumulation | 8 |
| Content masking | Span-based (max length 4) |
| Time token masking | 90% |

### Performance

- Content infill accuracy: ~54%
- Time token accuracy: ~70%

## Applications

1. **Long-arc analysis**: Track monotonic language trends (e.g., sentiment toward the future becoming increasingly negative from 1990-2019)

2. **Diachronic semantic change**: Detect meaning shifts through paradigm graph analysis—visualize how substitute words for a target change over time (e.g., "cell" shifting from biological contexts to technology)

3. **Temporal interpretability**: Time token embeddings self-organize into a "temporal control curve," suggesting potential for forecasting future language patterns

## Resources

- **Model weights**: [huggingface.co/bstadt/tlm-1](https://huggingface.co/bstadt/tlm-1)
- **Code & figure generation**: [github.com/bstadt/tlm-1-release](https://github.com/bstadt/tlm-1-release)
- **Technical report**: [calcifercomputing.com/reports/tlm](https://www.calcifercomputing.com/reports/tlm)

## Citation

```bibtex
@misc{tlm1-2025,
  author = {Duderstadt, Brandon},
  title = {TLM-1: A Temporal Language Model},
  year = {2025},
  publisher = {Calcifer Computing},
  url = {https://www.calcifercomputing.com/reports/tlm}
}
```

## Acknowledgments

This work was supported by **The Cosmos Institute** and **Helivan Corp**.

## License

MIT
