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
import torch
import torch.nn.functional as F

model = AutoModelForMaskedLM.from_pretrained("bstadt/tlm-1")
tokenizer = AutoTokenizer.from_pretrained("bstadt/tlm-1")
model.eval()
```

### Bayesian Query Formulation

Raw fill probabilities from the model are biased by training data frequencies. For example, "Obama" dominates naive predictions for "President [MASK]" across *all* years because he appears most frequently in COCA. Our Bayesian Query Framework corrects for this anachronism:

$$P(Fill| Context, Time) = \frac{P(Time | Context; Fill)}{P(Time | Context)}P(Fill | Context)$$

(See our [technical report](https://www.calcifercomputing.com/reports/tlm) for the full derivation)

```python
def lyear(phrase, model, tokenizer):
    years = list(range(1990, 2020))
    year_tokens = [f'[YEAR:{y}]' for y in years]
    year_token_ids = [tokenizer.encode(t)[1] for t in year_tokens]

    # Mask the year position
    input_ids = tokenizer.encode('[MASK] ' + phrase, add_special_tokens=False, return_tensors='pt')

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0][0]
        year_probs = F.softmax(logits[year_token_ids], dim=0)

    return years, year_probs

def bayes_factor(filled_phrase, template_phrase, model, tokenizer):
    years, fill_probs = lyear(filled_phrase, model, tokenizer)
    _, template_probs = lyear(template_phrase, model, tokenizer)
    return years, fill_probs / template_probs

# Example: What is the most likely fill for each year?
template = "President [MASK] made a speech today"

# Candidate fills can be manually specified or surfaced from nontemporal model likelihood
candidates = ["Trump", "Obama", "Bush", "Clinton"]

bayes_factors = {}
for name in candidates:
    filled = f"President {name} made a speech today"
    years, bf = bayes_factor(filled, template, model, tokenizer)
    bayes_factors[name] = bf

# Normalize to get posteriors (Assumes uniform probability over fills)
import numpy as np
all_bf = np.stack([bayes_factors[n].numpy() for n in candidates])
posteriors = all_bf / all_bf.sum(axis=0)

# Result: posteriors now correctly peak during each president's actual term
# Clinton peaks 1992-2000, Bush peaks 2000-2008, Obama peaks 2008-2016, Trump peaks 2016-2019
```

**See `eda/bayes.ipynb` in the [code repository](https://github.com/bstadt/tlm-1-release) for the complete example** with visualizations showing how naive probabilities fail while Bayesian posteriors correctly identify presidential terms.

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
  author = {Duderstadt, Brandon and Helm, Hayden},
  title = {A Model of the Language Process},
  year = {2025},
  publisher = {Calcifer Computing},
  url = {https://www.calcifercomputing.com/reports/tlm}
}
```

## Acknowledgments

This work was supported by **The Cosmos Institute** and **Helivan Corp**.

## License

MIT
