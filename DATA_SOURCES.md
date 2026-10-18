# Data and model provenance

## Clothing reviews (main experiment)

- Original: nicapotato, [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews), version 1, CC0 public-domain dedication.
- Downloaded directly from the public Kaggle API, without scraping a retailer.
- 23,486 original rows. Exact cleaning/filtering/sampling counts are recorded in `data/splits.json` and the report.
- The original CSV is included in `data/raw/clothing_reviews.csv`. Its SHA-256 is `bd93cc515747ad1f87b8bc863c9e40da758509e6db6506a96d06976e61e36ee0`.
- `data/reviews.jsonl` contains only the 10,000 sampled body texts, original row IDs, ratings, derived sentiment labels and normalized-text hashes. Metadata is excluded from model features.
- Labels are rating-derived: 1–2 negative, 3 neutral, 4–5 positive. They are weak labels, not independent human annotations of the words. A 3-star review can be mixed, mildly negative or mildly positive.

## Yelp sentences (out-of-domain test only)

Kotzias, D. (2015). [Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences). UCI Machine Learning Repository. [DOI: 10.24432/C57604](https://doi.org/10.24432/C57604). CC BY 4.0, [license](https://creativecommons.org/licenses/by/4.0/).

This project uses only `yelp_labelled.txt` from the archive, 500 negative and 500 positive restaurant-review sentences. The distributed processed copy applies Unicode/HTML/whitespace cleaning and adds row IDs; it preserves source labels. It is never used to train, choose a model, tune a hyperparameter or fit calibration. No neutral examples are provided. This source tests both domain shift and sentence-versus-review length shift.

The associated paper is Kotzias, Denil, de Freitas and Smyth, *From Group to Individual Labels Using Deep Features*, KDD 2015. Exact archive and processed-file hashes are in `data/ood_provenance.json`.

## Pretrained transformer

- [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny), MIT, checkpoint revision `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`.
- Two transformer layers, hidden size 128. This is a language-pretrained encoder, not a pre-existing sentiment classifier. A fresh sentiment head and all encoder parameters are fine-tuned here on the saved training IDs.
- Model source cites Turc et al., [Well-Read Students Learn Better](https://arxiv.org/abs/1908.08962), and Bhargava et al., [Generalization in NLI](https://arxiv.org/abs/2110.01518). See the model card for original pretraining provenance. No known labeled clothing-review fine-tuning precedes this experiment; unknown pretraining-text overlap cannot be ruled out.
- Original checkpoint files are included in `models/pretrained/`. Their hashes are recorded in `data/splits.json`. Fine-tuned models are saved as safetensors with explicit label mappings.

## Implementation references

- [Hugging Face sentiment quickstart](https://huggingface.co/blog/sentiment-analysis-python).
- [BERT in Transformers](https://huggingface.co/docs/transformers/model_doc/bert).
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Generated predictions and metric files are outputs of the included code, not copied benchmark scores.
