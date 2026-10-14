# Product Category Classification from Text (Etsy)

Predicts a product's category and colour from its title, description and tags. Built on about 245,000 Etsy product listings, comparing four text classification models.

## Business problem

Sellers on marketplaces like Etsy often put products in the wrong category or leave fields empty. That hurts search, recommendations and inventory planning. If the category can be predicted from the text a seller already writes, the platform can auto-fill or correct it, and the business gets cleaner data for forecasting demand per category.

The model predicts three labels:

- **Top category** (a few broad classes, e.g. jewellery, clothing, home)
- **Bottom category** (100+ detailed classes)
- **Colour**

## Data

- Etsy product listings provided by the company as Parquet files (not included in this repo)
- **245,485 rows, 21 columns**: product ID, title, description, tags, attributes (material, occasion, style…) and the target IDs
- Columns with more than 50% missing values (room, craft type, recipient, material, occasion, holiday, art subject, style, shape, pattern) were dropped

## Approach

1. **Clean text** – lowercase, remove URLs, digits, special characters and extra spaces; fill missing text with blanks.
2. **Remove stop words** and **lemmatise** with part-of-speech tags (NLTK WordNet) to shrink the vocabulary.
3. **Combine** title, description and tags into one text field.
4. **Vectorise** with `CountVectorizer` (top 10,000 words).
5. **Train in batches** of 20,000 rows to fit in memory, with an 80/20 train/validation split.
6. **Compare models**: Logistic Regression (baseline), Multinomial Naive Bayes, linear SVM, Decision Tree.

## Results (top category)

| Model | Train accuracy | Validation accuracy | Validation F1 |
|---|---|---|---|
| **Logistic Regression** | 0.99 | **0.89** | **0.89** |
| SVM | 0.93 | 0.83 | 0.83 |
| Multinomial NB | 0.88 | 0.78 | 0.79 |
| Decision Tree | 0.99 | 0.74 | 0.74 |

- The simplest model, Logistic Regression, won. Clean text mattered more than a complex algorithm.
- Bottom category scored lower because it has 100+ classes.
- Colour was hardest: the colour written in the text often did not match the labelled colour.
- There is a gap of about 0.10 between train and validation scores, so the models overfit somewhat.

Full write-up: [Report.pdf](Report.pdf)

## Tech stack

Python · pandas · PyArrow · NLTK · scikit-learn

## How to run

```bash
git clone https://github.com/Omkyao1998/Product_Category_Classification.git
cd Product_Category_Classification
pip install -r requirements.txt
jupyter notebook product_category_classification.ipynb
```

Set `PATH` in the second cell to the folder that contains `parquet/train`.

## What I would improve next

- Fit the vectoriser once on all training data. Right now the vectoriser and model are refitted for every batch, so each batch effectively trains a fresh model. `HashingVectorizer` with `SGDClassifier.partial_fit` would let the model learn from all 245K rows.
- Try TF-IDF and class weights for the unbalanced bottom categories.
- Add regularisation tuning to close the train/validation gap.
- Predict all three labels with one multi-output model.

## Author

**Omkar Gokhale** — Data Engineer & BI Analyst · [GitHub](https://github.com/Omkyao1998)
