# Support Ticket Classification Pipeline

## Overview

NLP classification pipeline for support ticket categorization using multiple feature extraction methods and classifiers.

**Main Notebook:** `Analysis.ipynb`

## Repository Structure

### Root (Active Project)
```
CLAUDE.md                              # Project instructions & conventions
full_classification_pipeline.ipynb      # Main classification pipeline
requirements.txt                       # Python dependencies
README.md                              # This file
```

### `/dataset` (Actively Referenced Data)
```
support-ticket-classification.csv  # 798 labeled support tickets
lemma.en.txt                              # BNC-derived lemma dictionary (186K entries)
stopwords.txt                             # English stopwords (127 entries)
wordnet_noun.exc                          # WordNet irregular noun forms (2K entries)
wordnet_verb.exc                          # WordNet irregular verb forms (2.4K entries)
```

## Dataset

The dataset is available in the `/dataset/support-ticket-classification.csv` or
can be downloaded from
[Kaggle](https://www.kaggle.com/datasets/devtry3d/support-ticket-classification).
