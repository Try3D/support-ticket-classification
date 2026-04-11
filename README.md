# Support Ticket Classification Pipeline

## Overview

NLP classification pipeline for support ticket categorization using multiple feature extraction methods and classifiers.

**Main Notebook:** `full_classification_pipeline.ipynb`

---

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
support-ticket-classification-sorted.csv  # 798 labeled support tickets
lemma.en.txt                              # BNC-derived lemma dictionary (186K entries)
stopwords.txt                             # English stopwords (127 entries)
wordnet_noun.exc                          # WordNet irregular noun forms (2K entries)
wordnet_verb.exc                          # WordNet irregular verb forms (2.4K entries)
```

### `/tmp` (Documentation, Analysis, Experimental Code)

All supporting documentation and experimental code:
- `ALGORITHM.md` — Detailed morphy algorithm documentation
- `COMPARISON_RESULTS.md` — Comprehensive comparison of OLD vs NEW implementations
- `LEMMATIZATION_IMPROVEMENTS.md` — Summary of all 4 bug fixes
- `compare_implementations.py` — Script comparing implementations on real data
- `test_lemmatizer.py` — Comprehensive test suite (20 tests)
- Plus: old notebooks, experimental code, analysis files, etc.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Pipeline
```bash
cd /Users/rsaran/Projects/nlp
./venv/bin/python -m jupyter notebook full_classification_pipeline.ipynb
```

### 3. Run Tests (to verify lemmatization)
```bash
./venv/bin/python tmp/test_lemmatizer.py
```

---

## What's Been Improved

### Lemmatization Algorithm (WordNet Morphy)

The pipeline implements **WordNet's morphy algorithm** (Fellbaum 1998), a research-backed lemmatizer using only datasets + rules (no external NLP libraries).

**Key Improvements:**
- ✅ Fixed broken lemma.en.txt parser (now handles 90.5K valid entries)
- ✅ Fixed morphology selection (returns first-valid, not non-words)
- ✅ Fixed rule ordering (longest-suffix-first)
- ✅ Fixed priority chain (exceptions → dictionary → rules)

**Impact:**
- 21.6% more tokens get valid lemmatization
- 92.4% of tickets have improved preprocessing
- Vocabulary reduced by 12.9% (better consolidation)

### Feature Extraction Methods

1. **TF-IDF** (custom implementation)
2. **Word2Vec** (Google News 300d pre-trained)
3. **GloVe** (Wikipedia Gigaword 100d pre-trained)
4. **BERT** (bert-base-uncased, mean-pooled)

Each method tested on both raw and preprocessed text.

### Classifiers

- SVM (RBF kernel)
- Logistic Regression
- K-Nearest Neighbors (k=5)

---

## Data References

| File | Size | Purpose | Format |
|------|------|---------|--------|
| `support-ticket-classification-sorted.csv` | 135 KB | 798 labeled tickets | CSV (subject, text, label) |
| `lemma.en.txt` | 2.3 MB | BNC-derived lemmas | `lemma/freq -> form1,form2,...` |
| `stopwords.txt` | 622 B | 127 English stopwords | Text (one per line) |
| `wordnet_noun.exc` | 38 KB | Irregular noun plurals | `irregular base` |
| `wordnet_verb.exc` | 38 KB | Irregular verb forms | `irregular base` |

---

## Project Instructions

See `CLAUDE.md` for coding standards:
- Python: Use venv at `./venv/bin/python`
- Commits: Never add Claude as co-author
- Code: Follow existing patterns

---

## Documentation

**For implementation details:**
- `tmp/ALGORITHM.md` — Complete morphy algorithm with walkthroughs
- `tmp/COMPARISON_RESULTS.md` — Before/after analysis

**For test results:**
- `tmp/test_lemmatizer.py` — Run this to verify lemmatization works correctly
- `tmp/compare_implementations.py` — Compare OLD vs NEW vs CURRENT on real data

---

## License & Attribution

- **Lemma dictionary:** Lin Wei, BNC-derived (https://github.com/skywind3000/lemma.en.txt)
- **WordNet:** Princeton University (https://wordnet.princeton.edu)
- **Word2Vec:** Google News vectors (pre-trained)
- **GloVe:** Pennington et al., Wikipedia Gigaword
- **BERT:** Google (bert-base-uncased)

---

## References

1. Fellbaum, C. (1998). *WordNet: An Electronic Lexical Database.* MIT Press.
2. Miller, G.A., et al. (1990). "Introduction to WordNet." *International Journal of Lexicography*, 3(4):235–244.
3. Krovetz, R. (1993). "Viewing morphology as an inference process." *ACM SIGIR*, pp. 191–202.
