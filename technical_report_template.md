# Technical Report: Comparative Study of Traditional Machine Learning and Transformer Models for Spam Classification

**Author:** [Your Name]  
**Internship:** [Organization / Internship Program]  
**Date:** [Submission Date]

## 1. Abstract
This project compares two classical text-classification models—Logistic Regression and Multinomial Naive Bayes—with a fine-tuned DistilBERT transformer for classifying emails as spam or ham. Classical models use TF-IDF features, while DistilBERT learns contextual representations from tokenized text. All models are evaluated on the same held-out test split using accuracy, spam precision, spam recall, spam F1-score, confusion matrices, and measured training time.

## 2. Dataset Exploration and Preparation
The dataset was obtained from the Kaggle source specified in the assignment: https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification. The script prints the dataset dimensions, first rows, missing-value counts, duplicate count, detected text/label columns, and class distribution. It removes rows with missing labels, empty messages, and duplicate text-label pairs. Text normalization is intentionally conservative: null bytes are replaced and repeated whitespace is collapsed. URLs, numbers, punctuation, and other message cues are retained because they may be useful for spam detection.

**Dataset-specific findings:** Fill this section from `outputs/experiment_summary.json`, the console log, and `outputs/class_distribution.png`. Do not claim counts before running the experiment.

## 3. Experimental Design
The data is split into 70% training, 15% validation, and 15% test subsets using stratified sampling and a fixed random seed (42). Stratification helps preserve class proportions. The test split is kept separate from model fitting. For the traditional pipelines, TF-IDF is fitted only on the training split to prevent vocabulary leakage.

### 3.1 Traditional models
TF-IDF uses lowercasing, Unicode accent stripping, unigrams and bigrams, sublinear term-frequency scaling, a minimum document frequency of two, and a vocabulary cap of 100,000 features. Logistic Regression uses class balancing and an increased iteration limit. Multinomial Naive Bayes uses additive smoothing (alpha = 0.5). These models are computationally efficient and serve as interpretable baselines.

### 3.2 DistilBERT
The transformer is `distilbert-base-uncased` with a two-class sequence-classification head. Text is tokenized with truncation and fixed-length padding (maximum length 192 by default). Fine-tuning uses a custom PyTorch training loop, AdamW (learning rate 2e-5, weight decay 0.01), gradient clipping (maximum norm 1.0), and a default of two epochs. The model with the best validation spam F1-score is retained for test evaluation. No high-level Hugging Face Trainer API is used. Record the actual hardware/device from `outputs/experiment_summary.json`.

## 4. Results
Insert the actual output from `outputs/model_comparison.csv` here. Include all models and their test accuracy, spam precision, spam recall, spam F1-score, training time, and inference time.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score | Training time (s) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [run result] | [run result] | [run result] | [run result] | [run result] |
| Multinomial Naive Bayes | [run result] | [run result] | [run result] | [run result] | [run result] |
| DistilBERT | [run result] | [run result] | [run result] | [run result] | [run result] |

Include the generated figures: `metric_comparison.png`, all `confusion_*.png` files, `distilbert_training_curves.png`, and `training_time_comparison.png`. Discuss false positives (ham predicted as spam) and false negatives (spam predicted as ham) using the confusion matrices.

## 5. Comparative Discussion
Use the measured results rather than assuming one model is best. TF-IDF models often train quickly and can perform strongly when spam vocabulary is distinctive. Naive Bayes is simple and efficient, while Logistic Regression can learn discriminative weights across sparse features. DistilBERT can use context and word relationships beyond a bag-of-words representation, but fine-tuning is typically more computationally expensive and may require GPU resources. The actual ranking depends on this dataset, split, hyperparameters, and training conditions.

Compare:
- Which model has the highest spam F1-score?
- Which model has the lowest false-negative count?
- How much longer did DistilBERT take to train than the classical models?
- Do the training and validation loss curves suggest stable learning or possible overfitting?
- Is the extra computational cost justified by the metric improvement?

## 6. Limitations
1. Results depend on the dataset's size, source, labeling quality, and representativeness of real email traffic.
2. A random split may not measure performance on future emails or distribution shifts.
3. TF-IDF cannot fully model contextual meaning and long-range relationships.
4. DistilBERT training is more computationally demanding; results may vary by hardware, random seed, and library version.
5. The default hyperparameters are starting points, not guaranteed optimal values. More systematic tuning may improve results.
6. The script uses a single fixed split and seed; repeated runs or cross-validation would give a stronger estimate of variability.
7. Spam/ham label interpretation must be checked against the printed label mapping.

## 7. Conclusion
Complete this section after running the experiment. State which model performed best on the held-out test set, support the conclusion with the measured metrics, and discuss whether the improvement (if any) from DistilBERT is worth its additional training cost. Do not leave placeholder results in the submitted report.
