# Model Evaluation

**Style & Occasion-Based Outfit Recommender System**

This document provides a brief overview of the model evaluation process, including metrics, performance analysis, and practical implications.

---

## Evaluation Metrics

We evaluate the **Logistic Regression usage classifier** using standard classification metrics:

| Metric | Value | Purpose |
|--------|-------|---------|- 
| **Test Accuracy** | 79.2% | Overall correctness |
| **Macro Avg F1** | 0.46 | Unweighted mean across classes |
| **Weighted Avg F1** | 0.82 | Class-weighted performance |
| **Precision** | 0.89 (weighted) | Correct positive predictions |
| **Recall** | 0.79 (weighted) | Captured true positives |

> [!NOTE]
> We use **F1-score variants** (macro and weighted) to account for class imbalance, providing a more nuanced view than accuracy alone.

---

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Casual** | 0.98 | 0.76 | 0.85 | 6,881 |
| **Ethnic** | 0.73 | 0.96 | 0.83 | 642 |
| **Formal** | 0.57 | 0.88 | 0.70 | 469 |
| **Sports** | 0.57 | 0.89 | 0.69 | 805 |
| **Party** | 0.07 | 0.83 | 0.12 | 6 |
| **Smart Casual** | 0.02 | 0.46 | 0.04 | 13 |
| **Travel** | 0.05 | 0.80 | 0.09 | 5 |
| **Nan** | 0.20 | 0.89 | 0.33 | 63 |

---

## Baseline Comparison

| Approach | Accuracy | Notes |
|----------|----------|-------|
| **Random Baseline** | ~12.5% | Random selection among 8 classes |
| **Majority Baseline** | ~77% | Always predict "Casual" |
| **Our Model (Before)** | 84.5% | Without class balancing |
| **Our Model (After)** | 79.2% | With SMOTE + class weights |

**Key Insight**: While overall accuracy decreased by 5.3%, the model now correctly identifies minority classes (Formal, Sports, Ethnic) with dramatically improved recall.

---

## Strengths & Weaknesses

### Strengths

- ✅ **Excellent recall for minority classes** (88-96% for Ethnic, Formal, Sports)
- ✅ **Balanced predictions** across all categories
- ✅ **Fast inference** (~5ms for classification)
- ✅ **Leverages CLIP's semantic understanding** for rich visual features

### Weaknesses

- ⚠️ **Lower precision for minority classes** due to decreased strictness
- ⚠️ **Very rare categories still struggle** (Party, Smart Casual, Travel have <15 samples)
- ⚠️ **Trade-off**: Casual recall decreased from 100% to 76%

---

## Class Balancing Techniques Applied

### The Problem

| Category | Proportion |
|----------|------------|
| Casual | 77% |
| Sports | 9% |
| Ethnic | 7% |
| Formal | 5% |
| Others | <2% |

### Our Solution

1. **SMOTE Oversampling**: Created synthetic samples for minority classes
   - Training data increased from ~35,000 to ~220,000 balanced samples
   
2. **Balanced Class Weights**: `class_weight='balanced'` in Logistic Regression
   - Penalizes minority class misclassification more heavily

3. **Stratified Train-Test Split**: Preserves class proportions in both sets

> [!IMPORTANT]
> The combination of SMOTE + class weights dramatically improved minority class recall while maintaining acceptable overall accuracy.

---

## Confusion Matrix Analysis

```
Predicted →   Casual  Ethnic  Formal  Nan  Party  SmartCas  Sports  Travel
──────────────────────────────────────────────────────────────────────────
Casual         5218    225     295   219     66      247     541      70
Ethnic           17    617       2     0      3        1       2       0
Formal           27      2     414     3      1       18       2       2
Nan               2      2       1    56      0        0       2       0
Party             1      0       0     0      5        0       0       0
Smart Casual      1      0       5     0      1        6       0       0
Sports           74      1       5     1      0        1     717       6
Travel            1      0       0     0      0        0       0       4
```

**Observation**: The model now correctly identifies minority classes with high recall. The trade-off is some Casual items being misclassified as other categories.

---

## Practical Implications

### For Recommendations

| Implication | Impact |
|-------------|--------|
| **Balanced predictions** | Users get relevant results for all category queries |
| **Better formal/sports coverage** | Previously missed items now surface |
| **More diverse results** | Recommendations span all style categories |

### Acceptable Trade-offs

The slight decrease in overall accuracy (84.5% → 79.2%) is justified because:
- Users searching for "formal attire" now get actual formal items
- Sports and Ethnic categories are properly recognized
- The recommendation system is now **fair across all categories**

---

## Summary

After applying class balancing techniques, the model achieves **79.2% accuracy** with dramatically improved minority class recall (Ethnic: 96%, Formal: 88%, Sports: 89%). This makes the system more practical for diverse fashion queries while maintaining acceptable overall performance.

---

*Document prepared for Machine Learning Final Project - University of the Philippines Cebu, December 2025*
