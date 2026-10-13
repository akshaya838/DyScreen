# DyScreen - Result Analysis (auto-generated from real results)

## Model comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Class weight) | 0.658 | 0.154 | 0.487 | 0.234 | 0.610 |
| Logistic Regression (SMOTE) | 0.654 | 0.158 | 0.513 | 0.241 | 0.604 |
| KNN (SMOTE) | 0.664 | 0.167 | 0.538 | 0.255 | 0.657 |
| SVM (Class weight) | 0.896 | 0.625 | 0.064 | 0.116 | 0.738 |
| SVM (SMOTE) | 0.750 | 0.229 | 0.564 | 0.326 | 0.748 |
| Decision Tree (Class weight) | 0.623 | 0.142 | 0.500 | 0.221 | 0.583 |
| Decision Tree (SMOTE) | 0.597 | 0.138 | 0.526 | 0.218 | 0.596 |
| Random Forest (Class weight) | 0.705 | 0.178 | 0.487 | 0.261 | 0.672 |
| Random Forest (SMOTE) | 0.716 | 0.191 | 0.513 | 0.279 | 0.649 |
| XGBoost (Class weight) | 0.802 | 0.246 | 0.410 | 0.308 | 0.709 |
| XGBoost (SMOTE) | 0.812 | 0.271 | 0.449 | 0.338 | 0.688 |

## Class balance

```
MODULE 4 - CLASS IMBALANCE (training set only; SMOTE is never applied before the split)
Before balancing:  Class 0 = 2601   Class 1 = 314
After SMOTE:       Class 0 = 2601   Class 1 = 2601
Class-weight approach: data unchanged, minority errors weighted about 8.28x more.
```

## Selected model

The model was selected on **cross-validated recall** (among models with CV precision of at least 0.20), not on accuracy. Among the evaluated models, **SVM (Class weight)** achieved a cross-validated dyslexia-class recall of **0.576**. On the held-out test set its recall was **0.064**, precision **0.625**, F1-score **0.116** and ROC-AUC **0.738**.

On the test set it correctly flagged **5** of **78** dyslexic participants and missed **73**. It also produced **3** false positives out of **651** non-dyslexic participants, which remain a consideration: a false positive means a child is referred for assessment unnecessarily.

For comparison, the highest *accuracy* was **0.896** (SVM (Class weight)) - this shows why accuracy alone is misleading on imbalanced data.

## Most influential features (permutation importance, drop in recall)

- memory_score: 0.0385
- executive_score: 0.0295
- visual_score: 0.0038
- age: 0.0038
- miss_rate: -0.0115
- phonological_score: -0.0603
- overall_accuracy: -0.1205

## Classification report (selected model)

```
              precision    recall  f1-score   support

 No dyslexia       0.90      1.00      0.94       651
    Dyslexia       0.62      0.06      0.12        78

    accuracy                           0.90       729
   macro avg       0.76      0.53      0.53       729
weighted avg       0.87      0.90      0.86       729

```

## Practical insights

1. The system can provide an inexpensive first-level screening aid.
2. Browser-based games reduce the need for specialised hardware such as eye trackers.
3. Skill-level features make the output more interpretable than a simple yes/no.
4. Recall is prioritised because a screening system should reduce missed at-risk cases.
5. Class imbalance strongly affects evaluation - compare the accuracy and recall columns above.
6. The system can help identify children who may benefit from professional assessment.
7. It does not replace teachers, special educators, psychologists or clinicians.
8. Only anonymous game-performance data is collected (no name, contact, photo or video).

## Limitations

- Trained on Spanish-speaking participants; transfer to Indian children is not yet validated.
- Risk thresholds are prototype values, not clinically validated.
- The browser game produces the same feature types as the dataset, but on different tasks.