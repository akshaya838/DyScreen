# DyScreen - Viva Notes

**1. Why did you choose this project?**
Dyslexia is often identified late in Indian schools because specialists are rare. I wanted a low-cost tool that runs in any browser and helps flag children who may need a professional assessment. It also let me apply a full classical ML pipeline to a problem with real social value.

**2. What is dyslexia screening?**
Screening is a quick first check that estimates whether a child might be at risk. It does not confirm anything; it only suggests who should get a detailed assessment from a professional.

**3. Why not deep learning?**
The dataset is tabular with a few thousand rows, where classical models perform well and train in seconds. Classical models are also easier to explain, and this is an ML lab project focused on classical methods.

**4. Why Logistic Regression?**
It is a simple, fast and interpretable baseline. Its coefficients show how each feature pushes the probability up or down, so every other model is compared against it.

**5. Why Random Forest?**
It combines many decision trees trained on random subsets of data (bagging), which reduces overfitting. It handles non-linear patterns and gives feature importance.

**6. Why SVM?**
SVM finds the boundary with the widest margin between the classes, and the RBF kernel can capture curved boundaries. It works well on small, scaled datasets.

**7. Why use SMOTE?**
Only about 1 in 9 participants has dyslexia, so the model sees very few positive examples. SMOTE creates synthetic minority examples by interpolating between real ones. We apply it only to training data, never to the test set.

**8. Why use class weights?**
Class weights make mistakes on the minority class cost more during training, without creating new data. We compare this against SMOTE to see which works better.

**9. What is stratified splitting?**
It splits the data so the train and test sets keep the same class ratio as the full dataset. Without it, the small dyslexia class could end up unevenly split.

**10. What is cross-validation?**
The training data is divided into 5 folds; the model trains on 4 and is tested on the 5th, rotating 5 times. Averaging the scores gives a more reliable estimate than a single split.

**11. Why is recall important?**
Recall is the share of truly at-risk children the model catches. In screening, missing an at-risk child is worse than a false alarm, so we prioritise recall.

**12. What is precision?**
Precision is the share of children flagged as at risk who really are. Low precision means many false alarms, which wastes assessment effort.

**13. What is F1-score?**
F1 is the harmonic mean of precision and recall. It is high only when both are reasonably high, so it balances the two.

**14. What is ROC-AUC?**
The ROC curve plots recall against the false positive rate at every threshold. AUC is the area under it: 0.5 is random guessing and 1.0 is perfect separation.

**15. What is feature engineering?**
Creating better input features from raw data. We turned 192 question-level columns into skill-level scores (phonological, visual, memory, executive), overall accuracy and miss rate, which the game can also produce.

**16. What is feature selection?**
Finding which features actually help prediction. We used correlation, chi-square, mutual information and Recursive Feature Elimination.

**17. Why use StandardScaler?**
It rescales features to mean 0 and standard deviation 1. Distance-based models like KNN and SVM would otherwise be dominated by features with larger numbers, such as age.

**18. What is data leakage?**
Leakage is when information from the test set sneaks into training, making results look better than they really are. We avoid it by putting the imputer, scaler and SMOTE inside the pipeline so they are fitted only on training folds.

**19. What is SHAP?**
SHAP explains a prediction by showing how much each feature pushed the probability up or down. We use it, with permutation importance, to show which skill areas are most linked to higher predicted risk.

**20. Why is this not a diagnosis?**
Diagnosis needs a qualified professional, standardised tests, and ruling out other causes such as vision problems or poor schooling. Our model only estimates risk from game performance, and it was trained on a different population.

**21. How does the frontend communicate with Flask?**
When the games finish, JavaScript sends a JSON POST request to /api/submit-game with the age and each question's clicks, hits and misses. Flask returns JSON with the risk level, probability and skill analysis, which the page displays.

**22. How is the model saved?**
The full pipeline (imputer, scaler and model) is saved with joblib as dyslexia_model.joblib, along with the scaler and feature column order. Flask loads it once at start-up instead of retraining.

**23. How does the live game data match the training data?**
Both sides use the same build_features() function. Each game question records clicks, hits and misses just like each dataset question, and both are grouped into the same four skill areas with the same formulas.

**24. What are the limitations?**
The dataset is from Spanish-speaking participants, the game tasks are not identical to the dataset tasks, some question groupings are approximate, and the risk thresholds are not clinically validated. It has not been tested on Indian children.

**25. What are future enhancements?**
Collect labelled data from Indian children with a special educator, add Tamil and Hindi tasks, recalibrate thresholds on local data, and add a webcam eye-tracking experiment. Age-wise norms and anomaly detection could also help with unlabelled local data.
