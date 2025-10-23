# Hospital readmission prediction for diabetes patients
By Dr. David Wasserman

## Analysis goal:
Given the data from a hospital visit, predict whether or not the patient will be readmitted within 30 days. I will train a classifier on a subset of the dataset, and test it on an unseen portion of the dataset, with the goal of achieving a high area under curve (AUC) metric.

**Note**: medical practice has changed since 2008, so the resulting model should not be expected to work well with current data. It is intended to shows the skills I would use if I was working with current data.

# Dataset citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.

As of 12 Sept 2025, https://doi.org/10.24432/C5230J redirects to https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 . File diabetes+130-us+hospitals+for+years+1999-2008.zip retrieved from this site on 12 Sept 2025 by pressing "Download". Files diabetic_data.csv and IDS_mapping.csv were extracted from this zip file.

# Paper written by the dataset creators:

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed research international, 2014(1), 781670.

I will refer to this paper as the "original paper". I will refer to the work described in this paper as the "original study".

# Implementation
The analysis is done in the Jupyter notebook [https://github.com/david-r-wasserman/Hospital-readmission-prediction-project/blob/main/Readmission.ipynb](Readmission.ipynb). The analysis is repeatable. The notebook ran in 5 hours 22 minutes on Intel Core i-7-10750 H CPU @ 2.60 GHz.

# Results

I trained and tested four models. Here are the AUC values on the test set:
- LightGBM: 0.6679
- XGBoost: 0.6721
- CatBoost: 0.6649
- Ensemble of the three above: 0.6720

# Comparison to other people's results

I found three recent, high-quality, open-access studies that also trained classifiers on part of the same dataset, tested them on an unseen portion, and reported AUC values:

1. Emi-Johnson, Oluwabukola G., and Kwame J. Nkrumah. "Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data." Cureus 17.4 (2025).
2. Lu, Haohui, and Shahadat Uddin. "Explainable stacking-based model for predicting hospital readmission for diabetic patients." Information 13.9 (2022): 436.
3. Liu, Vincent B., Laura Y. Sue, and Yingnian Wu. "Comparison of machine learning models for predicting 30-day readmission rates for patients with diabetes." Journal of Medical Artificial Intelligence 7 (2024).

[1] tested four different types of models. XGBoost performed best, with AUC 0.667.

[2] tested six types of models, and five of these were used to form an ensemble classifier using a method called stacking. Stacking performed best, with AUC 0.6736.

[3] tested 11 types of models. XGBoost performed best, with AUC 0.64.

None of these studies included the step of using only one sample per patient, so they may involve some data leakage, as I explained in Section 3 of [https://github.com/david-r-wasserman/Hospital-readmission-prediction-project/blob/main/Readmission.ipynb](Readmission.ipynb).
