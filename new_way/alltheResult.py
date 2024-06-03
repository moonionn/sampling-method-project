import pandas as pd

# Logistic Regression Table
lr_data = {
    'Sampling Method': ['None', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma'],
    'F1-Score': [0.9802, 0.7710, 0.7650, 0.6446, 0.7323, 0.7344],
    'AUC': [0.7427, 0.8478, 0.8445, 0.6917, 0.8158, 0.8165],
    'Minority Recall': [0.0143, 0.7946, 0.7661, 0.7000, 0.7248, 0.7355]
}

lr_df = pd.DataFrame(lr_data)
print("\n--- Logistic Regression ---\n")
print(lr_df.to_markdown(index=False, numalign="left", stralign="left"))

# Random Forest Table
rf_data = {
    'Sampling Method': ['None', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma'],
    'F1-Score': [0.9805, 0.9679, 0.9703, 0.7141, 0.9961, 0.9961],
    'AUC': [0.8230, 0.9959, 0.9958, 0.7707, 1.0000, 1.0000],
    'Minority Recall': [0.0468, 0.9760, 0.9747, 0.7530, 0.9980, 0.9980]
}

rf_df = pd.DataFrame(rf_data)
print("\n--- Random Forest ---\n")
print(rf_df.to_markdown(index=False, numalign="left", stralign="left"))

# SVM Table
svm_data = {
    'Sampling Method': ['None', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma'],
    'F1-Score': [0.9799, 0.7727, 0.7656, 0.6339, 0.7507, 0.7403],
    'AUC': [0.6837, 0.8508, 0.8491, 0.6890, 0.8164, 0.8153],
    'Minority Recall': [0.0000, 0.7972, 0.7754, 0.6818, 0.7541, 0.7653]
}

svm_df = pd.DataFrame(svm_data)
print("\n--- SVM ---\n")
print(svm_df.to_markdown(index=False, numalign="left", stralign="left"))

# KNN Table
knn_data = {
    'Sampling Method': ['None', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma'],
    'F1-Score': [0.9799, 0.9106, 0.9083, 0.6819, 0.9584, 0.9584],
    'AUC': [0.6138, 0.9722, 0.9710, 0.6704, 0.9851, 0.9851],
    'Minority Recall': [0.0468, 0.9412, 0.9387, 0.7197, 0.9719, 0.9719]
}

knn_df = pd.DataFrame(knn_data)
print("\n--- KNN ---\n")
print(knn_df.to_markdown(index=False, numalign="left", stralign="left"))

# Naive Bayes Table
nb_data = {
    'Sampling Method': ['None', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma'],
    'F1-Score': [0.9799, 0.7046, 0.7035, 0.6493, 0.6884, 0.6805],
    'AUC': [0.7340, 0.7902, 0.7759, 0.7173, 0.7555, 0.7554],
    'Minority Recall': [0.0000, 0.7554, 0.7321, 0.7364, 0.7129, 0.7142]
}

nb_df = pd.DataFrame(nb_data)
print("\n--- Naive Bayes ---\n")
print(nb_df.to_markdown(index=False, numalign="left", stralign="left"))
