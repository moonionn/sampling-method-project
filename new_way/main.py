import pandas as pd
from prepreprocessing import load_and_scale_data
from Kfold_CrossValidation import kfold_crossValidation
from Models_training import logistic_regression_model, random_forrest_model, svm_model, knn_model, naive_bayes_model
from Samplings_Collection import no_sampling, oversample_balance, undersample_balance, smote_balance, adasyn_balance, gamma_balance

# --- 參數配置 ---
K_FOLD = 5
RANDOM_STATE = 42

# --- 資料集 ---
datasets = {
    'wine': ('../newdataset/new_winequality.csv', 'quality'),
    # 'diabetes': ('../newdataset/new_diabetes.csv', 'diabetes')
}

# --- 採樣方法和模型 ---
sampling_methods = {
    'No Sampling': no_sampling,
    'SMOTE': smote_balance,
    'ADASYN': adasyn_balance,
    'Undersampling': undersample_balance,
    'Oversampling': oversample_balance,
    'Gamma': gamma_balance
}

models = {
    'Logistic Regression': logistic_regression_model,
    'Random Forest': random_forrest_model,
    'SVM': svm_model,
    'KNN': knn_model,
    'Naive Bayes': naive_bayes_model
}

# --- 儲存结果 ---
results = {}

# --- 循環遍歷資料集、採樣方法和模型 ---
for dataset_name, (file_path, target_column) in datasets.items():
    print(f"\n--- Processing dataset: {dataset_name} ---")
    df = pd.read_csv(file_path)
    X, y = load_and_scale_data(df, target_column)

    for sampling_name, sampling_func in sampling_methods.items():
        print(f"  - Sampling method: {sampling_name}")
        X_resampled, y_resampled = sampling_func(X, y)

        for model_name, model_func in models.items():
            print(f"    * Model: {model_name} ") # 添加打印语句，表示模型训练开始
            f1, auc, mean_minority_recall = kfold_crossValidation(X_resampled, y_resampled, model_func, k=K_FOLD)
            results[(dataset_name, model_name, sampling_name)] = (f1, auc, mean_minority_recall) # 将 mean_accuracy 也存储到 results 中

# --- 結果 ---
print("\n--- Results ---")
for (dataset, model, sampling), (f1, auc, mean_minority_recall) in results.items():
    print(f"{dataset} - {model} ({sampling}): F1 = {f1:.4f}, AUC = {auc:.4f}, Mean Minority Recall = {mean_minority_recall:.4f}")
