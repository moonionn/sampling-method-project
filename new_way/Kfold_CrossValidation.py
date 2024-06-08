# Kfold_CrossValidation.py
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from Models_training import minority_class_accuracy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def kfold_crossValidation(data, labels, model_func, k=5, balance_function=None, **kwargs):
    """
    使用 k-fold 交叉驗證評估模型性能。

    参数：
    - data: 特徵數據 (DataFrame)
    - labels: 標籤數據
    - model_func: 模型構建函函數
    - k: k-fold 交叉驗證的折數，默認為5
    - balance_function: 可選的函數，用於訓練集上執行數據平衡處理
    - kwargs: 傳遞给模型構造函数的額外參數

    返回：
    - mean_f1_score: 平均F1-score
    - mean_auc: 平均AUC
    - mean_minority_accuracy: 平均少數類別準確率
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []
    aucs = []
    minority_accuracies = []
    accuracies = []

    for train_index, test_index in kf.split(data):
        # 将 X_train 和 X_test 轉換回 DataFrame，使用 data 的列名
        X_train = pd.DataFrame(data.iloc[train_index], columns=data.columns)
        X_test = pd.DataFrame(data.iloc[test_index], columns=data.columns)

        # Convert labels to Pandas Series
        labels = pd.Series(labels)

        # Reset the indices of y_train and y_test after undersampling
        y_train, y_test = labels.iloc[train_index].reset_index(drop=True), labels.iloc[test_index].reset_index(
            drop=True)

        # MinMax 縮放 (在每次迭代中分別縮放)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 數據平衡處理 (在縮放後進行)
        if balance_function is not None:
            X_train_scaled, y_train = balance_function(X_train_scaled, y_train)

        # 模型訓練與預測
        model = model_func(X_train_scaled, y_train, **kwargs)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # 性能評估
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        minority_acc = minority_class_accuracy(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        f1_scores.append(f1)
        aucs.append(auc)
        minority_accuracies.append(minority_acc)
        accuracies.append(acc)

    mean_f1_score = sum(f1_scores) / len(f1_scores)
    mean_auc = sum(aucs) / len(aucs)
    mean_minority_accuracy = sum(minority_accuracies) / len(minority_accuracies)


    return mean_f1_score, mean_auc, mean_minority_accuracy
