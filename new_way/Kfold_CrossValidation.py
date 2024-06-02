from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score


def kfold_crossValidation(data, labels, model_func, k=5, balance_function=None, **kwargs):
    """
    使用 k-fold 交叉驗證評估模型性能。

    参数：
    - data: 特徵數據
    - labels: 標籤數據
    - model_func: 模型構建函函數
    - k: k-fold 交叉驗證的折數，默認為5
    - balance_function: 可選的函數，用於訓練集上執行數據平衡處理
    - kwargs: 傳遞给模型構造函数的額外參數

    返回：
    - mean_f1_score: 平均F1-score
    - mean_auc: 平均AUC
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []
    aucs = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if balance_function is not None:
            X_train, y_train = balance_function(X_train, y_train)

        model = model_func(X_train, y_train, **kwargs)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 计算 AUC 时需要预测概率

        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        f1_scores.append(f1)
        aucs.append(auc)

    mean_f1_score = sum(f1_scores) / len(f1_scores)
    mean_auc = sum(aucs) / len(aucs)
    return mean_f1_score, mean_auc