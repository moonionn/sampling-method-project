from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def display_confusion_matrix(y_true, y_pred):

    # Generating the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Assuming binary classification for simplicity; adjust if needed for multi-class
    cm_df = pd.DataFrame(cm,
                         index=['Actual Negative:0', 'Actual Positive:1'],
                         columns=['Predicted Negative:0', 'Predicted Positive:1'])

    # Extracting TN, FP, FN, TP
    # TN, FP, FN, TP = cm.ravel()
    # print(f"True Negatives (TN): {TN}")
    # print(f"False Positives (FP): {FP}")
    # print(f"False Negatives (FN): {FN}")
    # print(f"True Positives (TP): {TP}")

    return cm_df

def minority_class_accuracy(y_true, y_pred):
    """
    計算少數類別的準確率。

    參數：
    - y_true: 真實標籤。
    - y_pred: 預測標籤。
    """
    # 找出少數類別
    classes, counts = np.unique(y_true, return_counts=True)
    minority_class = classes[np.argmin(counts)]

    # 計算少數類別預測正確的數量
    true_positives = np.sum((y_true == minority_class) & (y_pred == minority_class))

    # 計算少數類別總數
    total_minority_samples = np.sum(y_true == minority_class)

    # 計算少數類別召回率
    minority_recall = true_positives / total_minority_samples

    return minority_recall

def logistic_regression_model(X_train, y_train):
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model

def random_forrest_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def svm_model(X_train, y_train):
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)  # 訓練模型
    return svm_model

def naive_bayes_model(X_train, y_train):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def knn_model(X_train, y_train):
    # 創建 KNN 模型，選擇鄰居數量
    knn_model = KNeighborsClassifier(n_neighbors=5)
    # 訓練模型
    knn_model.fit(X_train, y_train)
    return knn_model