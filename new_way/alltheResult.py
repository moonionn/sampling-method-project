import pandas as pd

results_data = {
    ('wine', 'Logistic Regression', 'SMOTE'): (0.7470, 0.8306),
    ('wine', 'Random Forest', 'SMOTE'): (0.9794, 0.9979),
    ('wine', 'SVM', 'SMOTE'): (0.7581, 0.8352),
    ('wine', 'KNN', 'SMOTE'): (0.9175, 0.9780),
    ('wine', 'Naive Bayes', 'SMOTE'): (0.6981, 0.7850),
    ('wine', 'Logistic Regression', 'ADASYN'): (0.7387, 0.8222),
    ('wine', 'Random Forest', 'ADASYN'): (0.9811, 0.9981),
    ('wine', 'SVM', 'ADASYN'): (0.7461, 0.8302),
    ('wine', 'KNN', 'ADASYN'): (0.9149, 0.9769),
    ('wine', 'Naive Bayes', 'ADASYN'): (0.6838, 0.7666),
    ('wine', 'Logistic Regression', 'RUS'): (0.6769, 0.7077),
    ('wine', 'Random Forest', 'RUS'): (0.7242, 0.7745),
    ('wine', 'SVM', 'RUS'): (0.6386, 0.6958),
    ('wine', 'KNN', 'RUS'): (0.6871, 0.6754),
    ('wine', 'Naive Bayes', 'RUS'): (0.6261, 0.7183),
    ('wine', 'Logistic Regression', 'ROS'): (0.7323, 0.8158),
    ('wine', 'Random Forest', 'ROS'): (0.9961, 1.0000),
    ('wine', 'SVM', 'ROS'): (0.7511, 0.8162),
    ('wine', 'KNN', 'ROS'): (0.9587, 0.9847),
    ('wine', 'Naive Bayes', 'ROS'): (0.6884, 0.7555),
    ('wine', 'Logistic Regression', 'Gamma'): (0.7279, 0.8163),
    ('wine', 'Random Forest', 'Gamma'): (0.9968, 1.0000),
    ('wine', 'SVM', 'Gamma'): (0.7396, 0.8148),
    ('wine', 'KNN', 'Gamma'): (0.9587, 0.9847),
    ('wine', 'Naive Bayes', 'Gamma'): (0.6865, 0.7577)
}

# 创建 DataFrame
results_df = pd.DataFrame.from_dict(results_data, orient='index', columns=['F1 Score', 'AUC'])

# 添加数据集、模型和采样方法列
results_df[['Dataset', 'Model', 'Sampling Method']] = pd.DataFrame(results_df.index.tolist(), index=results_df.index)

# 调整列顺序
results_df = results_df[['Dataset', 'Model', 'Sampling Method', 'F1 Score', 'AUC']]

# 打印表格
print(results_df)


results_df.to_csv('results.csv', index=False)  # 保存为 CSV
