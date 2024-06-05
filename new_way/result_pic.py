import altair as alt
import pandas as pd
import altair_saver

# 数据准备
models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes']
metrics = ['F1', 'AUC', 'Mean Minority Recall']
sampling_methods = ['No Sampling', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma']

data = {
    'F1': {
        'No Sampling': [0.9802, 0.9805, 0.9799, 0.9799, 0.9799],
        'SMOTE': [0.7710, 0.9679, 0.7727, 0.9106, 0.7046],
        'ADASYN': [0.7650, 0.9703, 0.7656, 0.9083, 0.7035],
        'Undersampling': [0.6446, 0.7141, 0.6339, 0.6819, 0.6493],
        'Oversampling': [0.7323, 0.9961, 0.7507, 0.9584, 0.6884],
        'Gamma': [0.7287, 0.9958, 0.7420, 0.9584, 0.6805]
    },
    'AUC': {
        'No Sampling': [0.7427, 0.8230, 0.6837, 0.6138, 0.7340],
        'SMOTE': [0.8478, 0.9959, 0.8508, 0.9722, 0.7902],
        'ADASYN': [0.8445, 0.9958, 0.8491, 0.9710, 0.7759],
        'Undersampling': [0.6917, 0.7707, 0.6890, 0.6704, 0.7173],
        'Oversampling': [0.8158, 1.0000, 0.8164, 0.9851, 0.7555],
        'Gamma': [0.8094, 1.0000, 0.8101, 0.9851, 0.7541]
    },
    'Mean Minority Recall': {
        'No Sampling': [0.0143, 0.0468, 0.0000, 0.0468, 0.0000],
        'SMOTE': [0.7946, 0.9760, 0.7972, 0.9412, 0.7554],
        'ADASYN': [0.7661, 0.9747, 0.7754, 0.9387, 0.7321],
        'Undersampling': [0.7000, 0.7530, 0.6818, 0.7197, 0.7364],
        'Oversampling': [0.7248, 0.9980, 0.7541, 0.9719, 0.7129],
        'Gamma': [0.7383, 0.9980, 0.7464, 0.9719, 0.7181]
    }
}

# 将数据转换为pandas DataFrame, 初始化DataFrame避免FutureWarning
initial_row = {'Model': models[0], 'Metric': metrics[0], 'Sampling_Method': sampling_methods[0], 'Value': data[metrics[0]][sampling_methods[0]][0]}
df = pd.DataFrame([initial_row])
for metric in metrics:
    for sampling_method in sampling_methods:
        for i, model in enumerate(models):
            # 使用 pd.concat 合并 DataFrame，而不是 append
            df = pd.concat([df, pd.DataFrame({'Model': [model], 'Metric': [metric], 'Sampling_Method': [sampling_method], 'Value': [data[metric][sampling_method][i]]})], ignore_index=True)

# 绘制图表并保存为JSON文件
for model in models:
    # 将scale定义移动到alt.Y中
    chart = alt.Chart(df[df['Model'] == model]).mark_bar().encode(
        x=alt.X('Metric:N', axis=alt.Axis(title='Metric', labelAngle=-45)),
        y=alt.Y('Value:Q', title='Value', scale=alt.Scale(domain=[0, 1])),
        column='Sampling_Method:N',
        color='Sampling_Method:N',
        tooltip=['Metric:N', 'Sampling_Method:N', 'Value:Q']
    ).properties(
        title=f'Model: {model}',
    ).interactive()

    chart.save(f'{model}_results.json')

# 绘图并保存为PNG图片
for model in models:
    chart = alt.Chart(df[df['Model'] == model]).mark_bar().encode(
        x=alt.X('Metric:N', axis=alt.Axis(title='Metric', labelAngle=-45)),
        y=alt.Y('Value:Q', title='Value', scale=alt.Scale(domain=[0, 1])),
        column='Sampling_Method:N',
        color='Sampling_Method:N',
        tooltip=['Metric:N', 'Sampling_Method:N', 'Value:Q']
    ).properties(
        title=f'Model: {model}',
    ).interactive()

    altair_saver.save(chart, f'{model}_results.png', scale_factor=2.0, method='node')  # 保存为PNG, scale_factor放大图像
