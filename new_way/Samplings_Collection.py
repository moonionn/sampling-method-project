# SMOTE
def smote_balance(X, y):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def adasyn_balance(X, y):
    from imblearn.over_sampling import ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

# RUS (Random Under Sampling)
def undersample_balance(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    return X_resampled, y_resampled

# ROS (Random Over Sampling)
def oversample_balance(X, y):
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    return X_resampled, y_resampled


import numpy as np
from sklearn.utils import resample


def gamma_balance(X, y, shape=2.0, scale=1.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # 分離少數類別和多數類別
    minority_class = y.min()
    majority_class = y.max()
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    X_majority = X[y == majority_class]
    y_majority = y[y == majority_class]

    # 計算少數類別樣本數量
    n_minority = len(y_minority)
    n_majority = len(y_majority)

    # 生成 Gamma 分佈的樣本數量
    n_samples_to_generate = n_majority - n_minority
    gamma_samples = np.random.gamma(shape, scale, n_samples_to_generate)

    # 使用 Gamma 分布採樣增加少數類別樣本
    X_resampled_minority = resample(X_minority, replace=True, n_samples=n_samples_to_generate,
                                    random_state=random_state)
    y_resampled_minority = np.full(n_samples_to_generate, minority_class)

    # 合併重新採樣的資料集
    X_resampled = np.vstack((X, X_resampled_minority))
    y_resampled = np.hstack((y, y_resampled_minority))

    return X_resampled, y_resampled
