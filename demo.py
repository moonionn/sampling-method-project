import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 設定參數
alpha = 3
theta = 2
x = np.linspace(0, 20, 1000)

# 計算伽瑪分佈的PDF
pdf = gamma.pdf(x, alpha, scale=theta)

# 繪製伽瑪分佈的PDF
plt.plot(x, pdf, label=f'Gamma Distribution (α={alpha}, θ={theta})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Gamma Distribution PDF')
plt.legend()
plt.show()
