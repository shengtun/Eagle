import numpy as np
from scipy.stats import genextreme  # SciPy: c = -xi

def standardized_t(p, xi):
    if xi == 0:
        return -np.log(-np.log(p))
    return ((-np.log(p))**(-xi) - 1.0) / xi

def p_max_weibull(xi, b=0.3):
    assert xi < 0
    t_bound = -1.0/xi - b
    return np.exp(- (1 + xi * t_bound) ** (-1.0/xi))

def pick_percentile(xi, b=0.3):
    # 1) 起点
    if xi > 0:        p0 = 0.97
    elif xi == 0:     p0 = 0.97
    else:             p0 = min(0.98, p_max_weibull(xi, b))  # 带上界约束

    return p0

def gev_quantile(p, xi, mu, sigma):
    c = -xi  # SciPy参数化
    return genextreme.ppf(p, c, loc=mu, scale=sigma)

# # 示例
# xi, mu, sigma = 0.2, 0.0, 1.0
# p = pick_percentile(xi, N=100, k=5)  # 希望100张里约有5张超阈(5%)
# q = gev_quantile(p, xi, mu, sigma)   # 阈值
