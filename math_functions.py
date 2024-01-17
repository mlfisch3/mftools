import numpy as np
import mpmath as mp

#  bayes_bias(N, M, granularity=100):
#  bias_probability(bias, N, M):


inf = float('inf')

def bias_probability(bias, N, M):
    return 1 / (1 + 1 / ((np.power(1-2*bias, N-M) * np.power(1+2*bias, M)))) # probability of bias vs fair

bias_probability_vec = np.vectorize(bias_probability)

def bayes_bias(N, M, granularity=100):
    points = np.complex(0, granularity)
    probs = bias_probability_vec(np.r_[-0.499999:0.499999:points], N, M)
    return probs[probs.argmax()]

bayes_bias_vec = np.vectorize(bayes_bias)

N = 22
a = 12
b = N+1
x = np.r_[a:b]
y = bayes_bias_vec(N,x.tolist(), granularity=1000)
plt.plot(x, y)

mpmath.mp.pretty = True
mpmath.mp.dps = 100
# mpmath.power(0.00001, 100)
hundred = mpmath.mpmathify('100')
hundreth = mpmath.mpmathify('0.01')
one = mpmath.fprod([hundred, hundreth])
e_inv = mpmath.exp(-1)
e = mpmath.exp(1)
print(e)
print(e_inv)
print(mpmath.fprod([e, e_inv]))
# 2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427
# 0.3678794411714423215955237701614608674458111310317678345078368016974614957448998033571472743459196437
# 1.0
