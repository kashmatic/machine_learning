from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

## Random generation of arrays of equal length
# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(howmany, variance, step=2, correlation=False):
    val = 1
    ys = []
    ## For number of datapoints we need
    for i in range(howmany):
        ## Get a number between and append it to ys
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        ## If correlation is positive, increase the val
        if correlation and correlation == 'pos':
            val += step
        ## else reduce the val
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

## function to generate the slope
def best_fit_slope(xs, ys):
    i = mean(xs) * mean(ys)
    j = mean(xs*ys)
    k = mean(xs) ** 2
    l = mean(xs ** 2)
    m = (i - j)/(k - l)
    return m

def yintercept(m, xs, ys):
    i = mean(ys)
    j = m * mean(xs)
    b = i - j
    return b

def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line) ** 2)

def coefficient_of_variance(ys_orig, ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, ys_mean_line)
    return 1 - (squared_error_regr/squared_error_mean)
    # return None

xs, ys = create_dataset(40, 100, 2)
m = best_fit_slope(xs, ys)
b = yintercept(m, xs, ys)

## Generate the regression line using the formula
regression_line = [(m * x)+b for x in xs]
# print(m, b)

rsquared = coefficient_of_variance(ys, regression_line)
print(rsquared)

predict_x = 42
predict_y = (m * predict_x)+b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
