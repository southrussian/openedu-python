import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('queue.csv')


def loss_func(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].Y

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return  m, b

plt.scatter(data.X, data.Y)
plt.show()