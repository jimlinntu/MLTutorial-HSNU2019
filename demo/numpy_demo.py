import numpy as np

def L(o, target):
    return np.sum((1/2) * (o - target) ** 2)

# 神經網路
def f(x, W_1, b_1, W_2, b_2):
    h = (x.dot(W_1) + b_1)
    o = h.dot(W_2) + b_2
    return o, h

def gradient(x, W_1, W_2, h, o, target):
    diff = np.reshape(o - target, (1, (o - target).shape[0]))
    h = np.reshape(h, (1, h.shape[0])) # one row
    x = np.reshape(x, (1, x.shape[0])) # one row
    gradient_W_1 = x.T.dot(diff).dot(W_2.T)
    gradient_W_2 = h.T.dot(diff) 
    return gradient_W_1, gradient_W_2


# D = { ([0.05, 0.10], [0.01, 0.99]) }
x = np.array([0.05, 0.10])
target = np.array([0.01, 0.99])
# Learning rate
eta = 0.001
# 初始化神經網路參數 w_1 ~ w_6
W_1 = np.array([[0.15, 0.20], [0.25, 0.30]])
b_1 = np.array([0.35, 0.35])
W_2 = np.array([[0.40, 0.45], [0.50, 0.55]])
b_2 = np.array([0.60, 0.60])
# Training Loop
for i in range(1):
    o, h = f(x, W_1, b_1, W_2, b_2)
    g_W_1, g_W_2 = gradient(x, W_1, W_2, h, o, target)
    W_1 = W_1 - eta * g_W_1
    W_2 = W_2 - eta * g_W_2
    loss = L(o, target)
    print("Loss:")
    print(loss)


print("o:")
print(o)






