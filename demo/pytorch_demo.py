import torch
from torch import optim

def L(o, target):
    return torch.sum((1/2) * (o - target) ** 2)

def f(x, W_1, b_1, W_2, b_2):
    # torch.matmul(): https://pytorch.org/docs/stable/torch.html#torch.matmul
    h = torch.matmul(x, W_1) + b_1
    o = torch.matmul(h, W_2) + b_2
    return o, h

x = torch.tensor([0.05, 0.10])
target = torch.tensor([0.01, 0.99])

W_1 = torch.tensor([[0.15, 0.20], [0.25, 0.30]], requires_grad=True)
b_1 = torch.tensor([0.35, 0.35])
W_2 = torch.tensor([[0.40, 0.45], [0.50, 0.55]], requires_grad=True)
b_2 = torch.tensor([0.60, 0.60])

# 設定好 optimizer(優化器), 可以想像成有一個東西在幫你管參數. 在這裡使用 (stochastic) gradient descent
optimizer = optim.SGD([W_1, W_2], lr=0.001)
# Training Loop
for i in range(1):
    # 先清空存在各個參數旁的 gradient
    optimizer.zero_grad()
    # 把 x 餵給你的神經網路(pytorch 會在背後偷偷幫你建 computational graph)
    o, h = f(x, W_1, b_1, W_2, b_2)
    # 計算這次的 loss
    loss = L(o, target)
    # 計算在這個 computational graph 上面, 那些可以微分的 variable 的微分值
    loss.backward()
    # 幫你做 Gradient Descent(W <- W - eta * g)
    optimizer.step()
    print("Loss:")
    print(loss)
    
print("o:")
print(o)


