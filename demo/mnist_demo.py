from torchvision import datasets
import numpy as np
import torch
from torch import nn
from torch import optim
DEBUG = False

# 宣告我們的神經網路的矩陣參數
class Nets(nn.Module):
    def __init__(self):
        super(Nets, self).__init__()
        self.relu = nn.ReLU() # 請看: https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU
        self.W_1 = nn.Linear(784, 100, bias=False)
        self.W_2 = nn.Linear(100, 10, bias=False)
        self.softmax = nn.Softmax(dim=1) # 會幫你把最後一個維度做 softmax
        
    def forward(self, x):
        output = self.W_1(x)
        output = self.relu(output)
        output = self.W_2(output)
        return output

    def predict(self, x):
        output = self.forward(x)
        output = self.softmax(output)
        return output
        


mnist_dataset = datasets.MNIST('./data', train=True, download=True)
mnist_testset = datasets.MNIST('./data', train=False, download=True)

print("Number of training set's examples: {}".format(len(mnist_dataset)))
print("Number of training set's examples: {}".format(len(mnist_testset)))
if DEBUG:
    # 看看第一張圖片長什麼樣子
    mnist_dataset[0][0].show()


# 特別注意: 這個函數會幫你 softmax 完後再算 cross entropy(也就是我們投影片所列的損失函數)
loss = nn.CrossEntropyLoss()
net = Nets()
optimizer = optim.SGD(net.parameters(), lr=0.0001)
train_iterations = 4


# 訓練
for i in range(train_iterations):
    # 看 100 筆資料
    loss_value_list = []
    for j in range(1000):
        # 把 MNIST 資料讀進來存成 torch.Tensor(注意一下, 這裏 x 要轉型成 float, y 要轉成 long)
        x = torch.from_numpy(np.array(mnist_dataset[j][0])).float()
        y = torch.from_numpy(np.array(mnist_dataset[j][1])).long()
        # 把 x,y 的形狀轉換一下 (1 (代表 batch_size), 784)
        x = x.view(1, 784)
        y = y.view(1)
        # 清空 gradient 
        optimizer.zero_grad()
        # 把 x 餵給模型 實際上會跑 forward() 這個函數
        y_hat = net(x)
        # 計算損失值於這筆資料上
        loss_value = loss(y_hat, y)
        # 反向傳播
        loss_value.backward()
        # 隨機梯度下降(每一筆資料我都更新一次參數) Stochastic Gradient Descent
        optimizer.step()
        # 把 loss 值記起來 特別注意這裡要放 .item(), 不然 computational graph 不會被釋放掉, 會佔用很多記憶體
        loss_value_list.append(loss_value.item())
    print("Iteration: {}, Mean loss: {}".format(i, np.mean(loss_value_list)))

# 用沒看過的測試集預測看看
loss_value_list = []
for i in range(1000):
    x = torch.from_numpy(np.array(mnist_testset[i][0])).float()
    y = torch.from_numpy(np.array(mnist_testset[i][1])).long()
    x = x.view(1, 784)
    y = y.view(1)
    y_hat = net(x)
    loss_value = loss(y_hat, y)
    loss_value_list.append(loss_value.item())

print("Predict first 1000 image's mean loss: {}".format(np.mean(loss_value_list)))
    
# 實際跟你的模型互動看看吧！
while True:
    index = input("選一道測試集裡面的題目吧！(0 ~ {})\n".format(len(mnist_testset)-1))
    index = int(index)
    image = mnist_testset[index][0]
    image.show()
    x = torch.from_numpy(np.array(image)).float()
    y = mnist_testset[index][1]
    x = x.view(1, 784)
    y_hat = net.predict(x)
    predict_number = torch.argmax(y_hat, dim=1)
    print(predict_number)
    print("你的神經網路預測的機率分布是: {}".format(y_hat.detach().numpy()))
    print("正確答案是數字: {}, 而機器的預測值是: {}".format(y, predict_number[0].item()))


