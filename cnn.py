import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class NetCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积：输入通道=1（灰度图），输出通道=32，卷积核大小=3x3，步幅=1，填充=1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积：输入通道=32，输出通道=64，卷积核大小=3x3，步幅=1，填充=1
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 池化层：2x2 核，步幅=2
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：输入大小=64 * 7 * 7（特征图展平后的大小），输出大小=128
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输入大小=128，输出大小=10（分类数）
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # 经过第1卷积层和池化
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))  # 经过第2卷积层和池化
        x = x.view(-1, 64 * 7 * 7)  # 展平特征图
        x = torch.nn.functional.relu(self.fc1(x))  # 全连接层1
        x = torch.nn.functional.log_softmax(self.fc2(x), dim=1)  # 全连接层2（输出层）
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = NetCNN()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0]))
        plt.subplot(1, 4, n + 1)  # 第 n+1 个子图
        plt.imshow(x[0].view(28, 28))  # 显示图像
        plt.title("Prediction: " + str(int(predict)))  # 添加标题
        plt.axis("off")  # 隐藏坐标轴
    plt.tight_layout()  # 调整布局以避免标题和图像重叠
    plt.show()


if __name__ == "__main__":
    main()
