# 对数据集中的图数据进行节点分类任务

# 从模板中导入planetoid类，用于加载planetoid数据集
from torch_geometric.datasets import Planetoid
# 创建一个planetoid类的实例，命名为dataset，并指定数据集的根目录和名称
dataset = Planetoid(root='/tmp/Cora', name='Cora')

print(dataset[0])

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 定义一个GCN类，继承了torch.nn.Module类，是一个图卷积网络模型
class GCN(torch.nn.Module):
    # init方法中，初始化了两个GCNConv层，分别用于将节点特征从原始维度映射到16维，再从16维映射到类别数目的维度
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    # 在forward方法中，接收一个data参数，它是一个torch_geometric.data.Data对象，包含了节点特征x和边索引edge_index
    # 然后它依次调用两个GCNConv层，并在中间加入了一个relu激活函数和一个dropout层，用于增加非线性和防止过拟合
    # 最后返回一个对数概率向量，用于分类任务
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# 创建一个torch.device对象，表示要使用的设备类型，如果有可用的gpu就是用cuda，否则使用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建一个GCN模型，将它和它的参数移动到指定的设备上，可以加速计算
model = GCN().to(device)
# 从数据集中取出第一个图数据，并将它移动到指定的设备上，这样可以和模型在同一个设备上进行计算
data = dataset[0].to(device)
# 创建一个torch.optim.Adam优化器，用于更新模型的参数，它的参数包括模型的参数、学习率和权重衰减
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# 将模型设置为训练模式，可以启用一些训练时需要的功能，比如dropout等
model.train()
# 开始一个循环，每次循环表示一轮训练，共进行200轮
for epoch in range(200):
    # 将优化器中梯度清零，这样可以避免梯度累积
    optimizer.zero_grad()
    # 将数据输入模型，得到模型的输出
    out = model(data)
    # 计算损失函数，使用负对数似然损失（nll_loss），只计算训练集上的损失
    # data.train_mask是一个布尔数组，用于指定哪些节点用于训练
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # 反向传播，计算梯度
    loss.backward()
    # 更新参数，使用梯度下降法
    optimizer.step()

# 将模型设置为评估模式，可以关闭一些训练时需要的功能，例如dropout等
model.eval()
# 将数据输入模型，得到模型的输出，并取出每个节点最大概率对应的类别作为预测结果
pred = model(data).argmax(dim=1)
# 计算测试集上预测正确的节点个数，使用布尔掩码来筛选测试集上的节点，并比较预测结果和真实标签是否相等
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# 计算测试集上的准确率，使用正确个数除以总个数得到比例
acc = int(correct) / int(data.test_mask.sum())
# 打印测试集上的准确率，保留四位小数
print(f'Accuracy: {acc:.4f}')









