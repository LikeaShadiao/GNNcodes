# 通过随机函数构造的适合于geometric中data格式的数据集
import torch
from torch_geometric.data import Data

num_nodes = 10
num_features = 5
num_classes = 3
num_edges = 20

x = torch.randn(num_nodes, num_features)
y = torch.randint(num_classes, (num_nodes,))
edges_index = torch.randint(num_nodes, (2, num_edges))

data = Data(x=x, y=y, edges_index=edges_index)

print(data)
print(data.x)
print(data.edges_index)
