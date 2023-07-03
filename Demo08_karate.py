# 导入kartaclub数据集并取其中的第一个为data，可视化对其分类的结果
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')



data = dataset[0]

print(data)

from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)

import Demo07_visfunc

Demo07_visfunc.coun()
Demo07_visfunc.visualize(G, color=data.y)

#不容易 终于成功了 算是个开始了



