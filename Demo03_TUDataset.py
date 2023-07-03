from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
data = dataset[0]

dataset = dataset.shuffle()
print(dataset)

train_dataset = dataset[:540]
test_dataset = dataset[540:]




