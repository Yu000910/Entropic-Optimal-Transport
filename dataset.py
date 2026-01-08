import numpy as np

path = '/Users/nnn/Desktop/temp/博士毕业/第五篇/verb-tool-project/datasets/D_BEDR.npz'
data = np.load(path, allow_pickle=True)

print("Keys:", list(data.keys()))

# 检查 labels 到底存了什么
labels = data['labels']
print(f"Labels 类型: {type(labels)}")
print(f"Labels 元素类型: {type(labels[0])}")
print(f"前 10 个 Label: {labels[:10]}")

# 检查是否有隐藏的映射 Key
for key in data.keys():
    if key not in ['vectors', 'texts', 'labels']:
        print(f"\n发现潜在的映射 Key: {key}")
        print(data[key])