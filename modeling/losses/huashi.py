import torch
import numpy as np

# a = torch.arange(12).reshape(3, 4)
# k = [[[0, 1, 2], [1, 0, 2]], [[0, 1, 2], [1, 0, 2]]]
# k = torch.tensor(k, dtype=torch.long)
# print(a)
# print(a[k])
# print(a[[0, 1, 2], [1, 0, 2]])

a = torch.arange(12).reshape(3, 4)
k = [True, False, True]
# k = [[True, True, True, True], [False, False, False, False], [True, True, True, True]]
print(a[torch.tensor(k)])
