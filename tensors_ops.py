import torch

my_torch = torch.arange(10)
# print(my_torch)

# Reshape and View

my_torch = my_torch.reshape(2, 5)
# print(my_torch)

# Reshape if we don´t know the number of items using -1
my_torch2 = torch.arange(10)
my_torch2 = my_torch2.reshape(-1, 5)
# print(my_torch2)


my_torch3 = torch.arange(10)
my_torch4 = my_torch3.view(5, 2)
# print(my_torch4)

# with reshape and view, they will update

my_torch5 = torch.arange(10)

my_torch6 = my_torch5.reshape(2, 5)
# print(my_torch6)

my_torch5[1] = 4141
# print(my_torch5)

# Slices
my_torch7 = torch.arange(10)
# print(my_torch7)

# Grab a specific item
# print(my_torch7[7])

# Grab slice
my_torch8 = my_torch7.reshape(5, 2)
print(my_torch8)

print(my_torch8[:, 1:])
