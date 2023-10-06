import torch
import numpy as np

tensor_a = torch.tensor([1, 2, 3, 4])
tensor_b = torch.tensor([5, 6, 7, 8])

# Addition

# tensor_c = tensor_a + tensor_b
# print(tensor_add)

## Addition Longhand
print(torch.add(tensor_a, tensor_b))

## Subtraction

# tensor_c = tensor_a - tensor_b
# print(tensor_add)
print(torch.sub(tensor_a, tensor_b))

## Multiplication

# tensor_c = tensor_a * tensor_b
# print(tensor_add)
print(torch.mul(tensor_a, tensor_b))

## Division

# tensor_c = tensor_a / tensor_b
# print(tensor_c)
print(torch.div(tensor_a, tensor_b))

## Remainder Modulus

# tensor_c = tensor_b % tensor_a
# print(tensor_c)

print(torch.remainder(tensor_b, tensor_a))

## Exponents / power

print(torch.pow(tensor_a, tensor_b))

## Another way to write longhand

print(tensor_a.add_(tensor_b))
