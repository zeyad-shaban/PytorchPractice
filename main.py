import torch

points = torch.tensor([[4, 1], [5, 3], [2, 1]])
points_t = points.t().contiguous()

print(points_t.stride())

print(points.is_contiguous())
print(points_t.is_contiguous())