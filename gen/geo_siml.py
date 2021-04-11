import torch
from torch.nn import Softmax

def compute_geo_siml(att_size):
    sm = Softmax(dim=-1)
    geo_diff = torch.zeros(att_size, att_size)
    for i, row in enumerate(geo_diff):
        for j, element in enumerate(row):
            x1, y1 = i // att_size, i % att_size
            x2, y2 = j // att_size, j % att_size
            geo_diff[i][j] = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 + 1    # Euclidean Distance
        if (i+1) % 600 == 0:
            print('%4d/3600 done' % (i+1))
    # geo_siml = (1 / geo_diff).expand(batch_size, att_size, att_size)
    geo_siml = sm(1 / geo_diff)
    torch.save({'geo_siml': geo_siml}, 'gs.pth')

compute_geo_siml(3600)
