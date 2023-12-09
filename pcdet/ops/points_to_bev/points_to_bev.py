from . import points_to_bev_extension
import torch
def points_to_bev(points, point_range, batch_size,size):
    x_scale = size[0]/ (point_range[3] - point_range[0])
    y_scale = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev_width = size[0]
    bev_height = size[1]
    bev_zmax = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))*-10
    bev_i = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    x_min=point_range[0]
    y_min=point_range[1]
    points_to_bev_extension.points_to_bev(
    points, bev_zmax, bev_i,
    x_min, y_min, x_scale, y_scale,
    bev_width, bev_height)
    return torch.cat([bev_zmax,bev_i],dim=1)