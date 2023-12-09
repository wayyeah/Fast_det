#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" {
    void launchPointsToBEVKernel(
        const float* points,
        int num_points,
        float x_min, float y_min,
        float x_scale, float y_scale,
        int bev_width, int bev_height,
        float* bev_zmax,
        float* bev_intensity_max
    );
}

void pointsToBEV(
    const at::Tensor points,
    at::Tensor bev_zmax,
    at::Tensor bev_intensity_max,
    float x_min, float y_min,
    float x_scale, float y_scale,
    int bev_width, int bev_height
) {
    AT_ASSERTM(points.is_cuda(), "points must be a CUDA tensor");
    AT_ASSERTM(bev_zmax.is_cuda(), "bev_zmax must be a CUDA tensor");
    AT_ASSERTM(bev_intensity_max.is_cuda(), "bev_intensity_max must be a CUDA tensor");

    int num_points = points.size(0);
    launchPointsToBEVKernel(
        points.data_ptr<float>(),
        num_points,
        x_min, y_min,
        x_scale, y_scale,
        bev_width, bev_height,
        bev_zmax.data_ptr<float>(),
        bev_intensity_max.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("points_to_bev", &pointsToBEV, "Convert points to BEV representation");
}
