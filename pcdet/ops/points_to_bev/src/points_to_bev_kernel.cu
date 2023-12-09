#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int floatToOrderedInt(float floatVal) {
    int intVal = __float_as_int(floatVal);
    return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__global__ void pointsToBEVKernel(
    const float* points, 
    int num_points, 
    float x_min, float y_min,
    float x_scale, float y_scale,
    int bev_width, int bev_height,
    float* bev_zmax,  
    float* bev_intensity_max  
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = points[idx * 4];
    float y = points[idx * 4 + 1];
    float z = points[idx * 4 + 2];
    float intensity = points[idx * 4 + 3];

    int px = static_cast<int>((x - x_min) * x_scale);
    int py = static_cast<int>((y - y_min) * y_scale);

    if (px >= 0 && px < bev_width && py >= 0 && py < bev_height) {
        int pixel_index = py * bev_width + px;

        unsigned int* bev_zmax_as_uint = (unsigned int*)bev_zmax;
        atomicMax(bev_zmax_as_uint + pixel_index, floatToOrderedInt(z));

        unsigned int* bev_intensity_max_as_uint = (unsigned int*)bev_intensity_max;
        atomicMax(bev_intensity_max_as_uint + pixel_index, floatToOrderedInt(intensity));
    }
}

extern "C" void launchPointsToBEVKernel(
    const float* points, 
    int num_points, 
    float x_min, float y_min,
    float x_scale, float y_scale,
    int bev_width, int bev_height,
    float* bev_zmax,  
    float* bev_intensity_max  
) {
    int threads = 1024;
    int blocks = (num_points + threads - 1) / threads;
    pointsToBEVKernel<<<blocks, threads>>>(
        points,
        num_points,
        x_min, y_min,
        x_scale, y_scale,
        bev_width, bev_height,
        bev_zmax,
        bev_intensity_max
    );
}
