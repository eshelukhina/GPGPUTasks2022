#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int bit, unsigned int n) {
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }

    unsigned int ind = global_id + 1;

    if ((ind >> bit) & 1) {
        as[global_id] += bs[(ind >> bit) - 1];
    }
}

__kernel void reduce_sum(__global unsigned int *as, __global unsigned int *cs, unsigned int n) {
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }
    cs[global_id] = as[global_id * 2] + as[global_id * 2 + 1];
}
