#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 128

__kernel void sum_atomic_add(__global const unsigned int* a,
                             __global unsigned int* sum,
                             int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, a[index]);
}

__kernel void sum_fewer_atoimic_adds(__global const unsigned int* a,
                                    __global unsigned int* sum,
                                    int n)
{
    unsigned int res = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = get_global_id(0) * VALUES_PER_WORKITEM + i;
        if (index >= n) {
            break;
        }
        res += a[index];
    }

    atomic_add(sum, res);
}

__kernel void sum_fewer_atoimic_adds_coalesced(__global const unsigned int* a,
                                              __global unsigned int* sum,
                                              int n)
{
    unsigned int res = 0;
    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);
    unsigned int grs = get_local_size(0);

    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = wid * grs * VALUES_PER_WORKITEM + lid + i * grs;
        if (index >= n) {
            break;
        }
        res += a[index];
    }

    atomic_add(sum, res);
}

__kernel void sum_local_buffer(__global const unsigned int* a,
                         __global unsigned int* sum,
                         int n)
{
    unsigned int group_res = 0;
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];
    
    if (gid < n) {
        buffer[lid] = a[gid];
    } else {
        buffer[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buffer[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void sum_tree(__global const unsigned int* a,
                       __global unsigned int* sum,
                       int n)
{
    unsigned int group_res = 0;
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];
    
    if (gid < n) {
        buffer[lid] = a[gid];
    } else {
        buffer[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues >>= 1) {
        if (2 * lid < nValues) {
            buffer[lid] += buffer[lid + (nValues >> 1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buffer[0]);
    }
}