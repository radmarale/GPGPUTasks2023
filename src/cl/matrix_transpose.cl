#define WORKGROUP_SIZE 16

void shift(unsigned int* i, unsigned int* j) {
    *j = (*j + *i) % WORKGROUP_SIZE;
}

__kernel void matrix_transpose(__global const float* a,
                               __global float* aT,
                               unsigned int M, unsigned int K) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int tile_i = get_group_id(0) * get_local_size(0);
    unsigned int tile_j = get_group_id(1) * get_local_size(1);

    __local float buffer[WORKGROUP_SIZE][WORKGROUP_SIZE];

    int index = i * K + j;
    unsigned int buffer_i = local_i;
    unsigned int buffer_j = local_j;
    shift(&buffer_i, &buffer_j);

    if (i < M && j < K) {
        buffer[buffer_i][buffer_j] = a[index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    buffer_i = local_j;
    buffer_j = local_i;
    shift(&buffer_i, &buffer_j);

    unsigned int aT_i = tile_j + local_i;
    unsigned int aT_j = tile_i + local_j;
    if (aT_i < K && aT_j < M) {
        aT[aT_i * M + aT_j] = buffer[buffer_i][buffer_j];
    }
}