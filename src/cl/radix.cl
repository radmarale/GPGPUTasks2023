#define TRANSPOSE_WORKGROUP_SIZE 16
#define GROUP_SIZE 256

unsigned int get_bits(unsigned int x, unsigned int step, unsigned int number_of_bits) {
    return (x >> (step * number_of_bits)) & ((1 << number_of_bits) - 1);
}

__kernel void fill_with_zeros(__global unsigned int *as, unsigned int n) {
    int gid = get_global_id(0);
    if (gid > n) {
        return;
    }
    as[gid] = 0;
}

__kernel void fill_counters(__global const unsigned int *as,
                            __global unsigned int* counters,
                            unsigned int step,
                            unsigned int number_of_bits) {
    int gid = get_global_id(0);
    int wid = get_group_id(0);

    int counters_index = wid * (1 << number_of_bits) + get_bits(as[gid], step, number_of_bits);
    atomic_add(counters + counters_index, 1);
}

__kernel void radix(__global const unsigned int* as,
                    __global unsigned int* counters,
                    __global unsigned int* work_group_prefix_sum,
                    __global unsigned int* res,
                    unsigned int step,
                    unsigned int number_of_bits) {
    int gid = get_global_id(0);
    int wid = get_group_id(0);
    int lid = get_local_id(0);
    int bits = get_bits(as[gid], step, number_of_bits);
    int number_of_lower_elements_in_group = bits ? work_group_prefix_sum[wid * (1 << number_of_bits) + bits - 1] : 0;
    int counters_index = bits * get_num_groups(0) + wid - 1;
    int number_of_lower_elements_in_array = counters_index >= 0 ? counters[counters_index] : 0;
    int index_in_res = number_of_lower_elements_in_array + lid - number_of_lower_elements_in_group;
    res[index_in_res] = as[gid];
}

void shift(unsigned int* i, unsigned int* j) {
    *j = (*j + *i) % TRANSPOSE_WORKGROUP_SIZE;
}

__kernel void matrix_transpose(__global const unsigned int* a,
                               __global unsigned int* aT,
                               unsigned int M, unsigned int K) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int tile_i = get_group_id(0) * get_local_size(0);
    unsigned int tile_j = get_group_id(1) * get_local_size(1);

    __local unsigned int buffer[TRANSPOSE_WORKGROUP_SIZE][TRANSPOSE_WORKGROUP_SIZE];

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

__kernel void prefix_sum(__global unsigned int* as, int n, int loglength, int stage) {
    int gid = get_global_id(0);
    if (gid >= (n >> loglength)) {
        return;
    }
    int as_index = (gid << loglength) + (1 << loglength) - 1;
    if (stage == 0) {
        as[as_index] += as[as_index - (1 << (loglength - 1))];
    } else {
        if (as_index + 1 < n) {
            as[as_index + (1 << (loglength - 1))] += as[as_index];
        }
    }
}


__kernel void prefix_sum_on_many_segments(__global unsigned int* as,
                                          int n, int loglength, int segment_length,
                                          int stage) {
    int gid = get_global_id(0);
    if (gid >= (n >> loglength)) {
        return;
    }
    int as_index = (gid << loglength) + (1 << loglength) - 1;
    if (stage == 0) {
        as[as_index] += as[as_index - (1 << (loglength - 1))];
    } else {
        if (as_index % segment_length + 1 < segment_length) {
            as[as_index + (1 << (loglength - 1))] += as[as_index];
        }
    }
}


__kernel void merge(__global const unsigned int* a,
                    __global unsigned int* res,
                    unsigned int length,
                    unsigned int step,
                    unsigned int number_of_bits) {
    // length < GROUP_SIZE
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int beginning_of_even_array = gid - gid % (length * 2);
    int beginning_of_odd_array = beginning_of_even_array + length;
    int index_in_array = gid % length;
    int group_beginning = get_local_size(0) * get_group_id(0);
    int index_in_res;

    __local unsigned int input_buffer[GROUP_SIZE];
    input_buffer[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);


    if (gid < beginning_of_odd_array) { // элемент находится в четном массиве
        int left = -1, right = length;
        while (left + 1 != right) {
            int middle = (left + right) >> 1;
            unsigned int cur_element = get_bits(input_buffer[beginning_of_odd_array + middle - group_beginning],
                                         step, number_of_bits);
            // пусть среди равных элементов в результате сначала
            // идут элементы из четного массива, поэтому знак меньше
            if (cur_element < get_bits(input_buffer[lid], step, number_of_bits)) {
                left = middle;
            } else {
                right = middle;
            }
        }
        index_in_res = gid + right;
    } else { // элемент находится в нечетном массиве
        int left = -1, right = length;
        while (left + 1 != right) {
            int middle = (left + right) >> 1;
            unsigned int cur_element =
                get_bits(input_buffer[beginning_of_even_array + middle - group_beginning], step, number_of_bits);
            // тут знак меньше либо равно, так как среди равных
            // элементов в результате сначала идут элементы из
            // четного массива
            if (cur_element <= get_bits(input_buffer[lid], step, number_of_bits)) {
                left = middle;
            } else {
                right = middle;
            }
        }
        index_in_res = beginning_of_even_array + right + index_in_array;
    }
    res[index_in_res] = input_buffer[lid];
}
