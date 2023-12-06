#define GROUP_SIZE 256

__kernel void merge(__global const float* a, __global float* res, unsigned int length) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int beginning_of_even_array = gid - gid % (length * 2);
    int beginning_of_odd_array = beginning_of_even_array + length;
    int index_in_array = gid % length;
    int group_beginning = get_local_size(0) * get_group_id(0);
    int index_in_res;

    __local float input_buffer[GROUP_SIZE];
    input_buffer[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);


    if (gid < beginning_of_odd_array) { // элемент находится в четном массиве
        int left = -1, right = length;
        while (left + 1 != right) {
            int middle = (left + right) >> 1;
            float cur_element = GROUP_SIZE > length ? input_buffer[beginning_of_odd_array + middle - group_beginning] : a[beginning_of_odd_array + middle];
            // пусть среди равных элементов в результате сначала
            // идут элементы из четного массива, поэтому знак меньше
            if (cur_element < input_buffer[lid]) {
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
            float cur_element = GROUP_SIZE > length ? input_buffer[beginning_of_even_array + middle - group_beginning] : a[beginning_of_even_array + middle];
            // тут знак меньше либо равно, так как среди равных
            // элементов в результате сначала идут элементы из
            // четного массива
            if (cur_element <= input_buffer[lid]) {
                left = middle;
            } else {
                right = middle;
            }
        }
        index_in_res = beginning_of_even_array + right + index_in_array;
    }
    res[index_in_res] = GROUP_SIZE > length ? input_buffer[lid] : a[gid];
}


int get_intersection_with_merge_path(__global const float* a,
                                     int first_begin,
                                     int first_length,
                                     int second_begin,
                                     int second_length,
                                     int sum_count) {
    int left = max(0, sum_count - second_length) - 1, right = min(sum_count, first_length);
    while (left + 1 != right) {
        int middle = (left + right) >> 1;
        if (sum_count > middle && a[first_begin + middle] <= a[second_begin + sum_count - middle - 1]) {
            left = middle;
        } else {
            right = middle;
        }
    }
    return right;
}

int get_intersection_with_merge_path_with_buffers(__local float* first,
                                                 int first_length,
                                                 __local float* second,
                                                 int second_length,
                                                 int sum_count) {
    int left = max(0, sum_count - second_length) - 1, right = min(sum_count, first_length);
    while (left + 1 != right) {
        int middle = (left + right) >> 1;
        if (sum_count > middle && first[middle] <= second[sum_count - middle - 1]) {
            left = middle;
        } else {
            right = middle;
        }
    }
    return right;
}

__kernel void fast_merge(__global const float* a, __global float* res, unsigned int length) {
    if (GROUP_SIZE > length) {
        merge(a, res, length);
        return;
    }

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int beginning_of_even_array = gid - gid % (length * 2);
    int beginning_of_odd_array = beginning_of_even_array + length;
    int group_beginning = get_local_size(0) * get_group_id(0);
    int index_in_res;

    __local int even_array_buffer_begin, even_array_buffer_end;
    __local int odd_array_buffer_begin, odd_array_buffer_end;
    __local float input_buffer[GROUP_SIZE];
    __local float output_buffer[GROUP_SIZE];
    if (lid == 0) {
        int sum_count = group_beginning - beginning_of_even_array;
        int number_of_elements_in_even_array = get_intersection_with_merge_path(a, beginning_of_even_array, length, beginning_of_odd_array, length, sum_count);
        even_array_buffer_begin = beginning_of_even_array + number_of_elements_in_even_array;
        odd_array_buffer_begin = beginning_of_odd_array + sum_count - number_of_elements_in_even_array;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int sum_count = GROUP_SIZE;
    __local float first[GROUP_SIZE], second[GROUP_SIZE];
    if (lid < beginning_of_even_array + length - even_array_buffer_begin) {
        first[lid] = a[even_array_buffer_begin + lid];
    }
    if (lid < beginning_of_odd_array + length - odd_array_buffer_begin) {
        second[lid] = a[odd_array_buffer_begin + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int number_of_elements_in_even_array = get_intersection_with_merge_path_with_buffers(first, length - (even_array_buffer_begin - beginning_of_even_array),
                                                                                        second, length - (odd_array_buffer_begin - beginning_of_odd_array), sum_count);
        even_array_buffer_end = even_array_buffer_begin + number_of_elements_in_even_array;
        odd_array_buffer_end = odd_array_buffer_begin + sum_count - number_of_elements_in_even_array;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int even_array_buffer_length = even_array_buffer_end - even_array_buffer_begin;
    int odd_array_buffer_length = odd_array_buffer_end - odd_array_buffer_begin;

    if (lid < even_array_buffer_length) {
        input_buffer[lid] = a[even_array_buffer_begin + lid];
    } else {
        input_buffer[lid] = a[odd_array_buffer_begin + lid - even_array_buffer_length];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < even_array_buffer_length) {
        int left = -1, right = odd_array_buffer_length;
        while (left + 1 != right) {
            int middle = (left + right) >> 1;
            // пусть среди равных элементов в результате сначала
            // идут элементы из четного массива, поэтому знак меньше
            if (input_buffer[even_array_buffer_length + middle] < input_buffer[lid]) {
                left = middle;
            } else {
                right = middle;
            }
        }
        index_in_res = lid + right;
    } else {
        int left = -1, right = even_array_buffer_length;
        while (left + 1 != right) {
            int middle = (left + right) >> 1;
            // тут знак меньше либо равно, так как среди равных
            // элементов в результате сначала идут элементы из
            // четного массива
            if (input_buffer[middle] <= input_buffer[lid]) {
                left = middle;
            } else {
                right = middle;
            }
        }
        index_in_res = right + lid - even_array_buffer_length;
    }
    output_buffer[index_in_res] = input_buffer[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    res[gid] = output_buffer[lid];
}