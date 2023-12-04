__kernel void merge(__global const float* a, __global float* res, unsigned int length) {
    unsigned int gid = get_global_id(0);
    unsigned int beginning_of_even_array = gid - gid % (length * 2);
    unsigned int beginning_of_odd_array = beginning_of_even_array + length;
    unsigned int index_in_array = gid % length;
    unsigned int index_in_res;
    if (gid < beginning_of_odd_array) { // элемент находится в четном массиве
        int left = -1, right = length;
        while (left + 1 != right) {
            int middle = (left + right) >> 1;
            // пусть среди равных элементов в результате сначала
            // идут элементы из четного массива, поэтому знак меньше
            if (a[beginning_of_odd_array + middle] < a[gid]) {
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
            // тут знак меньше либо равно, так как среди равных
            // элементов в результате сначала идут элементы из
            // четного массива
            if (a[beginning_of_even_array + middle] <= a[gid]) {
                left = middle;
            } else {
                right = middle;
            }
        }
        index_in_res = beginning_of_even_array + right + index_in_array;
    }
    res[index_in_res] = a[gid];
}