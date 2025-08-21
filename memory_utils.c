#include "memory_utils.h"

float ***create_3d_array(int depth, int height, int width) {
    float ***array = (float ***)malloc(depth * sizeof(float **));
    for (int d = 0; d < depth; d++) {
        array[d] = (float **)malloc(height * sizeof(float *));
        for (int h = 0; h < height; h++) {
            array[d][h] = (float *)malloc(width * sizeof(float));
        }
    }
    return array;
}

// free 3d array
// c언어... 가비지컬렉터같은 메모리 해제해주는 무언가를 컴파일러쪽에서 추가해줬으면
void free_3d_array(float ***array, int depth, int height) {
    for (int d = 0; d < depth; d++) {
        for (int h = 0; h < height; h++) {
            free(array[d][h]);
        }
        free(array[d]);
    }
    free(array);
}

float **create_2d_array(int rows, int cols) {
    float **array = (float **)malloc(rows * sizeof(float *));
    for (int r = 0; r < rows; r++) {
        array[r] = (float *)malloc(cols * sizeof(float));
    }
    return array;
}

void free_2d_array(float **array, int rows) {
    for (int r = 0; r < rows; r++) {
        free(array[r]);
    }
    free(array);
}

float *create_1d_array(int size) {
    return (float *)malloc(size * sizeof(float));
}

void free_1d_array(float *array) {
    free(array);
}
