#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <stdio.h>
#include <stdlib.h>

/**
 * 메모리 관리해주는 무언가
*/

float ***create_3d_array(int depth, int height, int width);
void free_3d_array(float ***array, int depth, int height);

float **create_2d_array(int rows, int cols);
void free_2d_array(float **array, int rows);

float *create_1d_array(int size);
void free_1d_array(float *array);

#endif // MEMORY_UTILS_H
