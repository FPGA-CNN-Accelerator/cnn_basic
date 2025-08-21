#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "cnn_types.h"
#include "memory_utils.h"
#include <stdio.h>
#include <jpeglib.h>

/**
 * img io & pre-processing
 * 시간 되면 jpeglib 그대로 사용하는 대신 fpga 카메라에서 주는 대로 받아서 처리할 수 있도록---
 */

// jpeg img load & convert to array
// rgb -> greyscale transform 후 to normalize 0-1
float ***load_jpg_to_array(const char *filename);

void normalize_image(float ***image);

void free_image(float ***image);

#endif // IMAGE_PROCESSING_H
