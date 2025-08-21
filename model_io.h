#ifndef MODEL_IO_H
#define MODEL_IO_H

#include "cnn_types.h"
#include <stdio.h>

/**
 * 모델 직렬화/역직렬화
 */

void save_model(cnn_t *cnn, const char *filename);

void load_model(cnn_t *cnn, const char *filename);

#endif // MODEL_IO_H
