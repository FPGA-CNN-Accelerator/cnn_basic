#ifndef CNN_CORE_H
#define CNN_CORE_H

#include "cnn_types.h"
#include "memory_utils.h"
#include <stdlib.h>

/**
 * cnn core module
 * cnn structure 생성 & 해제
 * 네트워크 아키텍처 변경 시 이 모듈만 수정하면 되도록 구성
 */

// 랜덤 가중치로--
void init_cnn(cnn_t *cnn);

void free_cnn(cnn_t *cnn);

#endif // CNN_CORE_H
