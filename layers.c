#include "layers.h"

float ***conv_forward(float ***input, conv_layer_t *layer) {
    float ***output = create_3d_array(layer->num_filters, layer->output_height, layer->output_width);
    
    // bias로 초기화
    for (int f = 0; f < layer->num_filters; f++) {
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                output[f][h][w] = layer->biases[f];
            }
        }
    }
    
    // 컨볼루션 연산
    for (int f = 0; f < layer->num_filters; f++) {
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                for (int c = 0; c < layer->input_channels; c++) {
                    for (int kh = 0; kh < layer->filter_size; kh++) {
                        for (int kw = 0; kw < layer->filter_size; kw++) {
                            int input_h = h * layer->stride + kh - layer->padding;
                            int input_w = w * layer->stride + kw - layer->padding;
                            
                            if (input_h >= 0 && input_h < layer->input_height &&
                                input_w >= 0 && input_w < layer->input_width) {
                                output[f][h][w] += input[c][input_h][input_w] * 
                                                   layer->filters[f][c][kh][kw];
                            }
                        }
                    }
                }
                
                // 렐루 activation func 적용.. // 렐루 어감 되게 귀엽지 않나요 렐루~
                output[f][h][w] = relu(output[f][h][w]);
            }
        }
    }
    
    return output;
}

// pooling layer forward (max pooling)
float ***pool_forward(float ***input, pool_layer_t *layer) {
    float ***output = create_3d_array(layer->input_channels, layer->output_height, layer->output_width);
    
    for (int c = 0; c < layer->input_channels; c++) {
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                float max_val = -INFINITY;
                
                // find max value on? in? pooling window
                for (int ph = 0; ph < layer->pool_size; ph++) {
                    for (int pw = 0; pw < layer->pool_size; pw++) {
                        int input_h = h * layer->stride + ph;
                        int input_w = w * layer->stride + pw;
                        
                        if (input_h < layer->input_height && input_w < layer->input_width) {
                            if (input[c][input_h][input_w] > max_val) {
                                max_val = input[c][input_h][input_w];
                            }
                        }
                    }
                }
                
                output[c][h][w] = max_val;
            }
        }
    }
    
    return output;
}

float *fc_forward(float *input, fc_layer_t *layer) {
    float *output = create_1d_array(layer->output_size);
    
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            output[i] += input[j] * layer->weights[i][j];
        }
    }
    
    return output;
}

// 컨볼루션 레이어 역전파
// 각 channel, height, width 별로 gradient calculate
// 출력 특성맵의 모든 위치에서 gradient 수집해서 update
void backprop_conv(float ***gradient, conv_layer_t *layer, float ***input, float learning_rate) {
    // 필터 가중치 업데이트
    // 출력 너비(출력 높이(필터 너비(필터 높이(입력 채널(필터 번호)))))
    for (int f = 0; f < layer->num_filters; f++) {
        for (int c = 0; c < layer->input_channels; c++) {
            for (int h = 0; h < layer->filter_size; h++) {
                for (int w = 0; w < layer->filter_size; w++) {
                    float grad_sum = 0.0f;
                    for (int oh = 0; oh < layer->output_height; oh++) {
                        for (int ow = 0; ow < layer->output_width; ow++) {
                            int input_h = oh * layer->stride + h - layer->padding;
                            int input_w = ow * layer->stride + w - layer->padding;
                            if (input_h >= 0 && input_h < layer->input_height &&
                                input_w >= 0 && input_w < layer->input_width) {
                                grad_sum += gradient[f][oh][ow] * input[c][input_h][input_w];
                            }
                        }
                    }
                    layer->filters[f][c][h][w] -= learning_rate * grad_sum;
                }
            }
        }

        // update bias
        float bias_grad = 0.0f;
        for (int h = 0; h < layer->output_height; h++) {
            for (int w = 0; w < layer->output_width; w++) {
                bias_grad += gradient[f][h][w];
            }
        }
        layer->biases[f] -= learning_rate * bias_grad;
    }
}

// fully connected layer backprop
// 이런저런 기법 없이 그냥 단순히 미분 계산 해서 업데이트 하도록
void backprop_fc(float *gradient, fc_layer_t *layer, float *input, float learning_rate) {
    // update weights
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            layer->weights[i][j] -= learning_rate * gradient[i] * input[j];
        }

        // update bias
        layer->biases[i] -= learning_rate * gradient[i];
    }
}
