#include "cnn.h"
#include <time.h>

int main() {
    srand(time(NULL));
    cnn_t cnn;
    init_cnn(&cnn);
    
    const char *model_filename = "trained_cnn_model.bin";
    
    printf("load model from %s\n", model_filename);
    FILE *fp = fopen(model_filename, "rb");
    if (fp) {
        fclose(fp);
        load_model(&cnn, model_filename);
        printf("model loaded\n");
    } else {
        // 이친구들도헤더파일로분리할까하다가너무나눠지는느낌이라하드코딩을
        // 실제 학습 할 때는 에폭 넉넉하게 + 러닝레이트 반토막
        int epochs = 20;
        float learning_rate = 0.005;
            
        printf("epoch: %d | learning rate: %.4f\n", epochs, learning_rate);
        
        printf("training start\n");
        train_cnn(&cnn, "./trainset", epochs, learning_rate);
        printf("done!\n");

        printf("save model\n");
        save_model(&cnn, model_filename);
    }
    
    printf("validation start\n");
    float accuracy = validate_cnn(&cnn, "./testset");
    printf("accuracy: %.4f\n", accuracy);
    
    free_cnn(&cnn);
    
    printf("lgtm!");
    return 0;
} 