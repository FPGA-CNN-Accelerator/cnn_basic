#include "cnn_types.h"
#include "cnn_core.h"
#include "training.h"
#include "model_io.h"
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
        int epochs = EPOCHS;
        float learning_rate = LEARNING_RATE;
            
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