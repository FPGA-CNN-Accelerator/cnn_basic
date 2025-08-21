#include "model_io.h"

void save_model(cnn_t *cnn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("!fp: %s\n", filename);
        return;
    }

    // save conv1 layer
    fwrite(&cnn->conv1.num_filters, sizeof(int), 1, fp);
    fwrite(&cnn->conv1.input_channels, sizeof(int), 1, fp);
    fwrite(&cnn->conv1.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv1.num_filters; f++) {
        for (int c = 0; c < cnn->conv1.input_channels; c++) {
            for (int h = 0; h < cnn->conv1.filter_size; h++) {
                fwrite(cnn->conv1.filters[f][c][h], sizeof(float), cnn->conv1.filter_size, fp);
            }
        }
    }
    fwrite(cnn->conv1.biases, sizeof(float), cnn->conv1.num_filters, fp);

    // save conv2 layer
    fwrite(&cnn->conv2.num_filters, sizeof(int), 1, fp);
    fwrite(&cnn->conv2.input_channels, sizeof(int), 1, fp);
    fwrite(&cnn->conv2.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv2.num_filters; f++) {
        for (int c = 0; c < cnn->conv2.input_channels; c++) {
            for (int h = 0; h < cnn->conv2.filter_size; h++) {
                fwrite(cnn->conv2.filters[f][c][h], sizeof(float), cnn->conv2.filter_size, fp);
            }
        }
    }
    fwrite(cnn->conv2.biases, sizeof(float), cnn->conv2.num_filters, fp);

    // save fc1 layer
    fwrite(&cnn->fc1.input_size, sizeof(int), 1, fp);
    fwrite(&cnn->fc1.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc1.output_size; i++) {
        fwrite(cnn->fc1.weights[i], sizeof(float), cnn->fc1.input_size, fp);
    }
    fwrite(cnn->fc1.biases, sizeof(float), cnn->fc1.output_size, fp);

    // save fc2 layer
    fwrite(&cnn->fc2.input_size, sizeof(int), 1, fp);
    fwrite(&cnn->fc2.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc2.output_size; i++) {
        fwrite(cnn->fc2.weights[i], sizeof(float), cnn->fc2.input_size, fp);
    }
    fwrite(cnn->fc2.biases, sizeof(float), cnn->fc2.output_size, fp);

    fclose(fp);
    printf("saved! %s\n", filename);
}

void load_model(cnn_t *cnn, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("!fp: %s\n", filename);
        return;
    }

    // load conv1 layer
    fread(&cnn->conv1.num_filters, sizeof(int), 1, fp);
    fread(&cnn->conv1.input_channels, sizeof(int), 1, fp);
    fread(&cnn->conv1.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv1.num_filters; f++) {
        for (int c = 0; c < cnn->conv1.input_channels; c++) {
            for (int h = 0; h < cnn->conv1.filter_size; h++) {
                fread(cnn->conv1.filters[f][c][h], sizeof(float), cnn->conv1.filter_size, fp);
            }
        }
    }
    fread(cnn->conv1.biases, sizeof(float), cnn->conv1.num_filters, fp);

    // load conv2 layer
    fread(&cnn->conv2.num_filters, sizeof(int), 1, fp);
    fread(&cnn->conv2.input_channels, sizeof(int), 1, fp);
    fread(&cnn->conv2.filter_size, sizeof(int), 1, fp);
    for (int f = 0; f < cnn->conv2.num_filters; f++) {
        for (int c = 0; c < cnn->conv2.input_channels; c++) {
            for (int h = 0; h < cnn->conv2.filter_size; h++) {
                fread(cnn->conv2.filters[f][c][h], sizeof(float), cnn->conv2.filter_size, fp);
            }
        }
    }
    fread(cnn->conv2.biases, sizeof(float), cnn->conv2.num_filters, fp);

    // load fc1 layer
    fread(&cnn->fc1.input_size, sizeof(int), 1, fp);
    fread(&cnn->fc1.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc1.output_size; i++) {
        fread(cnn->fc1.weights[i], sizeof(float), cnn->fc1.input_size, fp);
    }
    fread(cnn->fc1.biases, sizeof(float), cnn->fc1.output_size, fp);

    // load fc2 layer
    fread(&cnn->fc2.input_size, sizeof(int), 1, fp);
    fread(&cnn->fc2.output_size, sizeof(int), 1, fp);
    for (int i = 0; i < cnn->fc2.output_size; i++) {
        fread(cnn->fc2.weights[i], sizeof(float), cnn->fc2.input_size, fp);
    }
    fread(cnn->fc2.biases, sizeof(float), cnn->fc2.output_size, fp);

    fclose(fp);
    printf("loaded! %s\n", filename);
}
