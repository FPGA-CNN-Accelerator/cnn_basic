#include "image_processing.h"

// jpeg img load & convert to array
// rgb -> greyscale transform 후 to normalize 0-1
float ***load_jpg_to_array(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;
    
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", filename);
        return NULL;
    }
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    
    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    float ***image = create_3d_array(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
    
    // init image array
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
        for (int h = 0; h < IMAGE_HEIGHT; h++) {
            for (int w = 0; w < IMAGE_WIDTH; w++) {
                image[c][h][w] = 0.0f;
            }
        }
    }
    
    // convert jpeg data to array
    int y = 0;
    while (cinfo.output_scanline < cinfo.output_height && y < IMAGE_HEIGHT) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        
        for (int x = 0; x < (int)cinfo.output_width && x < IMAGE_WIDTH; x++) {
            float gray = 0.0f;
            // rgb -> greyscale transform
            for (int c = 0; c < cinfo.output_components; c++) {
                gray += buffer[0][x * cinfo.output_components + c];
            }
            // normalize 0-1 // 255 그대로 쓰는지 확인 필요 // 저번에 논문에서 12bit 어쩌구 하지 않았나
            gray /= (255.0f * cinfo.output_components);
            
            image[0][y][x] = gray;
        }
        y++;
    }
    
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return image;
}

void normalize_image(float ***image) {
    (void)image;
    // load_jpg_to_array()에서 이미 정규화됨 -> 레거시 코드가 되어버린..
}

void free_image(float ***image) {
    free_3d_array(image, IMAGE_CHANNELS, IMAGE_HEIGHT);
}
