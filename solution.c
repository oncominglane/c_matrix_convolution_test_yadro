#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t height;
    uint32_t width;
    uint16_t kernel_height;
    uint16_t kernel_width;
    uint8_t *pixels; // interleaved matrix elements: A1, B1, C1, A2, B2, C2, ...
    int8_t *kernel;
} Image;

static int read_input(FILE *file, Image *image)
{
    if (fread(&image->height, sizeof(image->height), 1, file) != 1 ||
        fread(&image->width, sizeof(image->width), 1, file) != 1 ||
        image->height == 0 || image->width == 0) {
        return 0;
    }

    if ((uint64_t)image->height * image->width > SIZE_MAX / 3) {
        return 0;
    }
    size_t pixel_bytes = (size_t)image->height * image->width * 3;
    image->pixels = malloc(pixel_bytes);
    if (image->pixels == NULL ||
        fread(image->pixels, 1, pixel_bytes, file) != pixel_bytes) {
        return 0;
    }

    if (fread(&image->kernel_height, sizeof(image->kernel_height), 1, file) != 1 ||
        fread(&image->kernel_width, sizeof(image->kernel_width), 1, file) != 1 ||
        image->kernel_height % 2 == 0 || image->kernel_width % 2 == 0) {
        return 0;
    }

    if ((uint64_t)image->kernel_height * image->kernel_width > SIZE_MAX) {
        return 0;
    }
    size_t kernel_bytes = (size_t)image->kernel_height * image->kernel_width;
    image->kernel = malloc(kernel_bytes);
    return image->kernel != NULL &&
           fread(image->kernel, 1, kernel_bytes, file) == kernel_bytes;
}

static uint8_t normalize(int64_t value)
{
    if (value > 255) {
        return (uint8_t)(value % 251);
    }
    if (value < 0) {
        return (uint8_t)((-value) % 241);
    }
    return (uint8_t)value;
}

static int convolve_and_write(FILE *file, const Image *image)
{
    size_t row_bytes = (size_t)image->width * 3;
    uint8_t *row = malloc(row_bytes);
    if (row == NULL) {
        return 0;
    }

    int ok = fwrite(&image->height, sizeof(image->height), 1, file) == 1 &&
             fwrite(&image->width, sizeof(image->width), 1, file) == 1;

    int64_t half_height = image->kernel_height / 2;
    int64_t half_width = image->kernel_width / 2;
    for (uint32_t y = 0; ok && y < image->height; ++y) {
        int64_t first_y = (int64_t)y - half_height;
        int64_t last_y = (int64_t)y + half_height;
        if (first_y < 0) {
            first_y = 0;
        }
        if (last_y >= image->height) {
            last_y = (int64_t)image->height - 1;
        }

        for (uint32_t x = 0; x < image->width; ++x) {
            int64_t first_x = (int64_t)x - half_width;
            int64_t last_x = (int64_t)x + half_width;
            if (first_x < 0) {
                first_x = 0;
            }
            if (last_x >= image->width) {
                last_x = (int64_t)image->width - 1;
            }

            int64_t sums[3] = {0, 0, 0};
            // skip out-of-bounds cells: they contribute zero to the sum
            for (int64_t source_y = first_y; source_y <= last_y; ++source_y) {
                size_t kernel_y = (size_t)(source_y - y + half_height);
                for (int64_t source_x = first_x; source_x <= last_x; ++source_x) {
                    size_t kernel_x = (size_t)(source_x - x + half_width);
                    int64_t coefficient = image->kernel[
                        kernel_y * image->kernel_width + kernel_x];
                    size_t offset = ((size_t)source_y * image->width +
                                     (size_t)source_x) * 3;
                    for (size_t channel = 0; channel < 3; ++channel) {
                        sums[channel] += image->pixels[offset + channel] *
                                         coefficient;
                    }
                }
            }
            for (size_t channel = 0; channel < 3; ++channel) {
                row[(size_t)x * 3 + channel] = normalize(sums[channel]);
            }
        }
        ok = fwrite(row, 1, row_bytes, file) == row_bytes;
    }

    free(row);
    return ok;
}

int main(int argc, char **argv)
{
    const char *input_name = NULL;
    const char *output_name = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "-o") == 0) {
            int is_input = strcmp(argv[i], "-i") == 0;
            if (i + 1 >= argc || argv[i + 1][0] == '\0' ||
                strcmp(argv[i + 1], "-i") == 0 ||
                strcmp(argv[i + 1], "-o") == 0) {
                fprintf(stderr, "Missing file name.\n");
                return 10;
            }
            if (is_input) {
                input_name = argv[++i];
            } else {
                output_name = argv[++i];
            }
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 10;
        }
    }
    if (input_name == NULL || output_name == NULL) {
        fprintf(stderr, "Usage: %s -i input.bin -o output.bin\n", argv[0]);
        return 10;
    }

    FILE *input = fopen(input_name, "rb");
    if (input == NULL) {
        perror("Cannot open input file");
        return 1;
    }

    Image image = {0};
    int ok = read_input(input, &image);
    if (fclose(input) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "Cannot read input data or allocate memory.\n");
        free(image.pixels);
        free(image.kernel);
        return 1;
    }

    FILE *output = fopen(output_name, "wb");
    if (output == NULL) {
        perror("Cannot open output file");
        ok = 0;
    } else {
        ok = convolve_and_write(output, &image);
        if (fclose(output) != 0) {
            ok = 0;
        }
        if (!ok) {
            fprintf(stderr, "Cannot write result or allocate memory.\n");
        }
    }

    free(image.pixels);
    free(image.kernel);
    return ok ? 0 : 1;
}
