#include <iostream>
#include <chrono>
#include <lz4.h>
#include <random>  // Add this header for random number generation

void cpu_compress(char* input_data, const size_t in_bytes)
{
    const size_t compressed_size = LZ4_compressBound(in_bytes);
    char* compressed_data = new char[compressed_size];

    auto start = std::chrono::high_resolution_clock::now();
    int compressed_len = LZ4_compress_default(input_data, compressed_data, in_bytes, compressed_size);
    auto stop = std::chrono::high_resolution_clock::now();

    if (compressed_len <= 0) {
        std::cerr << "CPU compression failed!" << std::endl;
        delete[] compressed_data;
        return;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU compression time: " << duration.count() << " ms\n";
    float throughput = (in_bytes / (1024.0f * 1024.0f)) / (duration.count() / 1000.0f);
    std::cout << "CPU Throughput: " << throughput << " MB/s\n";

    delete[] compressed_data;
}

int main()
{
    const size_t in_bytes = 10000000;  // 10 MB for testing
    char* uncompressed_data = new char[in_bytes];

    // Initialize random data
    std::mt19937 random_gen(42);  // Use mt19937 for random number generation
    std::uniform_int_distribution<short> uniform_dist(0, 255);  // Uniform distribution from 0 to 255
    for (size_t ix = 0; ix < in_bytes; ++ix) {
        uncompressed_data[ix] = static_cast<char>(uniform_dist(random_gen));
    }

    // Run CPU compression
    cpu_compress(uncompressed_data, in_bytes);

    delete[] uncompressed_data;

    return 0;
}
