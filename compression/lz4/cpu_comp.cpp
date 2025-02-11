#include <fstream>
#include <iostream>
#include <chrono>
#include <lz4.h>
#include <random>  // Add this header for random number generation

void read_file_data(const char* filename, char*& data, size_t& size)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    size = file.tellg();
    file.seekg(0, std::ios::beg);
    data = new char[size];

    file.read(data, size);
    file.close();
}

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

    auto duration = std::chrono::duration<float, std::milli>(stop - start);
    std::cout << "CPU compression time: " << duration.count() << " ms\n";
    float throughput = (in_bytes / (1024.0f * 1024.0f)) / (duration.count() / 1000.0f);
    std::cout << "CPU Throughput: " << throughput << " MB/s\n";

    auto ratio = (float) compressed_size / in_bytes;
    std::cout << "CPU ratio: " << ratio << "\n";

    delete[] compressed_data;
}

int main(int argc, char ** argv)
{
    if (argc != 2) {
    	std::cout << "usage: ./cpu_comp <file>\n";
	return 1;
    }

    const char* filename = argv[1];  // Change this to your data file
    char* uncompressed_data = nullptr;
    size_t data_size = 0;

    // Read the data from the file
    read_file_data(filename, uncompressed_data, data_size);

    // Run CPU compression
    cpu_compress(uncompressed_data, data_size);

    delete[] uncompressed_data;

    return 0;
}
