#include <iostream>
#include <fstream>
#include <random>

void generate_sample_data(const char* filename, size_t size)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return;
    }

    std::mt19937 random_gen(42);

    for (size_t i = 0; i < size; ++i) {
        char byte = random_gen() % 256;
        file.write(&byte, sizeof(byte));
    }

    file.close();
    std::cout << "Sample data written to " << filename << " (" << size << " bytes)." << std::endl;
}

int main(int argc, char ** argv)
{
    const char * filename = argv[2];
    const size_t data_size = std::stoll(argv[1]);

    generate_sample_data(filename, data_size);

    return 0;
}