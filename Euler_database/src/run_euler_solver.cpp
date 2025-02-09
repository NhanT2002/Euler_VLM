#include <iostream>
#include <filesystem>
#include <cstdlib> // For std::system
#include <string>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <inputDir>" << std::endl;
        return 1;
    }

    // Path to the directory containing input files
    std::string inputDir = argv[1];

    // Path to the Euler solver executable
    std::string eulerSolver = "../Euler_solver/bin/euler_solver";

    // Check if the Euler solver exists
    if (!fs::exists(eulerSolver)) {
        std::cerr << "Error: Euler solver executable (" << eulerSolver << ") not found!" << std::endl;
        return 1;
    }

    // Check if the input directory exists
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        std::cerr << "Error: Input directory (" << inputDir << ") not found or is not a directory!" << std::endl;
        return 1;
    }

    // Iterate through all files in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        // Check if the current entry is a regular file
        if (fs::is_regular_file(entry)) {
            // Get the path of the input file
            std::string inputFile = entry.path().string();
            std::cout << "Running Euler solver with input file: " << inputFile << std::endl;

            // Construct the command to run the Euler solver
            std::string command = eulerSolver + " " + inputFile;

            // Execute the command
            int result = std::system(command.c_str());

            // Check if the command executed successfully
            if (result != 0) {
                std::cerr << "Error: Euler solver failed for " << inputFile << std::endl;
                return 1;
            }
        }
    }

    std::cout << "All input files processed successfully." << std::endl;
    return 0;
}
