#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

// Function to create an input file
void createInputFile(const std::string &filename, double mach, double alpha, const std::string &meshFile, int numThreads = 4) {
    // Template for the input file
    std::string templateContent = R"(num_threads = {NUM_THREADS}
mesh_file = {MESH_FILE}
Mach = {MACH}
alpha = {ALPHA}
p_inf = 1E5
T_inf = 300.0
CFL_number = 7.5
residual_smoothing = 1
k2 = 1.0
k4 = 2.0
it_max = 10000
output_file = {OUTPUT_FILE}
checkpoint_file = checkpoint_test.txt)";
    
    // Output file name
    std::ostringstream outputFile;
        outputFile << "output_files/output_Mach_" << std::fixed << std::setprecision(2) << mach
                << "_alpha_" << std::fixed << std::setprecision(2) << alpha
                << "_mesh_" << meshFile.substr(meshFile.find_last_of('/') + 1) << ".q";

    // Replace placeholders with actual values
    std::string content = templateContent;
    content.replace(content.find("{NUM_THREADS}"), 13, std::to_string(numThreads));
    content.replace(content.find("{MESH_FILE}"), 11, meshFile);
    content.replace(content.find("{OUTPUT_FILE}"), 13, outputFile.str());
    content.replace(content.find("{MACH}"), 6, std::to_string(mach));
    content.replace(content.find("{ALPHA}"), 7, std::to_string(alpha));

    // Write to the file
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << content;
        outFile.close();
        std::cout << "Input file created: " << filename << std::endl;
    } else {
        std::cerr << "Error: Unable to create file " << filename << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh_file>" << std::endl;
        return 1;
    }

    // Mesh file input from the command line
    std::string meshFile = argv[1];

    // Range of Mach and alpha values
    double machStart = 0.3, machEnd = 1.3, machStep = 0.1;
    double alphaStart = -5.0, alphaEnd = 12.0, alphaStep = 0.5;

    // Default number of threads
    int numThreads = 4;

    // Directory to store input files
    std::string directory = "input_files/";

    // Create input files for each combination of Mach and alpha
    for (double mach = machStart; mach <= machEnd; mach += machStep) {
        for (double alpha = alphaStart; alpha <= alphaEnd; alpha += alphaStep) {
            // Generate a filename for each input
            std::ostringstream filename;
            filename << directory << "input_Mach_" << std::fixed << std::setprecision(2) << mach
                     << "_alpha_" << std::fixed << std::setprecision(2) << alpha
                     << "_mesh_" << meshFile.substr(meshFile.find_last_of('/') + 1) << ".txt";

            // Create the input file
            createInputFile(filename.str(), mach, alpha, meshFile, numThreads);
        }
    }

    std::cout << "All input files generated in the directory: " << directory << std::endl;
    return 0;
}
