#include "read_PLOT3D.h"
#include <fstream>   // For file handling
#include <iomanip>
#include <sstream>   // For string stream
#include <stdexcept> // For exception handling
#include <iostream>  // For printing errors (optional)
#include <Eigen/Dense>

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> read_PLOT3D_mesh(const std::string& file_name) {

    std::ifstream file(file_name);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_name);
    }

    std::string line;
    // Read the first line to get grid dimensions
    if (!std::getline(file, line)) {
        throw std::runtime_error("Failed to read grid dimensions from file: " + file_name);
    }

    std::istringstream iss(line);
    int nx, ny;
    if (!(iss >> nx >> ny)) {
        throw std::runtime_error("Invalid grid dimensions format in file: " + file_name);
    }

    int total_points = nx * ny;

    // Initialize 1D arrays for x and y coordinates
    Eigen::ArrayXd x(total_points);
    Eigen::ArrayXd y(total_points);

    // Read the coordinates from the file
    for (int i = 0; i < total_points; ++i) {
        if (!(file >> x(i))) {
            throw std::runtime_error("Error reading x coordinates from file: " + file_name);
        }
    }

    for (int i = 0; i < total_points; ++i) {
        if (!(file >> y(i))) {
            throw std::runtime_error("Error reading y coordinates from file: " + file_name);
        }
    }

    // Reshape 1D x and y arrays into 2D arrays (vectors of vectors)
    Eigen::ArrayXXd x_2d(ny, nx);
    Eigen::ArrayXXd y_2d(ny, nx);


    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            x_2d(j, i) = x(j * nx + i);
            y_2d(j, i) = y(j * nx + i);
        }
    }

    return {x_2d, y_2d};
}

std::tuple<int, int, double, double, double, double, std::vector<std::vector<std::vector<double>>>> read_PLOT3D_solution(const std::string& solution_filename) {
    std::ifstream solution_file(solution_filename);
    if (!solution_file) {
        throw std::runtime_error("Could not open file: " + solution_filename);
    }

    int ni, nj;
    std::string line;

    // Read grid dimensions
    if (!std::getline(solution_file, line)) {
        throw std::runtime_error("Failed to read grid dimensions from file: " + solution_filename);
    }
    std::istringstream iss(line);
    if (!(iss >> ni >> nj)) {
        throw std::runtime_error("Invalid grid dimensions format in file: " + solution_filename);
    }

    // Read freestream conditions
    double mach, alpha, reyn, time;
    if (!std::getline(solution_file, line)) {
        throw std::runtime_error("Failed to read freestream conditions from file: " + solution_filename);
    }
    iss.clear();
    iss.str(line);
    if (!(iss >> mach >> alpha >> reyn >> time)) {
        throw std::runtime_error("Invalid freestream conditions format in file: " + solution_filename);
    }

    // Initialize the q array (nj, ni, 4)
    std::vector<std::vector<std::vector<double>>> q(nj, std::vector<std::vector<double>>(ni, std::vector<double>(4)));

    // Read flow variables
    for (int n = 0; n < 4; ++n) {  // Iterate over the 4 variables (density, x-momentum, y-momentum, energy)
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {  // Read in the reversed order: i first, then j
                if (!std::getline(solution_file, line)) {
                    throw std::runtime_error("Failed to read flow variable at (i, j): (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                }
                q[j][i][n] = std::stod(line);  // Convert string to double
            }
        }
    }

    // Return all parameters as a tuple
    return std::make_tuple(ni, nj, mach, alpha, reyn, time, q);
}

void write_plot3d_2d(const Eigen::ArrayXXd& W_0,
                    const Eigen::ArrayXXd& W_1,
                    const Eigen::ArrayXXd& W_2,
                    const Eigen::ArrayXXd& W_3,
                    double mach,
                    double alpha,
                    double reyn,
                    double time,
                    double rho_ref,
                    double U_ref,
                    const std::string& solution_filename)
{
    // Get dimensions
    auto nj = W_0.rows();
    auto ni = W_0.cols();

    // Write solution file (2D.q)
    std::ofstream solution_file(solution_filename);
    if (!solution_file) {
        throw std::runtime_error("Could not open solution file: " + solution_filename);
    }

    solution_file << ni << " " << nj << "\n";  // Grid dimensions again
    // Write freestream conditions
    solution_file << std::scientific << std::setprecision(16) << mach << " "
                  << alpha << " " << reyn << " " << time << "\n";

    // Write flow variables (density, x-momentum, y-momentum, energy)
    for (auto j = 0; j < nj; ++j) {
        for (auto i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << W_0(j, i)*rho_ref << "\n";
        }
    }
    for (auto j = 0; j < nj; ++j) {
        for (auto i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << W_1(j, i)*rho_ref*U_ref << "\n";
        }
    }
    for (auto j = 0; j < nj; ++j) {
        for (auto i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << W_2(j, i)*rho_ref*U_ref << "\n";
        }
    }
    for (auto j = 0; j < nj; ++j) {
        for (auto i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << W_3(j, i)*rho_ref*U_ref*U_ref << "\n";
        }
    }
    solution_file.close();  // Close the solution file

    std::cout << "PLOT3D file " << solution_filename <<  " written successfully." << std::endl;
}

void write_PLOT3D_mesh(const Eigen::ArrayXXd& x, 
                       const Eigen::ArrayXXd& y, 
                       const std::string& mesh_filename) {
    // Open the file
    std::ofstream file(mesh_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + mesh_filename);
    }

    // Get the dimensions of the grid
    int nx = x.rows(); // Number of points in the x-direction
    int ny = x.cols();    // Number of points in the y-direction

    // Write the dimensions of the grid (single block)
    file << nx << " " << ny << "\n";

    // Write the x-coordinates
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << x(j, i) << " ";
        }
        file << "\n";
    }

    // Write the y-coordinates
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << y(j, i) << " ";
        }
        file << "\n";
    }

    // Close the file
    file.close();

    std::cout << "PLOT3D file " << mesh_filename <<  " written successfully." << std::endl;
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> cell_dummy_to_vertex_centered_airfoil(const Eigen::ArrayXXd& W_0_dummy,
                                                                                                                    const Eigen::ArrayXXd& W_1_dummy,
                                                                                                                    const Eigen::ArrayXXd& W_2_dummy,
                                                                                                                    const Eigen::ArrayXXd& W_3_dummy)
{
    // Get dimensions
    const auto nj_cell = W_0_dummy.rows();
    const auto ni_cell = W_0_dummy.cols();


    // The vertex-centered grid will be reduced in both directions to exclude the dummy cells
    const auto ni_vertex = ni_cell - 1; // Excluding one dummy cell at the start and one at the end
    const auto nj_vertex = nj_cell - 1; // Excluding one dummy cell at the start and one at the end

    // Initialize an array for vertex-centered data
    Eigen::ArrayXXd W_0_vertex(nj_vertex, ni_vertex);
    Eigen::ArrayXXd W_1_vertex(nj_vertex, ni_vertex);
    Eigen::ArrayXXd W_2_vertex(nj_vertex, ni_vertex);
    Eigen::ArrayXXd W_3_vertex(nj_vertex, ni_vertex);


    // Compute the average of adjacent cell-centered values for interior vertices
    for (auto j = 0; j < nj_vertex; ++j) {
        for (auto i = 0; i < ni_vertex; ++i) {
            W_0_vertex(j, i) = 0.25 * (W_0_dummy(j,i) + W_0_dummy(j+1,i) + W_0_dummy(j,i+1) + W_0_dummy(j+1,i+1));
            W_1_vertex(j, i) = 0.25 * (W_1_dummy(j,i) + W_1_dummy(j+1,i) + W_1_dummy(j,i+1) + W_1_dummy(j+1,i+1));
            W_2_vertex(j, i) = 0.25 * (W_2_dummy(j,i) + W_2_dummy(j+1,i) + W_2_dummy(j,i+1) + W_2_dummy(j+1,i+1));
            W_3_vertex(j, i) = 0.25 * (W_3_dummy(j,i) + W_3_dummy(j+1,i) + W_3_dummy(j,i+1) + W_3_dummy(j+1,i+1));
        }
    }

    return {W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex}; // Return the vertex-centered data
}

void save_time_residuals(const std::vector<double>& iteration_times,
                        const std::vector<std::vector<double>>& Residuals,
                        const std::string& file_name) {
    // Open the file
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_name);
    }

    // Write the residuals to the file
    for (size_t i = 0; i < Residuals.size(); ++i) {
        file << iteration_times[i] << ",";
        for (size_t j = 0; j < Residuals[i].size(); ++j) {
            file << Residuals[i][j] << ",";
        }
        file << "\n";
    }

    // Close the file
    file.close();

    std::cout << "Residuals saved to file: " << file_name << std::endl;
}