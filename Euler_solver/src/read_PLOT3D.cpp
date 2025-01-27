#include "read_PLOT3D.h"
#include <fstream>   // For file handling
#include <iomanip>
#include <sstream>   // For string stream
#include <stdexcept> // For exception handling
#include <iostream>  // For printing errors (optional)

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> read_PLOT3D_mesh(const std::string& file_name) {

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
    std::vector<double> x(total_points);
    std::vector<double> y(total_points);

    // Read the coordinates from the file
    for (int i = 0; i < total_points; ++i) {
        if (!(file >> x[i])) {
            throw std::runtime_error("Error reading x coordinates from file: " + file_name);
        }
    }

    for (int i = 0; i < total_points; ++i) {
        if (!(file >> y[i])) {
            throw std::runtime_error("Error reading y coordinates from file: " + file_name);
        }
    }

    // Reshape 1D x and y arrays into 2D arrays (vectors of vectors)
    std::vector<std::vector<double>> x_2d(ny, std::vector<double>(nx));
    std::vector<std::vector<double>> y_2d(ny, std::vector<double>(nx));

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            x_2d[j][i] = x[j * nx + i];
            y_2d[j][i] = y[j * nx + i];
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

void write_plot3d_2d(
    const std::vector<std::vector<std::vector<double>>>& q,
    double mach,
    double alpha,
    double reyn,
    double time,
    double rho_ref,
    double U_ref,
    const std::string& solution_filename)
{
    // Get dimensions
    auto nj = q.size();
    auto ni = q[0].size();

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
    for (size_t j = 0; j < nj; ++j) {
        for (size_t i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << q[j][i][0]*rho_ref << "\n";
        }
    }
    for (size_t j = 0; j < nj; ++j) {
        for (size_t i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << q[j][i][1]*rho_ref*U_ref << "\n";
        }
    }
    for (size_t j = 0; j < nj; ++j) {
        for (size_t i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << q[j][i][2]*rho_ref*U_ref << "\n";
        }
    }
    for (size_t j = 0; j < nj; ++j) {
        for (size_t i = 0; i < ni; ++i) {  // Reverse the order: i first, then j
            solution_file << std::scientific << std::setprecision(16) << q[j][i][3]*rho_ref*U_ref*U_ref << "\n";
        }
    }
    solution_file.close();  // Close the solution file
}

void write_PLOT3D_mesh(const std::vector<std::vector<double>>& x, 
                       const std::vector<std::vector<double>>& y, 
                       const std::string& mesh_filename) {
    // Open the file
    std::ofstream file(mesh_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + mesh_filename);
    }

    // Get the dimensions of the grid
    int nx = x[0].size(); // Number of points in the x-direction
    int ny = x.size();    // Number of points in the y-direction

    // Write the dimensions of the grid (single block)
    file << nx << " " << ny << "\n";

    // Write the x-coordinates
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << x[j][i] << " ";
        }
        file << "\n";
    }

    // Write the y-coordinates
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << y[j][i] << " ";
        }
        file << "\n";
    }

    // Close the file
    file.close();
}

std::vector<std::vector<std::vector<double>>> cell_dummy_to_vertex_centered_airfoil(const std::vector<std::vector<std::vector<double>>>& q_cell)
{
    // Get dimensions
    const auto nj_cell = q_cell.size();
    const auto ni_cell = q_cell[0].size();
    const auto num_vars = q_cell[0][0].size();

    // The vertex-centered grid will be reduced in both directions to exclude the dummy cells
    const auto ni_vertex = ni_cell + 1;
    const auto nj_vertex = nj_cell - 1; // Excluding one dummy cell at the start and one at the end

    // Initialize an array for vertex-centered data
    std::vector q_vertex(nj_vertex,
        std::vector(ni_vertex, std::vector(num_vars, 0.0)));

    // Compute the average of adjacent cell-centered values for interior vertices
    for (size_t j = 0; j < nj_vertex; ++j) {
        for (size_t i = 0; i < ni_vertex; ++i) {
            for (size_t n = 0; n < num_vars; ++n) {
                q_vertex[j][i][n] = 0.25 * (q_cell[j][i % ni_cell][n] +
                                            q_cell[j][(i - 1 + ni_cell) % ni_cell][n] +
                                            q_cell[j + 1][(i - 1 + ni_cell) % ni_cell][n] +
                                            q_cell[j + 1][i % ni_cell][n]);
            }
        }
    }

    return q_vertex; // Return the vertex-centered data
}