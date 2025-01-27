#ifndef READ_PLOT3D_H
#define READ_PLOT3D_H

#include <vector>
#include <string>
#include <tuple>

// Function to read PLOT3D mesh from a file
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> read_PLOT3D_mesh(const std::string& file_name);

// Function to read PLOT3D solution from a file
std::tuple<int, int, double, double, double, double, std::vector<std::vector<std::vector<double>>>> read_PLOT3D_solution(const std::string& solution_filename);

// Function to write PLOT3D solution from a file
void write_plot3d_2d(const std::vector<std::vector<std::vector<double>>>& q,
                     double mach,
                     double alpha,
                     double reyn,
                     double time,
                     double rho_ref,
                     double U_ref,
                     const std::string& solution_filename = "2D.q");


// Function to write PLOT3D mesh to a file
void write_PLOT3D_mesh(const std::vector<std::vector<double>>& x,
                       const std::vector<std::vector<double>>& y,
                       const std::string& mesh_filename);


// Function to convert cell centered solution to vertex centered solution
std::vector<std::vector<std::vector<double>>> cell_dummy_to_vertex_centered_airfoil(const std::vector<std::vector<std::vector<double>>>& q_cell);

#endif //READ_PLOT3D_H
