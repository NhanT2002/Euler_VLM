#ifndef READ_PLOT3D_H
#define READ_PLOT3D_H

#include <vector>
#include <string>
#include <tuple>
#include <Eigen/Dense>

// Function to read PLOT3D mesh from a file
std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> read_PLOT3D_mesh(const std::string& file_name);

// Function to read PLOT3D solution from a file
std::tuple<int, int, double, double, double, double, std::vector<std::vector<std::vector<double>>>> read_PLOT3D_solution(const std::string& solution_filename);

// Function to write PLOT3D solution from a file
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
                    const std::string& solution_filename = "2D.q");


// Function to write PLOT3D mesh to a file
void write_PLOT3D_mesh(const Eigen::ArrayXXd& x,
                       const Eigen::ArrayXXd& y,
                       const std::string& mesh_filename);


// Function to convert cell centered solution to vertex centered solution
std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> cell_dummy_to_vertex_centered_airfoil(const Eigen::ArrayXXd& W_0_dummy,
                                                                                                                    const Eigen::ArrayXXd& W_1_dummy,
                                                                                                                    const Eigen::ArrayXXd& W_2_dummy,
                                                                                                                    const Eigen::ArrayXXd& W_3_dummy);

void save_time_residuals(const std::vector<double>& iteration_times,
                     const std::vector<std::vector<double>>& Residuals,
                     const std::string& file_name = "checkpoint.txt");                                                                                                                    

#endif //READ_PLOT3D_H
