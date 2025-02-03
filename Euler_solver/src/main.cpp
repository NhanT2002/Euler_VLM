#include  "read_PLOT3D.h"
#include "SpatialDiscretization.h"
#include "TemporalDiscretization.h"
#include "Multigrid.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>

void read_input_file(const std::string& filename, double& Mach, double& alpha, double& p_inf, double& T_inf, int& multigrid, double& CFL_number, 
                     int& residual_smoothing, double& k2, double& k4, int& it_max, std::string& output_file, std::string& checkpoint_file, 
                     std::string& mesh_file, int& num_threads) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file." << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string key;
        char equalSign;
        if (iss >> key >> equalSign) {
            if (key == "Mach") {
                iss >> Mach;
            } else if (key == "alpha") {
                iss >> alpha;
                alpha = alpha * M_PI / 180.0;  // Convert alpha from degrees to radians
            } else if (key == "p_inf") {
                iss >> p_inf;
            } else if (key == "T_inf") {
                iss >> T_inf;
            } else if (key == "multigrid") {
                iss >> multigrid;
            } else if (key == "CFL_number") {
                iss >> CFL_number;
            } else if (key == "residual_smoothing") {
                iss >> residual_smoothing;
            } else if (key == "k2") {
                iss >> k2;
            } else if (key == "k4") {
                iss >> k4;
            } else if (key == "it_max") {
                iss >> it_max;
            } else if (key == "output_file") {
                iss >> output_file;
            } else if (key == "checkpoint_file") {
                iss >> checkpoint_file;
            } else if (key == "mesh_file") {
                iss >> mesh_file;
            } else if (key == "num_threads") {
                iss >> num_threads;
            }
        }
    }


    std::cout << "Read parameters from input file:\n";
    std::cout << "mesh_file = " << mesh_file << "\n";
    std::cout << "Mach = " << Mach << "\n";
    std::cout << "alpha = " << alpha / M_PI * 180.0<< "\n";
    std::cout << "p_inf = " << p_inf << "\n";
    std::cout << "T_inf = " << T_inf << "\n";
    std::cout << "multigrid = " << multigrid << "\n";
    std::cout << "CFL_number = " << CFL_number << "\n";
    std::cout << "residual_smoothing = " << residual_smoothing << "\n";
    std::cout << "k2 = " << k2 << "\n";
    std::cout << "k4 = " << k4 << "\n";
    std::cout << "it_max = " << it_max << "\n";

    inputFile.close();
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument
 

    // Read parameters from the input file
    double Mach, alpha, p_inf, T_inf, CFL_number, k2_coeff, k4_coeff;
    int multigrid, residual_smoothing, it_max, num_threads;
    std::string output_file, checkpoint_file, mesh_file;

    read_input_file(input_filename, Mach, alpha, p_inf, T_inf, multigrid, CFL_number, residual_smoothing, k2_coeff, k4_coeff, it_max, output_file, checkpoint_file, mesh_file, num_threads);

    omp_set_num_threads(num_threads); // Set number of threads
    int max_threads = omp_get_max_threads();
    int eig_threads = Eigen::nbThreads();
    std::cout << "Maximum available threads----------------: " << max_threads << std::endl;
    std::cout << "Eigen threads-----------------------------: " << eig_threads << std::endl;

    auto start = std::chrono::high_resolution_clock::now();


    // Read the PLOT3D mesh from a file
    auto [x, y] = read_PLOT3D_mesh(mesh_file);
    // std::cout << "x size: " << x.size() << " element" << std::endl;
    // std::cout << "y size: " << y.size() << " element" << std::endl;
    // std::cout << "x" << x << std::endl;
    // std::cout << "y" << y << std::endl;


    double rho_inf = p_inf/(T_inf*287);

    double a = std::sqrt(1.4*p_inf/rho_inf);
    double Vitesse = Mach*a;
    double u_inf = Vitesse*std::cos(alpha);
    double v_inf = Vitesse*std::sin(alpha);
    double E_inf = p_inf/((1.4-1)*rho_inf) + 0.5*std::pow(Vitesse, 2);

    double l_ref = 1.0;
    double U_ref = std::sqrt(p_inf/rho_inf);

    double rho = 1.0;
    double u = u_inf/U_ref;
    double v = v_inf/U_ref;
    double E = E_inf/(U_ref*U_ref);
    double T = 1.0;
    double p = 1.0;

    if (multigrid == 1) {
        Eigen::ArrayXXd W_0, W_1, W_2, W_3;
        std::vector<std::vector<double>> Residuals;

        SpatialDiscretization h_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, T_inf, U_ref);
        Multigrid multigrid_solver(h_state, CFL_number, residual_smoothing, k2_coeff, k4_coeff);

        SpatialDiscretization h2_state = multigrid_solver.mesh_restriction(h_state);    
        multigrid_solver.restriction(h_state, h2_state);
        SpatialDiscretization h4_state = multigrid_solver.mesh_restriction(h2_state);
        multigrid_solver.restriction(h2_state, h4_state);
        SpatialDiscretization h8_state = multigrid_solver.mesh_restriction(h4_state);
        multigrid_solver.restriction(h4_state, h8_state);

        std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h8_state, it_max);
        multigrid_solver.prolongation(h8_state, h4_state);
        std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h4_state, it_max);
        multigrid_solver.prolongation(h4_state, h2_state);
        std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h2_state, it_max/10);
        multigrid_solver.prolongation(h2_state, h_state); // Starting grid
        multigrid_solver.restriction_timestep(h_state, 1);

        // W cycle
        for (int it = 1; it < it_max; it++) {
            
            multigrid_solver.restriction(h_state, h2_state);
            multigrid_solver.restriction_timestep(h2_state, 1, -1);
            multigrid_solver.prolongation(h2_state, h_state);
            h_state.run_even();

            // multigrid_solver.restriction(h_state, h2_state);
            // multigrid_solver.restriction_timestep(h2_state, 1, -1);
            // multigrid_solver.restriction(h2_state, h4_state);
            // multigrid_solver.restriction_timestep(h4_state, 1, -1);
            // multigrid_solver.restriction(h4_state, h8_state);
            // multigrid_solver.restriction_timestep(h8_state, 1, -1);

            // multigrid_solver.prolongation(h8_state, h4_state);
            // multigrid_solver.restriction_timestep(h4_state, 1, -1);
            // multigrid_solver.restriction(h4_state, h8_state);
            // multigrid_solver.restriction_timestep(h8_state, 1, -1);
            // multigrid_solver.prolongation(h8_state, h4_state);
            // multigrid_solver.prolongation(h4_state, h2_state);
            // multigrid_solver.restriction_timestep(h2_state, 1, -1);

            // multigrid_solver.restriction(h2_state, h4_state);
            // multigrid_solver.restriction_timestep(h4_state, 1, -1);
            // multigrid_solver.restriction(h4_state, h8_state);
            // multigrid_solver.restriction_timestep(h8_state, 1, -1);
            // multigrid_solver.prolongation(h8_state, h4_state);
            // multigrid_solver.restriction_timestep(h4_state, 1, -1);
            // multigrid_solver.restriction(h4_state, h8_state);
            // multigrid_solver.restriction_timestep(h8_state, 1, -1);

            // multigrid_solver.prolongation(h8_state, h4_state);
            // multigrid_solver.prolongation(h4_state, h2_state);
            // multigrid_solver.prolongation(h2_state, h_state);

            std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h_state, 1, it);
            if (multigrid_solver.multigrid_convergence) {
                break;
            }
        }
    }

    else {
        TemporalDiscretization FVM(x, y, rho, u, v, E, T, p, T_inf, U_ref, CFL_number, residual_smoothing, k2_coeff, k4_coeff);
        auto [W_0, W_1, W_2, W_3, Residuals] = FVM.RungeKutta(it_max);
    }

    


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serialDuration = end - start;
    std::cout << "\nSolver duration: " << serialDuration.count() << " seconds\n";

    // auto [W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex] = cell_dummy_to_vertex_centered_airfoil(W_0(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
    //                                                                                               W_1(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
    //                                                                                               W_2(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
    //                                                                                               W_3(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)));
    // write_plot3d_2d(W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex, Mach, alpha, 0, 0, rho_inf, U_ref, output_file);

    return 0;

}














