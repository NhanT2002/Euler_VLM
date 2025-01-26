#include "Multigrid.h"
#include "read_PLOT3D.h"
#include "vector_helper.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <omp.h>



// Constructor definition
Multigrid::Multigrid(SpatialDiscretization& h_state, double sigma, int res_smoothing, double k2_coeff, double k4_coeff)
    : TemporalDiscretization(h_state.x, h_state.y, h_state.rho, h_state.u, h_state.v,
                              h_state.E, h_state.T, h_state.p, h_state.T_ref,
                              h_state.U_ref, sigma, res_smoothing, k2_coeff, k4_coeff),
      h_state(h_state), sigma(sigma), res_smoothing(res_smoothing), k2_coeff(k2_coeff), k4_coeff(k4_coeff) {}


// Restriction implementation (fine to coarse grid)
SpatialDiscretization Multigrid::restriction(SpatialDiscretization& h_state) {
    int ny_2h = (h_state.ny+1) / 2;
    int nx_2h = (h_state.nx+1) / 2;
    std::vector<std::vector<double>> x_2h(ny_2h, std::vector<double>(nx_2h));
    std::vector<std::vector<double>> y_2h(ny_2h, std::vector<double>(nx_2h));
    for (int j = 0; j < (h_state.ny - 1)/2 + 1; ++j) {
        for (int i = 0; i < (h_state.nx - 1)/2 + 1; ++i) {
            x_2h[j][i] = h_state.x[2 * j][2 * i];
            y_2h[j][i] = h_state.y[2 * j][2 * i];
        }
    }

    // Verify the mesh restriction
    // write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xy");
    // write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xy");

    SpatialDiscretization h2_state(x_2h, y_2h, h_state.rho, h_state.u, h_state.v, h_state.E,
                                    h_state.T, h_state.p, h_state.k2_coeff, h_state.k4_coeff,
                                    h_state.T_ref, h_state.U_ref);

    int nx = h2_state.nx;
    int ny = h2_state.ny;

    std::cout << "nx: " << nx << " ny: " << ny << std::endl;
    
    for (int j=2; j < ny_2h-1+2; ++j) {
        for (int i=0; i < nx_2h-1; ++i) {
            // Transfer operators for the cell-centered scheme
            // std::cout << j << " " << i << std::endl;
            // std::cout << "h_state W" << std::endl;
            // printVector(h_state.W[2*j][2*i]);
            // printVector(h_state.W[2*j][2*i+1]);
            // printVector(h_state.W[2*+1][2*i+1]);
            // printVector(h_state.W[2*j+1][2*i]);
            h2_state.W[j][i] = vector_scale(1/(h_state.OMEGA[2*j][2*i] + h_state.OMEGA[2*j][2*i+1] + h_state.OMEGA[2*j+1][2*i+1] + h_state.OMEGA[2*j+1][2*i]),
                                vector_add(vector_add(vector_scale(h_state.OMEGA[2*j][2*i], h_state.W[2*j][2*i]), vector_scale(h_state.OMEGA[2*j][2*i+1], h_state.W[2*j][2*i+1])), 
                                vector_add(vector_scale(h_state.OMEGA[2*j+1][2*i+1], h_state.W[2*j+1][2*i+1]), vector_scale(h_state.OMEGA[2*j+1][2*i], h_state.W[2*j+1][2*i]))));
            // std::cout << "h2_state W" << std::endl;
            // printVector(h2_state.W[j][i]);
            // std::cout << " " << std::endl;
            
            // Restriction operator
            // std::cout << "h_state residual" << std::endl;
            // printVector(vector_subtract(h_state.R_c[2*(j-2)][2*i], h_state.R_d[2*(j-2)][2*i]));
            // printVector(vector_subtract(h_state.R_c[2*(j-2)][2*i+1], h_state.R_d[2*(j-2)][2*i+1]));
            // printVector(vector_subtract(h_state.R_c[2*(j-2)+1][2*i+1], h_state.R_d[2*(j-2)+1][2*i+1]));
            // printVector(vector_subtract(h_state.R_c[2*(j-2)+1][2*i], h_state.R_d[2*(j-2)+1][2*i]));
            h2_state.restriction_operator[(j-2)][i] = vector_add(vector_add(vector_subtract(h_state.R_c[2*(j-2)][2*i], h_state.R_d[2*(j-2)][2*i]), 
                                                                        vector_subtract(h_state.R_c[2*(j-2)][2*i+1], h_state.R_d[2*(j-2)][2*i+1])),
                                                            vector_add(vector_subtract(h_state.R_c[2*(j-2)+1][2*i+1], h_state.R_d[2*(j-2)+1][2*i+1]), 
                                                                        vector_subtract(h_state.R_c[2*(j-2)+1][2*i], h_state.R_d[2*(j-2)+1][2*i])));
            // std::cout << "h2_state residual" << std::endl;
            // printVector(h2_state.restriction_operator[(j-2)][i]);
            // std::cout << " " << std::endl;
        }
    }

    // Compute R_2h_0
    h2_state.run_even();

    // Forcing function
    for (int j=0; j < ny_2h-1; ++j) {
        for (int i=0; i < nx_2h-1; ++i) {
            h2_state.forcing_function[j][i] = vector_subtract(h2_state.restriction_operator[j][i], vector_subtract(h2_state.R_c[j][i], h2_state.R_d[j][i]));   
        }
    }
 
    return h2_state;
}

SpatialDiscretization Multigrid::restriction_timestep(int it_max) {
    current_state = h_state;
    current_state.run_even();

    // Store initial interpolated solution
    std::vector<std::vector<std::vector<double>>> W_2h_0 = current_state.W;

    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;

    int ny = current_state.W.size();
    int nx = current_state.W[0].size();
    std::cout << ny << " " << nx << std::endl;

    // Initialize R_d0
    for (int j = 2; j < ny - 2; ++j) {
        for (int i = 0; i < nx; ++i) {
            current_state.R_d0[j-2][i] = current_state.R_d[j-2][i];
        }
    }

    std::vector<std::vector<double>> Residuals;
    std::vector<int> iteration;


    Residuals = std::vector<std::vector<double>>{};
    iteration = std::vector<int>{};


    std::vector<std::vector<std::vector<double>>> all_Res(ny - 4, std::vector<std::vector<double>>(nx, std::vector<double>(4, 1.0)));
    std::vector<std::vector<std::vector<double>>> all_dw(ny - 4, std::vector<std::vector<double>>(nx, std::vector<double>(4, 1.0)));
    std::vector<std::vector<std::vector<double>>> q(ny - 4, std::vector<std::vector<double>>(nx, std::vector<double>(4, 1.0)));

    // Fill q array
    for (int j = 2; j < ny - 2; ++j) {
        for (int i = 0; i < nx; ++i) {
            q[j - 2][i] = current_state.W[j][i];
        }
    }

        std::vector<double> first_residual;
        int it = 0;
        std::vector<double> normalized_residuals = {1, 1, 1, 1};

    while (it < it_max) {
        if (res_smoothing == 0) {
            auto dt = compute_dt();
            std::vector<std::vector<std::vector<double>>> W_0 = current_state.W;
            // Stage 1
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = current_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd0));
                    std::vector<double> dW = vector_scale(-a1 * dt_loc, Res);
                    current_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            current_state.run_odd();

            // Stage 2
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = current_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd0));
                    std::vector<double> dW = vector_scale(-a2 * dt_loc, Res);
                    current_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            current_state.run_even();

            // Stage 3
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd20 = vector_add(vector_scale(b3, current_state.R_d[j-2][i]), vector_scale(1-b3, current_state.R_d0[j-2][i]));
                    current_state.R_d0[j-2][i] = Rd20;
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd20));
                    std::vector<double> dW = vector_scale(-a3 * dt_loc, Res);
                    current_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            current_state.run_odd();

            // Stage 4
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd20 = current_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd20));
                    std::vector<double> dW = vector_scale(-a4 * dt_loc, Res);
                    current_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            current_state.run_even();

            // Stage 5, Final update
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd42 = vector_add(vector_scale(b5, current_state.R_d[j-2][i]), vector_scale(1-b5, current_state.R_d0[j-2][i]));
                    current_state.R_d0[j-2][i] = Rd42;
                    std::vector<double> Res = vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd42);
                    std::vector<double> dW = vector_scale(-a5 * dt_loc / current_state.OMEGA[j][i], Res);
                    current_state.W[j][i] = vector_add(W_0[j][i], dW);

                    all_Res[j - 2][i] = Res;
                    all_dw[j - 2][i] = dW;
                    q[j - 2][i] = current_state.W[j][i];
                    // if (j==2) {
                    //     std::cout << i << std::endl;
                    //     std::cout << dt << std::endl;
                    //     std::cout << dW[0] << " " << dW[1] << " " << dW[2] << " " << dW[3] << " " << std::endl;
                    //     std::cout << current_state.W[j][i][0] << " " << current_state.W[j][i][1] << " " << current_state.W[j][i][2] << " " << current_state.W[j][i][3] << " " << std::endl;
                    //     std::cout << std::endl;
                    // }
                }
            }
            current_state.run_odd();
        }
        else {
            std::vector d((ny - 4)*nx, std::vector<double>(4));

            auto dt = compute_dt();
            std::vector<std::vector<std::vector<double>>> W_0 = current_state.W;
            // Stage 1
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = current_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd0));
                    std::vector<double> dW = vector_scale(dt_loc, Res);
                    d[(j-2)*nx + i] = dW;
                }
            }
            auto [a_I, b_I, c_I, a_J, b_J, c_J] = compute_abc();
            std::vector<std::vector<double>> R_star = thomasAlgorithm(a_I, b_I, c_I, d);
            R_star = reshapeColumnWise(R_star, ny-4, nx);
            std::vector<std::vector<double>> R_star_star = thomasAlgorithm(a_J, b_J, c_J, R_star);

            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    std::vector<double> dW = vector_scale(a1, R_star_star[i*nx + (j-2)]);
                    current_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            current_state.run_odd();

            // Stage 2
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = current_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd0));
                    std::vector<double> dW = vector_scale(dt_loc, Res);
                    d[(j-2)*nx + i] = dW;
                }
            }
            std::tie(a_I, b_I, c_I, a_J, b_J, c_J) = compute_abc();
            R_star = thomasAlgorithm(a_I, b_I, c_I, d);
            R_star = reshapeColumnWise(R_star, ny-4, nx);
            R_star_star = thomasAlgorithm(a_J, b_J, c_J, R_star);
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    std::vector<double> dW = vector_scale(a2, R_star_star[i*nx + (j-2)]);
                    current_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            current_state.run_even();

            // Stage 3
            // #pragma omp parallel for
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd20 = vector_add(vector_scale(b3, current_state.R_d[j-2][i]), vector_scale(1-b3, current_state.R_d0[j-2][i]));
                    current_state.R_d0[j-2][i] = Rd20;
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd20));
                    std::vector<double> dW = vector_scale(dt_loc, Res);
                    d[(j-2)*nx + i] = dW;
                }
            }
            std::tie(a_I, b_I, c_I, a_J, b_J, c_J) = compute_abc();
            R_star = thomasAlgorithm(a_I, b_I, c_I, d);
            R_star = reshapeColumnWise(R_star, ny-4, nx);
            R_star_star = thomasAlgorithm(a_J, b_J, c_J, R_star);
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    std::vector<double> dW = vector_scale(a3, R_star_star[i*nx + (j-2)]);
                    current_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            current_state.run_odd();

            // Stage 4
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd20 = current_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]), Rd20));
                    std::vector<double> dW = vector_scale(dt_loc, Res);
                    d[(j-2)*nx + i] = dW;
                }
            }
            std::tie(a_I, b_I, c_I, a_J, b_J, c_J) = compute_abc();
            R_star = thomasAlgorithm(a_I, b_I, c_I, d);
            R_star = reshapeColumnWise(R_star, ny-4, nx);
            R_star_star = thomasAlgorithm(a_J, b_J, c_J, R_star);
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    std::vector<double> dW = vector_scale(a4, R_star_star[i*nx + (j-2)]);
                    current_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            current_state.run_even();

            // Stage 5, Final update
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd42 = vector_add(vector_scale(b5, current_state.R_d[j-2][i]), vector_scale(1-b5, current_state.R_d0[j-2][i]));
                    current_state.R_d0[j-2][i] = Rd42;
                    std::vector<double> Res = vector_subtract(vector_add(current_state.R_c[j - 2][i], current_state.forcing_function[j-2][i]) , Rd42);
                    std::vector<double> dW = vector_scale(dt_loc / current_state.OMEGA[j][i], Res);
                    d[(j-2)*nx + i] = dW;
                }
            }
            std::tie(a_I, b_I, c_I, a_J, b_J, c_J) = compute_abc();
            R_star = thomasAlgorithm(a_I, b_I, c_I, d);
            R_star = reshapeColumnWise(R_star, ny-4, nx);
            R_star_star = thomasAlgorithm(a_J, b_J, c_J, R_star);
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    std::vector<double> dW = vector_scale(a5, R_star_star[i*nx + (j-2)]);
                    current_state.W[j][i] = vector_subtract(W_0[j][i], dW);

                    all_dw[j - 2][i] = dW;
                    q[j - 2][i] = current_state.W[j][i];
                }
            }
            current_state.run_odd();
        }



        // Compute L2 norm (placeholder logic)
        std::vector<double> l2_norm = compute_L2_norm(all_dw);

        if (it == 0) {
            first_residual = l2_norm;
        }
        normalized_residuals = vector_divide(l2_norm, first_residual);
        iteration.push_back(it);
        Residuals.push_back(l2_norm);

        auto [C_L, C_D, C_M] = compute_coeff();

        std::cout << "It " << it << ": L2 Norms = ";
        for (const auto &norm : normalized_residuals) {
            std::cout << norm << " ";
        }
        std::cout << "C_L:" << C_L << " C_D:" << C_D << " C_M:" << C_M << std::endl;

        // Check for convergence
        if (*std::max_element(normalized_residuals.begin(), normalized_residuals.end()) <= 1e-11) {
            break; // Exit the loop if convergence criterion is met
        }

        // Save checkpoint at each 1000 iteration
        if (it%1000 == 0) {
            save_checkpoint(q, iteration, Residuals);
        }



        it++;
    }

    // Corse grid correction
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            current_state.deltaW_2h[j][i] = vector_subtract(current_state.W[j][i], W_2h_0[j][i]);
        }
    }
    

    return current_state;
}

// Prolongation implementation (coarse to fine grid)
SpatialDiscretization Multigrid::prolongation(SpatialDiscretization& h2_state, std::vector<std::vector<double>>& x_h, std::vector<std::vector<double>>& y_h) {
    int ny_cell_2h = h2_state.ny - 1;
    int nx_cell_2h = h2_state.nx - 1;

    SpatialDiscretization h_state(x_h, y_h, h2_state.rho, h2_state.u, h2_state.v, h2_state.E,
                                    h2_state.T, h2_state.p, h2_state.k2_coeff, h2_state.k4_coeff,
                                    h2_state.T_ref, h2_state.U_ref);

    int nx = h_state.nx;
    int ny = h_state.ny;

    std::cout << "nx: " << nx << " ny: " << ny << std::endl;
    for (int j=2; j < ny; j += 2) {
        // upper right corner
        // std::cout << "upper right corner" << std::endl;
        for (int i=0; i < nx - 1; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = i/2;
            int j_I = j/2+1;
            int i_Im1 = (i/2 - 1 + nx_cell_2h) % nx_cell_2h;
            int j_Jm1 = j/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Im1: " << i_Im1 << " j_Jm1: " << j_Jm1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jm1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Im1]), h2_state.deltaW_2h[j_Jm1][i_Im1])));
        }
        // upper left corner
        // std::cout << "upper left corner" << std::endl;
        for (int i=1; i < nx; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = (i-1)/2;
            int j_I = j/2+1;
            int i_Ip1 = ((i+1)/2) % nx_cell_2h;
            int j_Jm1 = j/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Ip1: " << i_Ip1 << " j_Jm1: " << j_Jm1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jm1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Ip1]), h2_state.deltaW_2h[j_Jm1][i_Ip1])));
        }
    }
    for (int j=3; j < ny + 1; j += 2) {
        // lower right corner
        // std::cout << "lower right corner" << std::endl;
        for (int i=0; i < nx - 1; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = i/2;
            int j_I = (j+1)/2;
            int i_Im1 = (i/2 - 1 + nx_cell_2h) % nx_cell_2h;
            int j_Jp1 = (j+3)/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Im1: " << i_Im1 << " j_Jp1: " << j_Jp1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jp1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Im1]), h2_state.deltaW_2h[j_Jp1][i_Im1])));
        }
        // lower left corner
        // std::cout << "lower left corner" << std::endl;
        for (int i=1; i < nx; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = (i-1)/2;
            int j_I = (j+1)/2;
            int i_Ip1 = ((i+1)/2) % nx_cell_2h;
            int j_Jp1 = (j+3)/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Ip1: " << i_Ip1 << " j_Jp1: " << j_Jp1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jp1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Ip1]), h2_state.deltaW_2h[j_Jp1][i_Ip1])));
        }
    }
    
    return h_state;
}


