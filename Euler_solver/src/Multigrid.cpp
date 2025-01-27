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
    h_state.run_even();
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
    write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xy");
    write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xy");

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
            h2_state.restriction_operator[(j-2)][i] = vector_add(vector_add(vector_subtract(vector_add(h_state.R_c[2*(j-2)][2*i], h_state.forcing_function[2*(j-2)][2*i]), h_state.R_d[2*(j-2)][2*i]), 
                                                                        vector_subtract(vector_add(h_state.R_c[2*(j-2)][2*i+1], h_state.forcing_function[2*(j-2)][2*i+1]), h_state.R_d[2*(j-2)][2*i+1])),
                                                            vector_add(vector_subtract(vector_add(h_state.R_c[2*(j-2)+1][2*i+1], h_state.forcing_function[2*(j-2)+1][2*i+1]), h_state.R_d[2*(j-2)+1][2*i+1]), 
                                                                        vector_subtract(vector_add(h_state.R_c[2*(j-2)+1][2*i], h_state.forcing_function[2*(j-2)+1][2*i]), h_state.R_d[2*(j-2)+1][2*i])));
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

    Multigrid::h_state = h2_state;
 
    return h2_state;
}

std::vector<std::vector<double>> Multigrid::compute_dt(SpatialDiscretization& state) {
    auto ny = state.W.size();
    auto nx = state.W[0].size();

    std::vector dt(state.y.size() - 1 + 4, std::vector<double>(state.x.size() - 1));

    #pragma omp parallel for
    for (size_t j = 2; j < ny - 2; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            dt[j][i] = sigma*state.OMEGA[j][i]/(state.Lambda_I[j][i]+state.Lambda_J[j][i]);

        }
    }

    return dt;
}

std::tuple<double, double, double> Multigrid::compute_coeff(SpatialDiscretization& state) {
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;
    int nx = state.W[0].size();
    std::vector<double> p_array(nx);
    double Fx = 0.0;
    double Fy = 0.0;
    double M = 0.0;
    for (int i = 0; i < nx; ++i) {
        double& rho = state.W[2][i][0];
        double& rho_u = state.W[2][i][1];
        double& rho_v = state.W[2][i][2];
        double& rho_E = state.W[2][i][3];
        double p = (1.4-1)*(rho_E-0.5*(rho_u*rho_u+rho_v*rho_v)/rho);
        Fx += p*state.n[2][i][0][0]*state.Ds[2][i][0];
        Fy += p*state.n[2][i][0][1]*state.Ds[2][i][0];

        double x_mid = 0.5*(state.x[0][i] + state.x[0][i+1]);
        double y_mid = 0.5*(state.y[0][i] + state.y[0][i+1]);
        M += p*(-(x_mid-x_ref)*state.n[2][i][0][1] + (y_mid-y_ref)*state.n[2][i][0][0])*state.Ds[2][i][0];
    }

    double L = Fy*std::cos(state.alpha) - Fx*std::sin(state.alpha);
    double D = Fy*std::sin(state.alpha) + Fx*std::cos(state.alpha);

    double C_L = L/(0.5*rho*(u*u+v*v)*c);
    double C_D = D/(0.5*rho*(u*u+v*v)*c);
    double C_M = M/(0.5*rho*(u*u+v*v)*c*c);

    return {C_L, C_D, C_M};
}

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> Multigrid::restriction_timestep(SpatialDiscretization& h_state, int it_max) {

    h_state.run_even();

    // Store initial interpolated solution
    std::vector<std::vector<std::vector<double>>> W_2h_0 = h_state.W;

    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;

    int ny = h_state.W.size();
    int nx = h_state.W[0].size();
    std::cout << ny << " " << nx << std::endl;

    // Initialize R_d0
    for (int j = 2; j < ny - 2; ++j) {
        for (int i = 0; i < nx; ++i) {
            h_state.R_d0[j-2][i] = h_state.R_d[j-2][i];
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
            q[j - 2][i] = h_state.W[j][i];
        }
    }

        std::vector<double> first_residual;
        int it = 0;
        std::vector<double> normalized_residuals = {1, 1, 1, 1};

    while (it < it_max) {
        if (res_smoothing == 0) {
            auto dt = Multigrid::compute_dt(h_state);
            // std::cout << dt[0].size() << std::endl;
            std::vector<std::vector<std::vector<double>>> W_0 = h_state.W;
            // Stage 1
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = h_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd0));
                    std::vector<double> dW = vector_scale(-a1 * dt_loc, Res);
                    h_state.W[j][i] = vector_add(W_0[j][i], dW);

                    // std::cout << "dt_loc = " << dt_loc << std::endl;
                    // std::cout << "W = " << h_state.W[j][i][0] << " " << h_state.W[j][i][1] << " " << h_state.W[j][i][2] << " " << h_state.W[j][i][3] << std::endl;
                    // std::cout << "R_c = " << h_state.R_c[j - 2][i][0] << " " << h_state.R_c[j - 2][i][1] << " " << h_state.R_c[j - 2][i][2] << " " << h_state.R_c[j - 2][i][3] << std::endl;
                    // std::cout << "Rd0 = " << Rd0[0] << " " << Rd0[1] << " " << Rd0[2] << " " << Rd0[3] << std::endl;
                    // std::cout << "forcing = " << h_state.forcing_function[j-2][i][0] << " " << h_state.forcing_function[j-2][i][1] << " " << h_state.forcing_function[j-2][i][2] << " " << h_state.forcing_function[j-2][i][3] << std::endl;
                    // std::cout << "OMEGA = " << h_state.OMEGA[j][i] << std::endl;
                    // std::cout << "Res = " << Res[0] << " " << Res[1] << " " << Res[2] << " " << Res[3] << std::endl;
                    // std::cout << "dW = " << dW[0] << " " << dW[1] << " " << dW[2] << " " << dW[3] << std::endl;
                }
            }
            h_state.run_odd();

            // Stage 2
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = h_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd0));
                    std::vector<double> dW = vector_scale(-a2 * dt_loc, Res);
                    h_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            h_state.run_even();

            // Stage 3
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd20 = vector_add(vector_scale(b3, h_state.R_d[j-2][i]), vector_scale(1-b3, h_state.R_d0[j-2][i]));
                    h_state.R_d0[j-2][i] = Rd20;
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd20));
                    std::vector<double> dW = vector_scale(-a3 * dt_loc, Res);
                    h_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            h_state.run_odd();

            // Stage 4
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd20 = h_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd20));
                    std::vector<double> dW = vector_scale(-a4 * dt_loc, Res);
                    h_state.W[j][i] = vector_add(W_0[j][i], dW);
                }
            }
            h_state.run_even();

            // Stage 5, Final update
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd42 = vector_add(vector_scale(b5, h_state.R_d[j-2][i]), vector_scale(1-b5, h_state.R_d0[j-2][i]));
                    h_state.R_d0[j-2][i] = Rd42;
                    std::vector<double> Res = vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd42);
                    std::vector<double> dW = vector_scale(-a5 * dt_loc / h_state.OMEGA[j][i], Res);
                    h_state.W[j][i] = vector_add(W_0[j][i], dW);

                    all_Res[j - 2][i] = Res;
                    all_dw[j - 2][i] = dW;
                    q[j - 2][i] = h_state.W[j][i];
                    // if (j==2) {
                    //     std::cout << i << std::endl;
                    //     std::cout << dt << std::endl;
                    //     std::cout << dW[0] << " " << dW[1] << " " << dW[2] << " " << dW[3] << " " << std::endl;
                    //     std::cout << h_state.W[j][i][0] << " " << h_state.W[j][i][1] << " " << h_state.W[j][i][2] << " " << h_state.W[j][i][3] << " " << std::endl;
                    //     std::cout << std::endl;
                    // }
                }
            }
            h_state.run_odd();
        }
        else {
            std::vector d((ny - 4)*nx, std::vector<double>(4));

            auto dt = Multigrid::compute_dt(h_state);
            std::vector<std::vector<std::vector<double>>> W_0 = h_state.W;
            // Stage 1
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = h_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd0));
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
                    h_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            h_state.run_odd();

            // Stage 2
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd0 = h_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd0));
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
                    h_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            h_state.run_even();

            // Stage 3
            // #pragma omp parallel for
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd20 = vector_add(vector_scale(b3, h_state.R_d[j-2][i]), vector_scale(1-b3, h_state.R_d0[j-2][i]));
                    h_state.R_d0[j-2][i] = Rd20;
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd20));
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
                    h_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            h_state.run_odd();

            // Stage 4
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double>& Rd20 = h_state.R_d0[j-2][i];
                    std::vector<double> Res = vector_scale(1/h_state.OMEGA[j][i], vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]), Rd20));
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
                    h_state.W[j][i] = vector_subtract(W_0[j][i], dW);
                }
            }
            h_state.run_even();

            // Stage 5, Final update
            #pragma omp parallel for
            for (int j = 2; j < ny - 2; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double dt_loc = dt[j][i];
                    const std::vector<double> Rd42 = vector_add(vector_scale(b5, h_state.R_d[j-2][i]), vector_scale(1-b5, h_state.R_d0[j-2][i]));
                    h_state.R_d0[j-2][i] = Rd42;
                    std::vector<double> Res = vector_subtract(vector_add(h_state.R_c[j - 2][i], h_state.forcing_function[j-2][i]) , Rd42);
                    std::vector<double> dW = vector_scale(dt_loc / h_state.OMEGA[j][i], Res);
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
                    h_state.W[j][i] = vector_subtract(W_0[j][i], dW);

                    all_dw[j - 2][i] = dW;
                    q[j - 2][i] = h_state.W[j][i];
                }
            }
            h_state.run_odd();
        }



        // Compute L2 norm (placeholder logic)
        std::vector<double> l2_norm = compute_L2_norm(all_dw);

        if (it == 0) {
            first_residual = l2_norm;
        }
        normalized_residuals = vector_divide(l2_norm, first_residual);
        iteration.push_back(it);
        Residuals.push_back(l2_norm);

        auto [C_L, C_D, C_M] = compute_coeff(h_state);

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

    std::vector<std::vector<std::vector<double>>> q_cell_dummy(ny - 2, std::vector<std::vector<double>>(nx, std::vector<double>(4, 1.0)));

    // Compute q_vertex
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            q_cell_dummy[j - 1][i] = h_state.W[j][i];
        }
    }
    std::vector<std::vector<std::vector<double>>> q_vertex = cell_dummy_to_vertex_centered_airfoil(q_cell_dummy);

    // Corse grid correction
    // #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            h_state.deltaW_2h[j][i] = vector_subtract(h_state.W[j][i], W_2h_0[j][i]);
            // std::cout << "j: " << j << " i: " << i << " deltaW_2h: " << h_state.deltaW_2h[j][i][0] << " " << h_state.deltaW_2h[j][i][1] << " " << h_state.deltaW_2h[j][i][2] << " " << h_state.deltaW_2h[j][i][3] << std::endl;
        }
    }


    return {q_vertex, Residuals};
}

// Prolongation implementation (coarse to fine grid)
void Multigrid::prolongation(SpatialDiscretization& h2_state, SpatialDiscretization& h_state) {
    std::cout << "Prolongation" << std::endl;
    int ny_cell_2h = h2_state.ny - 1;
    int nx_cell_2h = h2_state.nx - 1;
    std::cout << "ny_cell_2h: " << ny_cell_2h << " nx_cell_2h: " << nx_cell_2h << std::endl;

    int nx = h_state.nx;
    int ny = h_state.ny;

    std::cout << "nx: " << nx << " ny: " << ny << std::endl;
    // std::cout << "prolongation_operator size: " << h_state.prolongation_operator.size() << " " << h_state.prolongation_operator[0].size() << std::endl;
    // std::cout << "deltaW_2h size: " << h2_state.deltaW_2h.size() << " " << h2_state.deltaW_2h[0].size() << std::endl;
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
            // std::cout << "deltaW_2h: " << h2_state.deltaW_2h[j_I][i_I][0] << " " << h2_state.deltaW_2h[j_I][i_I][1] << " " << h2_state.deltaW_2h[j_I][i_I][2] << " " << h2_state.deltaW_2h[j_I][i_I][3] << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jm1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Im1]), h2_state.deltaW_2h[j_Jm1][i_Im1])));
            // std::cout << "prolongation_operator: " << h_state.prolongation_operator[j-2][i][0] << " " << h_state.prolongation_operator[j-2][i][1] << " " << h_state.prolongation_operator[j-2][i][2] << " " << h_state.prolongation_operator[j-2][i][3] << std::endl;
            // std::cout << " " << std::endl;
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
            // std::cout << "deltaW_2h: " << h2_state.deltaW_2h[j_I][i_I][0] << " " << h2_state.deltaW_2h[j_I][i_I][1] << " " << h2_state.deltaW_2h[j_I][i_I][2] << " " << h2_state.deltaW_2h[j_I][i_I][3] << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jm1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Ip1]), h2_state.deltaW_2h[j_Jm1][i_Ip1])));
            // std::cout << "prolongation_operator: " << h_state.prolongation_operator[j-2][i][0] << " " << h_state.prolongation_operator[j-2][i][1] << " " << h_state.prolongation_operator[j-2][i][2] << " " << h_state.prolongation_operator[j-2][i][3] << std::endl;
            // std::cout << " " << std::endl;
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
            // std::cout << "deltaW_2h: " << h2_state.deltaW_2h[j_I][i_I][0] << " " << h2_state.deltaW_2h[j_I][i_I][1] << " " << h2_state.deltaW_2h[j_I][i_I][2] << " " << h2_state.deltaW_2h[j_I][i_I][3] << std::endl;
                        h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jp1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Im1]), h2_state.deltaW_2h[j_Jp1][i_Im1])));
            // std::cout << "prolongation_operator: " << h_state.prolongation_operator[j-2][i][0] << " " << h_state.prolongation_operator[j-2][i][1] << " " << h_state.prolongation_operator[j-2][i][2] << " " << h_state.prolongation_operator[j-2][i][3] << std::endl;
            // std::cout << " " << std::endl;
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
            // std::cout << "deltaW_2h: " << h2_state.deltaW_2h[j_I][i_I][0] << " " << h2_state.deltaW_2h[j_I][i_I][1] << " " << h2_state.deltaW_2h[j_I][i_I][2] << " " << h2_state.deltaW_2h[j_I][i_I][3] << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jp1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Ip1]), h2_state.deltaW_2h[j_Jp1][i_Ip1])));
            // std::cout << "prolongation_operator: " << h_state.prolongation_operator[j-2][i][0] << " " << h_state.prolongation_operator[j-2][i][1] << " " << h_state.prolongation_operator[j-2][i][2] << " " << h_state.prolongation_operator[j-2][i][3] << std::endl;
            // std::cout << " " << std::endl;
        }
    }

    // Compute W_h_+
    for (int j = 2; j < ny - 1 + 2; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            // std::cout << "j: " << j << " i: " << i;
            h_state.W[j][i] = vector_add(h_state.W[j][i], h_state.prolongation_operator[j-2][i]);
            // std::cout << "  W: " << h_state.W[j][i][0] << " " << h_state.W[j][i][1] << " " << h_state.W[j][i][2] << " " << h_state.W[j][i][3] << std::endl;
        }
    }


    Multigrid::h_state = h_state;
    
}


