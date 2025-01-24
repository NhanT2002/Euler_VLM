#include "TemporalDiscretization.h"
#include "vector_helper.h"
#include "read_PLOT3D.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <omp.h>

// https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
std::vector<std::vector<double>> thomasAlgorithm(std::vector<double>& a, // subdiagonal
                                                 const std::vector<double>& b, // main diagonal
                                                 std::vector<double>& c, // superdiagonal
                                                 const std::vector<std::vector<double>>& d) {  // right hand side
    int n = b.size();
    int numRHS = d[0].size();

    // a[0] = 0.0;
    // c[n-1] = 0.0;

    // Initialize modified vectors
    std::vector<double> cp(n, 0.0);                          // Modified super-diagonal
    std::vector<std::vector<double>> dp(n, std::vector<double>(numRHS, 0.0));  // Modified right-hand side
    std::vector<std::vector<double>> x(n, std::vector<double>(numRHS, 0.0));   // Solution vector

    // Forward elimination for each right-hand side
    cp[0] = c[0] / b[0];
    for (int j = 0; j < numRHS; j++) {
        dp[0][j] = d[0][j] / b[0];
    }

    #pragma omp parallel for
    for (int i = 1; i < n; i++) {
        double m = b[i] - a[i] * cp[i - 1];
        cp[i] = c[i] / m;
        for (int j = 0; j < numRHS; j++) {
            dp[i][j] = (d[i][j] - a[i] * dp[i-1][j]) / m;
        }
    }

    #pragma omp parallel for
    // Back substitution for each right-hand side
    for (int j = 0; j < numRHS; j++) {
        x[n-1][j] = dp[n-1][j];
        for (int i = n - 2; i >= 0; i--) {
            x[i][j] = dp[i][j] - cp[i] * x[i+1][j];
        }
    }

    return x;
}

std::vector<std::vector<double>> reshapeColumnWise(
    const std::vector<std::vector<double>>& input,
    int ny, int nx)
{
    // Check input dimensions
    if (input.size() != ny * nx) {
        throw std::invalid_argument("Input size does not match ny * nx.");
    }

    int depth = input[0].size(); // Assuming 4 in this case (nx * ny, 4)

    // Initialize reshaped array
    std::vector<std::vector<double>> reshaped(ny * nx, std::vector<double>(depth));

    // Perform column-wise reshaping
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                reshaped[i * nx + j][d] = input[j * ny + i][d];
            }
        }
    }

    return reshaped;
}

TemporalDiscretization::TemporalDiscretization(const std::vector<std::vector<double>>& x,
                                               const std::vector<std::vector<double>>& y,
                                               const double& rho,
                                               const double& u,
                                               const double& v,
                                               const double& E,
                                               const double& T,
                                               const double& p,
                                               const double& T_ref,
                                               const double& U_ref,
                                               const double sigma,
                                               const int res_smoothing,
                                               const double k2_coeff,
                                               const double k4_coeff)
    : x(x), y(y), rho(rho), u(u), v(v), E(E), T(T), p(p), T_ref(T_ref), U_ref(U_ref), sigma(sigma), res_smoothing(res_smoothing), k2_coeff(k2_coeff), k4_coeff(k4_coeff),
      current_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, T_ref, U_ref) {}

std::vector<std::vector<double>> TemporalDiscretization::compute_dt() const {
    auto ny = current_state.W.size();
    auto nx = current_state.W[0].size();

    std::vector dt(y.size() - 1 + 4, std::vector<double>(x.size() - 1));

    #pragma omp parallel for
    for (size_t j = 2; j < ny - 2; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            dt[j][i] = sigma*current_state.OMEGA[j][i]/(current_state.Lambda_I[j][i]+current_state.Lambda_J[j][i]);

        }
    }

    return dt;
}
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> TemporalDiscretization::compute_abc() const {
    auto ny = current_state.W.size();
    auto nx = current_state.W[0].size();

    std::vector<double> a_I((ny - 4)*nx);
    std::vector<double> b_I((ny - 4)*nx);
    std::vector<double> c_I((ny - 4)*nx);
    std::vector<double> a_J((ny - 4)*nx);
    std::vector<double> b_J((ny - 4)*nx);
    std::vector<double> c_J((ny - 4)*nx);

    double rr = 2;
    #pragma omp parallel for
    for (size_t j = 2; j < ny - 2; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            double r = current_state.Lambda_J[j][i]/current_state.Lambda_I[j][i];
            double eps_I = std::max(0.25*(std::pow(rr*(1 + std::sqrt(r))/(1+r), 2)-1), 0.0);

            if (i == 0) {
                a_I[(j-2)*nx + i] = 0.0;
            }
            else {
                a_I[(j-2)*nx + i] = -eps_I;
            }
            if (i == nx - 1) {
                c_I[(j-2)*nx + i] = 0.0;
            }
            else {
                c_I[(j-2)*nx + i] = -eps_I;
            }

            b_I[(j-2)*nx + i] = 1 + 2*eps_I;
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 2; j < ny - 2; ++j) {
            double r = current_state.Lambda_J[j][i]/current_state.Lambda_I[j][i];
            double eps_J = std::max(0.25*(std::pow(rr*(1 + std::sqrt(1/r))/(1+1/r), 2)-1), 0.0);

            if (j == 2) {
                a_J[i*nx + (j-2)] = 0.0;
            }
            else {
                a_J[i*nx + (j-2)] = -eps_J;
            }
            if (j == ny - 3) {
                c_J[i*nx + (j-2)] = 0.0;
            }
            else {
                c_J[i*nx + (j-2)] = -eps_J;
            }
            b_J[i*nx + (j-2)] = 1 + 2*eps_J;

        }
    }

    return std::make_tuple(a_I, b_I, c_I, a_J, b_J, c_J);
}


std::vector<double> TemporalDiscretization::compute_L2_norm(const std::vector<std::vector<std::vector<double>>> &residuals) {
    const auto m = residuals.size();          // Number of rows
    const auto n = residuals[0].size();       // Number of columns
    const auto num_components = residuals[0][0].size(); // Number of components (4)

    std::vector<double> l2_norms(num_components, 0.0);
    const auto N = static_cast<double>(m * n); // Total number of cells

    for (size_t k = 0; k < num_components; ++k) {
        double sum = 0.0;
        for (size_t j = 0; j < m; ++j) {
            for (size_t i = 0; i < n; ++i) {
                sum += residuals[j][i][k] * residuals[j][i][k]; // Squared component
            }
        }
        l2_norms[k] = std::sqrt(sum / N);
    }

    return l2_norms;
}

void TemporalDiscretization::save_checkpoint(const std::vector<std::vector<std::vector<double>>>& q,
                                             const std::vector<int>& iteration,
                                             const std::vector<std::vector<double>>& Residuals,
                                             const std::string& file_name) {
    // Open the file for writing
    std::ofstream file(file_name);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving checkpoint." << std::endl;
        return;
    }

    // Write the current iteration numbers
    file << "Iterations:\n";
    for (const auto& it : iteration) {
        file << it << " ";
    }
    file << "\n";

    // Get dimensions for q
    const auto nj = q.size();               // Number of rows (j)
    const auto ni = q[0].size();            // Number of columns (i)
    const auto nVars = q[0][0].size();      // Number of variables (e.g., density, momentum, energy)

    // Write the solution vector q
    file << "Solution vector (q):\n";
    file << ni << " " << nj << " " << nVars << "\n";  // Write dimensions for q

    // Write flow variables (density, x-momentum, y-momentum, energy) in row-major order
    for (size_t n = 0; n < nVars; ++n) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t i = 0; i < ni; ++i) {
                  // Iterate over the number of variables
                    file << std::scientific << std::setprecision(16) << q[j][i][n] << "\n";
                }
            }
    }

    // Write the residuals
    file << "Residuals (4D vectors):\n";
    for (const auto& res : Residuals) {
        for (const auto& val : res) {
            file << std::scientific << std::setprecision(16) << val << " ";  // Write each value in the residual vector
        }
        file << "\n";  // New line after each row of residuals
    }

    // Close the file
    file.close();
    std::cout << "Checkpoint saved to " << file_name << std::endl;
}


std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<int>,
           std::vector<std::vector<double>>> TemporalDiscretization::load_checkpoint(const std::string& file_name) {
    // Open the file for reading
    std::ifstream file(file_name);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file for loading checkpoint." << std::endl;
        return std::make_tuple(std::vector<std::vector<std::vector<double>>>(), std::vector<int>(), std::vector<std::vector<double>>());
    }

    std::string line;
    std::vector<int> iteration;
    std::vector<std::vector<std::vector<double>>> q;
    std::vector<std::vector<double>> Residuals;

    // Read iteration numbers
    std::getline(file, line);  // Read the header line
    std::getline(file, line);  // Read the iteration line
    std::istringstream iss(line);
    int it;
    while (iss >> it) {
        iteration.push_back(it);
    }

    // Read dimensions for q
    std::getline(file, line);  // Read the header for the solution vector
    std::getline(file, line);  // Read the dimensions line
    std::istringstream dimStream(line);
    int ni, nj, nVars;
    dimStream >> ni >> nj >> nVars;

    // Resize the solution vector q
    q.resize(nj, std::vector<std::vector<double>>(ni, std::vector<double>(nVars)));

    // Read the flow variables (density, x-momentum, y-momentum, energy) in row-major order
    for (int n = 0; n < nVars; ++n) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                double value;
                file >> value;  // Read the value directly
                q[j][i][n] = value;
            }
        }
    }

    // Read the residuals
    std::getline(file, line);  // Read the header for residuals
    std::getline(file, line);  // Read the residuals line
    while (std::getline(file, line)) {
        std::istringstream resStream(line);
        std::vector<double> resRow;
        double val;
        while (resStream >> val) {
            resRow.push_back(val);
        }
        Residuals.push_back(resRow);
    }

    // Close the file
    file.close();
    std::cout << "Checkpoint loaded from " << file_name << std::endl;

    return std::make_tuple(q, iteration, Residuals);
}

std::tuple<std::vector<std::vector<std::vector<double>>>,
           std::vector<std::vector<std::vector<double>>>,
           std::vector<std::vector<double>>> TemporalDiscretization::RungeKutta(int it_max) {

    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;

    int ny = current_state.W.size();
    int nx = current_state.W[0].size();
    std::cout << ny << " " << nx << std::endl;

    current_state.run_even();
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd0));
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd0));
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd20));
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd20));
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
                    std::vector<double> Res = vector_subtract(current_state.R_c[j-2][i], Rd42);
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd0));
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd0));
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd20));
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
                    std::vector<double> Res = vector_scale(1/current_state.OMEGA[j][i], vector_subtract(current_state.R_c[j - 2][i], Rd20));
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
                    std::vector<double> Res = vector_subtract(current_state.R_c[j-2][i], Rd42);
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

    std::vector<std::vector<std::vector<double>>> q_cell_dummy(ny - 2, std::vector<std::vector<double>>(nx, std::vector<double>(4, 1.0)));

    // Compute q_vertex
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            q_cell_dummy[j - 1][i] = current_state.W[j][i];
        }
    }
    std::vector<std::vector<std::vector<double>>> q_vertex = cell_dummy_to_vertex_centered_airfoil(q_cell_dummy);

    return std::make_tuple(q, q_vertex, Residuals);
}

std::tuple<double, double, double> TemporalDiscretization::compute_coeff() {
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;
    int nx = current_state.W[0].size();
    std::vector<double> p_array(nx);
    double Fx = 0.0;
    double Fy = 0.0;
    double M = 0.0;
    for (int i = 0; i < nx; ++i) {
        double& rho = current_state.W[2][i][0];
        double& rho_u = current_state.W[2][i][1];
        double& rho_v = current_state.W[2][i][2];
        double& rho_E = current_state.W[2][i][3];
        double p = (1.4-1)*(rho_E-0.5*(rho_u*rho_u+rho_v*rho_v)/rho);
        Fx += p*current_state.n[2][i][0][0]*current_state.Ds[2][i][0];
        Fy += p*current_state.n[2][i][0][1]*current_state.Ds[2][i][0];

        double x_mid = 0.5*(current_state.x[0][i] + current_state.x[0][i+1]);
        double y_mid = 0.5*(current_state.y[0][i] + current_state.y[0][i+1]);
        M += p*(-(x_mid-x_ref)*current_state.n[2][i][0][1] + (y_mid-y_ref)*current_state.n[2][i][0][0])*current_state.Ds[2][i][0];
    }

    double L = Fy*std::cos(current_state.alpha) - Fx*std::sin(current_state.alpha);
    double D = Fy*std::sin(current_state.alpha) + Fx*std::cos(current_state.alpha);

    double C_L = L/(0.5*rho*(u*u+v*v)*c);
    double C_D = D/(0.5*rho*(u*u+v*v)*c);
    double C_M = M/(0.5*rho*(u*u+v*v)*c*c);

    return std::tie<double>(C_L, C_D, C_M);
}

