#include "SpatialDiscretization.h"
#include "vector_helper.h"

#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>

template <typename T>
T combineBoundaryValues(const T& solidWall, const T& interior, const T& farfield) {
    T combined;
    combined.reserve(2 * solidWall.size() + interior.size() + 2 * farfield.size());

    // Insert `solidWall` twice
    combined.insert(combined.end(), solidWall.begin(), solidWall.end());
    combined.insert(combined.end(), solidWall.begin(), solidWall.end());

    // Insert `interior`
    combined.insert(combined.end(), interior.begin(), interior.end());

    // Insert `farfield` twice
    combined.insert(combined.end(), farfield.begin(), farfield.end());
    combined.insert(combined.end(), farfield.begin(), farfield.end());

    return combined;
}


SpatialDiscretization::SpatialDiscretization(const std::vector<std::vector<double>>& x,
                          const std::vector<std::vector<double>>& y,
                          double rho,
                          double u,
                          double v,
                          double E,
                          double T,
                          double p,
                          double k2_coeff,
                          double k4_coeff,
                          double T_ref,
                          double U_ref)
    : x(x), y(y), rho(rho), u(u), v(v), E(E), T(T), p(p), k2_coeff(k2_coeff), k4_coeff(k4_coeff), T_ref(T_ref), U_ref(U_ref) {
    ny = y.size();
    nx = x[0].size();
    alpha = std::atan2(v, u);

    std::vector OMEGA_domain(ny - 1, std::vector<double>(nx - 1));
    std::vector s_domain(ny - 1, std::vector(nx - 1, std::vector(2, std::vector<double>(2))));
    std::vector Ds_domain(ny - 1, std::vector(nx - 1, std::vector<double>(2)));
    std::vector n_domain(ny - 1, std::vector(nx - 1, std::vector(2, std::vector<double>(2))));
    std::vector W_domain(ny - 1, std::vector(nx - 1, std::vector<double>(4)));

    R_c.resize(ny - 1, std::vector(nx - 1, std::vector<double>(4)));
    R_d.resize(ny - 1, std::vector(nx - 1, std::vector<double>(4)));
    R_d0.resize(ny - 1, std::vector(nx - 1, std::vector<double>(4)));
    restriction_operator.resize(ny - 1, std::vector(nx - 1, std::vector<double>(4)));
    forcing_function.resize(ny - 1, std::vector(nx - 1, std::vector<double>(4)));
    prolongation_operator.resize(ny - 1, std::vector(nx - 1, std::vector<double>(4)));
        
    flux.resize(ny - 1 + 4, std::vector(nx - 1, std::vector(2, std::vector<double>(4))));
    D.resize(ny - 1 + 4, std::vector(nx - 1, std::vector(2, std::vector<double>(4))));
    eps_2.resize(ny - 1 + 4, std::vector(nx - 1, std::vector<double>(2)));
    eps_4.resize(ny - 1 + 4, std::vector(nx - 1, std::vector<double>(2)));
    Lambda_I.resize(ny - 1 + 4, std::vector<double>(nx - 1));
    Lambda_J.resize(ny - 1 + 4, std::vector<double>(nx - 1));
    Lambda_S.resize(ny - 1 + 4, std::vector(nx - 1, std::vector<double>(4)));

#pragma omp parallel for
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            // int thread_num = omp_get_thread_num();
            //
            // // Print thread number and the current index (iteration of the loop)
            // std::cout << "Thread " << thread_num << " processing index " << i << std::endl;
            //
            // int num_threads = omp_get_num_threads();
            // // Print the number of threads used and the current thread number
            // if (i == 0) {
            //     // Print the number of threads only once (at the first iteration)
            //     std::cout << "Number of threads: " << num_threads << std::endl;
            // }
            const double &x1 = x[j][i];
            const double &x2 = x[j][i + 1];
            const double &x3 = x[j + 1][i + 1];
            const double &x4 = x[j + 1][i];
            const double &y1 = y[j][i];
            const double &y2 = y[j][i + 1];
            const double &y3 = y[j + 1][i + 1];
            const double &y4 = y[j + 1][i];

            // Calculate OMEGA
            OMEGA_domain[j][i] = 0.5 * ((x1 - x3) * (y2 - y4) + (x4 - x2) * (y1 - y3));

            // Set s and compute Ds using s values
            s_domain[j][i][0] = {y2 - y1, x1 - x2};
            s_domain[j][i][1] = {y1 - y4, x4 - x1};

            // Length of s vectors
            Ds_domain[j][i][0] = std::hypot(s_domain[j][i][0][0], s_domain[j][i][0][1]);
            Ds_domain[j][i][1] = std::hypot(s_domain[j][i][1][0], s_domain[j][i][1][1]);

            // Normal vectors
            n_domain[j][i][0] = {s_domain[j][i][0][0] / Ds_domain[j][i][0], s_domain[j][i][0][1] / Ds_domain[j][i][0]};
            n_domain[j][i][1] = {s_domain[j][i][1][0] / Ds_domain[j][i][1], s_domain[j][i][1][1] / Ds_domain[j][i][1]};

            // Compute W
            W_domain[j][i] = {rho, rho * u, rho * v, rho * E};
        }
    }

    std::vector OMEGA_solidWall(OMEGA_domain.begin(), OMEGA_domain.begin() + 1);
    std::vector OMEGA_farfield(OMEGA_domain.end() - 1, OMEGA_domain.end());

    std::vector s_solidWall(s_domain.begin(), s_domain.begin() + 1);
    std::vector s_farfield(s_domain.end() - 1, s_domain.end());
    std::vector Ds_solidWall(Ds_domain.begin(), Ds_domain.begin() + 1);
    std::vector Ds_farfield(Ds_domain.end() - 1, Ds_domain.end());
    std::vector n_solidWall(n_domain.begin(), n_domain.begin() + 1);
    std::vector n_farfield(n_domain.end() - 1, n_domain.end());
    std::vector W_solidWall(W_domain.begin(), W_domain.begin() + 1);
    std::vector W_farfield(W_domain.end() - 1, W_domain.end());

    for (int i = 0; i < nx - 1; ++i) {
        const double &x1 = x[ny - 1][i];
        const double &x2 = x[ny - 1][i + 1];
        const double &y1 = y[ny - 1][i];
        const double &y2 = y[ny - 1][i + 1];

        // Set s and compute Ds using s values
        s_farfield[0][i][0] = {y2 - y1, x1 - x2};

        // Length of s vectors
        Ds_farfield[0][i][0] = std::hypot(s_farfield[0][i][0][0], s_farfield[0][i][0][1]);

        // Normal vectors
        n_farfield[0][i][0] = {
            s_farfield[0][i][0][0] / Ds_farfield[0][i][0], s_farfield[0][i][0][1] / Ds_farfield[0][i][0]
        };

        W_farfield[0][i] = {rho, rho * u, rho * v, rho * E};
    }

    OMEGA.resize(ny - 1 + 4, std::vector<double>(nx - 1));
    s.resize(ny - 1 + 4, std::vector(nx - 1, std::vector(2, std::vector<double>(2))));
    Ds.resize(ny - 1 + 4, std::vector(nx - 1, std::vector<double>(2)));
    n.resize(ny - 1 + 4, std::vector(nx - 1, std::vector(2, std::vector<double>(2))));
    W.resize(ny - 1 + 4, std::vector(nx - 1, std::vector<double>(4)));
    deltaW_2h.resize(ny - 1 + 4, std::vector(nx - 1, std::vector<double>(4)));

    // Combine using the helper function
    OMEGA = combineBoundaryValues(OMEGA_solidWall, OMEGA_domain, OMEGA_farfield);
    s = combineBoundaryValues(s_solidWall, s_domain, s_farfield);
    Ds = combineBoundaryValues(Ds_solidWall, Ds_domain, Ds_farfield);
    n = combineBoundaryValues(n_solidWall, n_domain, n_farfield);
    W = combineBoundaryValues(W_solidWall, W_domain, W_farfield);

    std::cout << "end" << std::endl;
}

void SpatialDiscretization::compute_dummy_cells() {
    // Solid wall
    // #pragma omp parallel for
    for (int i = 0; i < nx - 1; ++i) {

        auto [rho_val, u_val, v_val, E_val, T_val, p2] = conservative_variable_from_W(W[2][i]);
        std::vector<double> n1 = n[2][i][0];
        double V = n1[0]*u_val + n1[1]*v_val;
        double u_dummy = u_val - 2*V*n1[0];
        double v_dummy = v_val - 2*V*n1[1];
        W[0][i] = {rho_val, rho_val * u_dummy, rho_val * v_dummy, rho_val * E_val};
        W[1][i] = {rho_val, rho_val * u_dummy, rho_val * v_dummy, rho_val * E_val};


    }

    // Farfield
    // #pragma omp parallel for
    for (int i = 0; i < nx - 1; ++i) {
        auto [rho_d, u_d, v_d, E_d, T_d, p_d] = conservative_variable_from_W(W[W.size()-3][i]);
        const double c = std::sqrt(1.4 * p_d / rho_d);
        const double M = std::sqrt(u_d * u_d + v_d * v_d) / c;
        std::vector<double> n3 = vector_scale(-1, n[n.size()-2][i][0]);

        if (u_d * n3[0] + v_d * n3[1] > 0) { // Out of cell
            if (M >= 1) {
                W[W.size()-2][i] = {rho_d, rho_d * u_d, rho_d * v_d, rho_d * E_d};
                W[W.size()-1][i] = {rho_d, rho_d * u_d, rho_d * v_d, rho_d * E_d};

            }
            else {  // Subsonic
                const double p_b = this->p;  // Boundary pressure
                const double rho_b = rho_d + (p_b - p_d) / (c * c);
                const double u_b = u_d + n3[0] * (p_d - p_b) / (rho_d * c);
                const double v_b = v_d + n3[1] * (p_d - p_b) / (rho_d * c);
                const double E_b = p_b / ((1.4 - 1) * rho_b) + 0.5 * (u_b * u_b + v_b * v_b);

                std::vector<double> W_b = {rho_b, rho_b * u_b, rho_b * v_b, rho_b * E_b};
                std::vector<double> W_a = vector_subtract(vector_scale(2, W_b), W[W.size()-3][i]);

                W[W.size()-2][i] = W_a;
                W[W.size()-1][i] = W_a;

            }
        }
        else {  // Moving into the cell
            if (M >= 1) {  // Supersonic

                std::vector<double> W_a = {this->rho, this->rho * this->u, this->rho * this->v, this->rho * this->E};

                W[W.size()-2][i] = vector_subtract(vector_scale(2, W_a), W[W.size()-3][i]);
                W[W.size()-1][i] = vector_subtract(vector_scale(2, W_a), W[W.size()-3][i]);

            } else {  // Subsonic
                const double p_b = 0.5 * (this->p + p_d - rho_d * c * (n3[0] * (this->u - u_d) + n3[1] * (this->v - v_d)));
                const double rho_b = this->rho + (p_b - this->p) / (c * c);
                const double u_b = this->u - n3[0] * (this->p - p_b) / (rho_d * c);
                const double v_b = this->v - n3[1] * (this->p - p_b) / (rho_d * c);
                const double E_b = p_b / ((1.4 - 1) * rho_b) + 0.5 * (u_b * u_b + v_b * v_b);

                std::vector<double> W_b = {rho_b, rho_b * u_b, rho_b * v_b, rho_b * E_b};
                std::vector<double> W_a = vector_subtract(vector_scale(2, W_b), W[W.size()-3][i]);

                W[W.size()-2][i] = W_a;
                W[W.size()-1][i] = W_a;
            }
        }
    }
}

// Define the conservative_variable_from_W function as per your requirements
std::tuple<double, double, double, double, double, double> SpatialDiscretization::conservative_variable_from_W(const std::vector<double>& W) const {
    // Implement the conversion from W to (rho, u, v, E)
    double rho = W[0];
    double u = W[1] / rho;
    double v = W[2] / rho;
    double E = W[3] / rho;
    double qq = u*u+v*v;
    double p = (1.4-1)*rho*(E-qq/2);
    double T = p/(rho*287)*U_ref*U_ref/T_ref;
    return std::make_tuple(rho, u, v, E, T, p);
}

std::vector<double> SpatialDiscretization::FcDs(const std::vector<double>& W, const std::vector<double>& n, const double& Ds) const {
    auto [rho, u, v, E, T, p] = conservative_variable_from_W(W);
    double V = n[0]*u + n[1]*v;
    double H = E + p/rho;

    return {rho*V*Ds, (rho*u*V + n[0]*p)*Ds, (rho*v*V + n[1]*p)*Ds, rho*H*V*Ds};
}

double SpatialDiscretization::Lambdac(const std::vector<double>& W, const std::vector<double>& n, const double& Ds) const {
    auto [rho, u, v, E, T, p] = conservative_variable_from_W(W);
    double c = std::sqrt(1.4*p/rho);
    const double V = n[0]*u + n[1]*v;
    const double lambda = (std::abs(V) + c)*Ds;

    return lambda;
}

void SpatialDiscretization::compute_Fc_DeltaS() {
    const auto ny = W.size();
    const auto nx = W[0].size();

    #pragma omp parallel for
    for (size_t j = 2; j < ny - 1; ++j) {
        for (size_t i = 0; i < nx; ++i) {

            std::vector<double> avg_W1 = vector_scale(0.5, vector_add(W[j][i], W[j - 1][i]));
            std::vector<double> avg_W4 = vector_scale(0.5, vector_add(W[j][i], W[j][(i - 1 + nx) % nx]));

            std::vector<double> FcDs_1 = FcDs(avg_W1, n[j][i][0], Ds[j][i][0]);
            std::vector<double> FcDs_4 = FcDs(avg_W4, n[j][i][1], Ds[j][i][1]);

            flux[j][i][0] = FcDs_1;
            flux[j][i][1] = FcDs_4;
        }
    }

}

std::tuple<double, double> SpatialDiscretization::compute_epsilon(const std::vector<double>& W_Im1, const std::vector<double>& W_I,
                                                                 const std::vector<double>& W_Ip1, const std::vector<double>& W_Ip2,
                                                                 double k2, double k4) const {
    // Retrieve pressure from the conservative variables (assuming the last element is pressure)
    double p_Im1, p_I, p_Ip1, p_Ip2;
    std::tie(std::ignore, std::ignore,std::ignore, std::ignore, std::ignore, p_Im1) = conservative_variable_from_W(W_Im1);
    std::tie(std::ignore, std::ignore,std::ignore, std::ignore, std::ignore, p_I) = conservative_variable_from_W(W_I);
    std::tie(std::ignore, std::ignore,std::ignore, std::ignore, std::ignore, p_Ip1) = conservative_variable_from_W(W_Ip1);
    std::tie(std::ignore, std::ignore,std::ignore, std::ignore, std::ignore, p_Ip2) = conservative_variable_from_W(W_Ip2);

    // Calculate Gamma_I and Gamma_Ip1
    double Gamma_I = std::abs(p_Ip1 - 2.0 * p_I + p_Im1) / (p_Ip1 + 2.0 * p_I + p_Im1);
    double Gamma_Ip1 = std::abs(p_Ip2 - 2.0 * p_Ip1 + p_I) / (p_Ip2 + 2.0 * p_Ip1 + p_I);

    // Compute eps2 and eps4
    double eps2 = k2 * std::max(Gamma_I, Gamma_Ip1);
    double eps4 = std::max(0.0, k4 - eps2);

    // Return the results as a pair
    return std::make_tuple(eps2, eps4);
}

void SpatialDiscretization::compute_lambda() {

    const auto ny = W.size();
    const auto nx = W[0].size();

    #pragma omp parallel for
    for (size_t j = 0; j < ny-1; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            // Spectral radius in i-direction
            const double sx = 0.5*(s[j][i][1][0] + s[j][(i + 1) % nx][1][0]);
            const double sy = 0.5*(s[j][i][1][1] + s[j][(i + 1) % nx][1][1]);
            auto [rho, u, v, E, T, p] = conservative_variable_from_W(W[j][i]);
            const double cc = 1.4*p/rho;
            double u_dot_n = u*sx + v*sy;
            const double speci = std::abs(u_dot_n) + std::sqrt(cc*(sx*sx+sy*sy));
            Lambda_I[j][i] = speci;
        }
    }
    #pragma omp parallel for
    for (size_t j = 0; j < ny-1; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            // Spectral radius in j-direction
            const double sx = 0.5*(s[j][i][0][0] + s[j+1][i][0][0]);
            const double sy = 0.5*(s[j][i][0][1] + s[j+1][i][0][1]);
            auto [rho, u, v, E, T, p] = conservative_variable_from_W(W[j][i]);
            const double cc = 1.4*p/rho;
            double u_dot_n = u*sx + v*sy;
            const double specj = std::abs(u_dot_n) + std::sqrt(cc*(sx*sx+sy*sy));
            Lambda_J[j][i] = specj;
        }
    }
}

void SpatialDiscretization::compute_dissipation() {
    const auto ny = W.size();
    const auto nx = W[0].size();

    #pragma omp parallel for
    for (size_t j = 2; j < ny - 1; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            std::vector<double>& W_IJ = W[j][i];
            std::vector<double>& W_Ip1J = W[j][(i + 1) % nx];
            std::vector<double>& W_IJp1 = W[j + 1][i];
            std::vector<double>& W_Im1J = W[j][(i - 1 + nx) % nx];
            std::vector<double>& W_IJm1 = W[j - 1][i];
            std::vector<double>& W_Im2J = W[j][(i - 2 + nx) % nx];
            std::vector<double>& W_IJm2 = W[j - 2][i];

            // Lambda calculations

            const double Lambda_4_I = 0.5 * (Lambda_I[j][i] + Lambda_I[j][(i - 1 + nx) % nx]);
            const double Lambda_4_J = 0.5 * (Lambda_J[j][i] + Lambda_J[j][(i - 1 + nx) % nx]);
            const double Lambda_4_S = Lambda_4_I + Lambda_4_J;

            const double Lambda_1_J = 0.5 * (Lambda_J[j][i] + Lambda_J[j - 1][i]);
            const double Lambda_1_I = 0.5 * (Lambda_I[j][i] + Lambda_I[j - 1][i]);
            const double Lambda_1_S = Lambda_1_I + Lambda_1_J;

            Lambda_S[j][i][0] = Lambda_1_S;
            Lambda_S[j][i][3] = Lambda_4_S;


            // Epsilon calculations
            auto[eps2_4, eps4_4] = compute_epsilon(W_Ip1J, W_IJ, W_Im1J, W_Im2J, k2_coeff*0.25, k4_coeff*1/64);
            auto[eps2_1, eps4_1] = compute_epsilon(W_IJp1, W_IJ, W_IJm1, W_IJm2, k2_coeff*0.25, k4_coeff*1/64);

            eps_2[j-2][i][0] = eps2_1;
            eps_2[j-2][i][1] = eps2_4;
            eps_4[j-2][i][0] = eps4_1;
            eps_4[j-2][i][1] = eps4_4;

            // Dissipation terms
            std::vector<double> D_1 = vector_scale(Lambda_1_S,
               vector_subtract(
                   vector_scale(eps2_1, vector_subtract(W_IJm1, W_IJ)),
                   vector_scale(eps4_1,
                       vector_add(
                            vector_subtract(W_IJm2, vector_scale(3, W_IJm1)),
                            vector_subtract(vector_scale(3, W_IJ),W_IJp1
                       )
                   ))
               )
            );

            std::vector<double> D_4 = vector_scale(Lambda_4_S,
                vector_subtract(
                    vector_scale(eps2_4, vector_subtract(W_Im1J, W_IJ)),
                    vector_scale(eps4_4,
                        vector_add(
                            vector_subtract(W_Im2J, vector_scale(3, W_Im1J)),
                            vector_subtract(vector_scale(3, W_IJ),W_Ip1J
                        )
                    ))
                )
            );

            D[j][i][0] = D_1;
            D[j][i][1] = D_4;

        }
    }

    // Boundary conditions
    // #pragma omp parallel for
    for (size_t i = 0; i < nx; ++i) {

        // Calculate D_1 for cell (3, i)
        D[3][i][0] = vector_scale(Lambda_S[3][i][0],
            vector_subtract(
                vector_scale(eps_2[3][i][0], vector_subtract(W[2][i], W[3][i])),

                vector_scale(eps_4[3][i][0], vector_subtract(vector_subtract(
                    vector_scale(2.0, W[3][i]), W[2][i]), W[4][i]))
            )
        );

        // Calculate D_1 for cell (2, i)
        D[2][i][0] = vector_scale(Lambda_S[2][i][0],
            vector_subtract(
                vector_scale(eps_2[2][i][0], vector_subtract(W[2][i], W[3][i])),

                vector_scale(eps_4[2][i][0], vector_subtract(vector_subtract(
                    vector_scale(2.0, W[3][i]), W[2][i]), W[4][i]))
            )
    );
    }
}

void SpatialDiscretization::compute_R_c() {
    const auto ny = W.size();
    const auto nx = W[0].size();

    #pragma omp parallel for
    for (size_t j = 2; j < ny - 2; ++j) {
        for (size_t i = 0; i < nx; ++i) {

            // Extract the flux vectors
            const std::vector<double>& FcDS_1 = flux[j][i][0];
            const std::vector<double>& FcDS_2 = vector_scale(-1, flux[j][(i + 1) % nx][1]);
            const std::vector<double>& FcDS_3 = vector_scale(-1, flux[j+1][i][0]);
            const std::vector<double>& FcDS_4 = flux[j][i][1];


            R_c[j-2][i] = vector_add(vector_add(FcDS_1, FcDS_2), vector_add(FcDS_3, FcDS_4));

        }
    }
}

void SpatialDiscretization::compute_R_d() {
    const auto ny = W.size();
    const auto nx = W[0].size();

    #pragma omp parallel for
    for (size_t j = 2; j < ny - 2; ++j) {
        for (size_t i = 0; i < nx; ++i) {

            // Extract the dissipation vectors

            const std::vector<double>& D_1 = D[j][i][0];
            const std::vector<double>& D_2 = vector_scale(-1, D[j][(i + 1) % nx][1]);
            const std::vector<double>& D_3 = vector_scale(-1, D[j+1][i][0]);
            const std::vector<double>& D_4 = D[j][i][1];


            R_d[j-2][i] = vector_add(vector_add(D_1, D_2), vector_add(D_3, D_4));

        }
    }
}

void SpatialDiscretization::run_odd() {
    SpatialDiscretization::compute_dummy_cells();
    SpatialDiscretization::compute_lambda();
    SpatialDiscretization::compute_Fc_DeltaS();
    SpatialDiscretization::compute_R_c();
}

void SpatialDiscretization::run_even() {
    SpatialDiscretization::compute_dummy_cells();
    SpatialDiscretization::compute_lambda();
    SpatialDiscretization::compute_Fc_DeltaS();
    SpatialDiscretization::compute_dissipation();
    SpatialDiscretization::compute_R_c();
    SpatialDiscretization::compute_R_d();
}

