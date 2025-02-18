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
#include <Eigen/Dense>
#include <chrono>

Eigen::VectorXd thomasAlgorithm(const Eigen::VectorXd& a, const Eigen::VectorXd& b, const Eigen::VectorXd& c, const Eigen::VectorXd& d) {

    // std::cout << "a\n" << a << std::endl;
    // std::cout << "b\n" << b << std::endl;
    // std::cout << "c\n" << c << std::endl;
    // std::cout << "d\n" << d << std::endl;
    int n = b.size();
    Eigen::VectorXd c_prime(n);
    Eigen::VectorXd d_prime(n);
    Eigen::VectorXd x(n, 1);

    c_prime(0) = c(0)/b(0);
    d_prime(0) = d(0)/b(0);

    for (int i = 1; i < n; i++) {
        double m = b(i) - a(i)*c_prime(i-1);
        c_prime(i) = c(i)/m;
        // std::cout << "cp[i]: " << c_prime(i) << " c(i): " << c(i) << " m: " << m << std::endl;
        d_prime(i) = (d(i) - a(i)*d_prime(i-1))/m;
    }
    // std::cout << "c_prime\n" << c_prime << std::endl;

    x(n-1) = d_prime(n-1);
    for (int i = n-2; i >= 0; i--) {
        x(i) = d_prime(i) - c_prime(i)*x(i+1);
        // std::cout << "x[i]: " << x(i) << " d_prime[i]: " << d_prime(i) << " c_prime[i]: " << c_prime(i) << std::endl;
    }

    return x;
}

TemporalDiscretization::TemporalDiscretization(Eigen::ArrayXXd& x,
                            Eigen::ArrayXXd& y,
                            double rho,
                            double u,
                            double v,
                            double E,
                            double T,
                            double p,
                            double Mach,
                            double U_ref,
                            double sigma,
                            int res_smoothing,
                            double k2_coeff,
                            double k4_coeff)
    : x(x),
      y(y),
      rho(rho),
      u(u),
      v(v),
      E(E),
      T(T),
      p(p),
      Mach(Mach),
      U_ref(U_ref),
      current_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, Mach, U_ref),
      sigma(sigma),
      k2_coeff(k2_coeff),
      k4_coeff(k4_coeff),
      res_smoothing(res_smoothing) {}

Eigen::ArrayXXd TemporalDiscretization::compute_dt() const {

    auto seqy = Eigen::seq(2, current_state.ncells_y-3);
    auto seqx = Eigen::seq(2, current_state.ncells_x-3);
    Eigen::ArrayXXd dt_array = sigma*current_state.OMEGA(seqy, seqx)/(current_state.Lambda_I(seqy, seqx) + current_state.Lambda_J(seqy, seqx));
    // std::cout << "dt_array\n" << dt_array << std::endl;

    return dt_array;
}

Eigen::Array<double, 4, 1> TemporalDiscretization::compute_L2_norm(const Eigen::ArrayXXd &dW_0, const Eigen::ArrayXXd &dW_1, const Eigen::ArrayXXd &dW_2, const Eigen::ArrayXXd &dW_3) {
    Eigen::Array<double, 4, 1> L2_norms;

    L2_norms(0) = std::sqrt(((dW_0*dW_0).sum()/dW_0.size()));
    L2_norms(1) = std::sqrt(((dW_1*dW_1).sum()/dW_1.size()));
    L2_norms(2) = std::sqrt(((dW_2*dW_2).sum()/dW_2.size()));
    L2_norms(3) = std::sqrt(((dW_3*dW_3).sum()/dW_3.size()));

    return L2_norms;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> TemporalDiscretization::compute_abc() {
    auto seqy = Eigen::seq(2, current_state.ncells_y-3);
    auto seqx = Eigen::seq(2, current_state.ncells_x-3);
    int n = (current_state.ncells_x-4)*(current_state.ncells_y-4);

    double rr = 2.0;
    Eigen::ArrayXXd r = current_state.Lambda_J(seqy, seqx)/current_state.Lambda_I(seqy, seqx);
    Eigen::ArrayXXd eps_I = (0.25*((rr*(1+r.sqrt())/(1+r)).square()-1)).max(0.0);
    Eigen::ArrayXXd eps_J = (0.25*((rr*(1+(1/r).sqrt())/(1+1/r)).square()-1)).max(0.0); 

    Eigen::ArrayXXd a_I = -eps_I;
    a_I.col(0) = 0.0;

    Eigen::ArrayXXd c_I = -eps_I;
    c_I.col(c_I.cols()-1) = 0.0;

    Eigen::ArrayXXd b_I = (1 + 2*eps_I);

    Eigen::ArrayXXd a_J = -eps_J;
    a_J.row(0) = 0.0;

    Eigen::ArrayXXd c_J = -eps_J;
    c_J.row(c_J.rows()-1) = 0.0;

    Eigen::ArrayXXd b_J = (1 + 2*eps_J);

    return {a_I.transpose().reshaped(), b_I.transpose().reshaped(), c_I.transpose().reshaped(), a_J.reshaped(), b_J.reshaped(), c_J.reshaped()};
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> TemporalDiscretization::R_star(const Eigen::ArrayXXd& dW_0, const Eigen::ArrayXXd& dW_1, const Eigen::ArrayXXd& dW_2, const Eigen::ArrayXXd& dW_3) {
    Eigen::VectorXd d_0, d_1, d_2, d_3;
    Eigen::VectorXd R_star_0, R_star_1, R_star_2, R_star_3;
    Eigen::VectorXd R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3;

    d_0 = dW_0.transpose().reshaped();
    d_1 = dW_1.transpose().reshaped();
    d_2 = dW_2.transpose().reshaped();
    d_3 = dW_3.transpose().reshaped();

    auto [a_I, b_I, c_I, a_J, b_J, c_J] = TemporalDiscretization::compute_abc();
    R_star_0 = thomasAlgorithm(a_I, b_I, c_I, d_0);
    R_star_1 = thomasAlgorithm(a_I, b_I, c_I, d_1);
    R_star_2 = thomasAlgorithm(a_I, b_I, c_I, d_2);
    R_star_3 = thomasAlgorithm(a_I, b_I, c_I, d_3);

    R_star_star_0 = thomasAlgorithm(a_J, b_J, c_J, R_star_0.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());
    R_star_star_1 = thomasAlgorithm(a_J, b_J, c_J, R_star_1.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());
    R_star_star_2 = thomasAlgorithm(a_J, b_J, c_J, R_star_2.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());
    R_star_star_3 = thomasAlgorithm(a_J, b_J, c_J, R_star_3.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());

    return {R_star_star_0.reshaped(dW_0.rows(), dW_0.cols()).array(), R_star_star_1.reshaped(dW_0.rows(), dW_0.cols()).array(), R_star_star_2.reshaped(dW_0.rows(), dW_0.cols()).array(), R_star_star_3.reshaped(dW_0.rows(), dW_0.cols()).array()};
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, std::vector<std::vector<double>>, std::vector<double>> TemporalDiscretization::RungeKutta(int it_max) {
    auto start = std::chrono::high_resolution_clock::now();

    double convergence_tol = 1e-11;
    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;

    Eigen::ArrayXXd dt;
    Eigen::ArrayXXd dW_0, dW_1, dW_2, dW_3;
    Eigen::ArrayXXd W0_0, W1_0, W2_0, W3_0;
    Eigen::ArrayXXd Res_0, Res_1, Res_2, Res_3;
    Eigen::ArrayXXd Rd20_0, Rd20_1, Rd20_2, Rd20_3;
    Eigen::ArrayXXd Rd42_0, Rd42_1, Rd42_2, Rd42_3;
    std::vector<std::vector<double>> Residuals;
    std::vector<int> iteration;
    
    Residuals = std::vector<std::vector<double>>{};
    iteration = std::vector<int>{};

    std::vector<double> iteration_times;
    iteration_times = std::vector<double>{};

    auto seqy = Eigen::seq(2, current_state.ncells_y-3);
    auto seqx = Eigen::seq(2, current_state.ncells_x-3);

    if (res_smoothing == 0) {
        for (int it = 0; it < it_max; it++) {
            current_state.run_even();
            // Initialize Rd0
            current_state.update_Rd0();

            W0_0 = current_state.W_0(seqy, seqx);
            W1_0 = current_state.W_1(seqy, seqx);
            W2_0 = current_state.W_2(seqy, seqx);
            W3_0 = current_state.W_3(seqy, seqx);            
            dt = compute_dt();

            // Stage 1
            Eigen::ArrayXXd& Rd0_0 = current_state.Rd_0;
            Eigen::ArrayXXd& Rd1_0 = current_state.Rd_1;
            Eigen::ArrayXXd& Rd2_0 = current_state.Rd_2;
            Eigen::ArrayXXd& Rd3_0 = current_state.Rd_3;

            dW_0 = -a1*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a1*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a1*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a1*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
                  
            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 2
            Rd0_0 = current_state.Rd_0;
            Rd1_0 = current_state.Rd_1;
            Rd2_0 = current_state.Rd_2;
            Rd3_0 = current_state.Rd_3;

            dW_0 = -a2*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a2*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a2*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a2*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
        
            current_state.update_conservative_variables();
            current_state.run_even();
            
            // Stage 3
            Rd20_0 = b3*current_state.Rd_0 + (1-b3)*current_state.Rd0_0;
            Rd20_1 = b3*current_state.Rd_1 + (1-b3)*current_state.Rd0_1;
            Rd20_2 = b3*current_state.Rd_2 + (1-b3)*current_state.Rd0_2;
            Rd20_3 = b3*current_state.Rd_3 + (1-b3)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd20_0;
            current_state.Rd0_1 = Rd20_1;
            current_state.Rd0_2 = Rd20_2;
            current_state.Rd0_3 = Rd20_3;

            dW_0 = -a3*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd20_0);
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a3*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd20_1);
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a3*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd20_2);
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a3*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd20_3);
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 4
            Rd20_0 = current_state.Rd0_0;
            Rd20_1 = current_state.Rd0_1;
            Rd20_2 = current_state.Rd0_2;
            Rd20_3 = current_state.Rd0_3;

            dW_0 = -a4*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd20_0);
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a4*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd20_1);
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a4*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd20_2);
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a4*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd20_3);
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_even();

            // Stage 5
            Rd42_0 = b5*current_state.Rd_0 + (1-b5)*current_state.Rd0_0;
            Rd42_1 = b5*current_state.Rd_1 + (1-b5)*current_state.Rd0_1;
            Rd42_2 = b5*current_state.Rd_2 + (1-b5)*current_state.Rd0_2;
            Rd42_3 = b5*current_state.Rd_3 + (1-b5)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd42_0;
            current_state.Rd0_1 = Rd42_1;
            current_state.Rd0_2 = Rd42_2;
            current_state.Rd0_3 = Rd42_3;

            dW_0 = -a5*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd42_0);
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a5*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd42_1);
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a5*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd42_2);
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a5*dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd42_3);
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            // current_state.run_odd();
            
            // std::cout << "stage 5\n";
            // std::cout << "current_state.W_0\n" << current_state.W_0 << std::endl;
            // std::cout << "current_state.W_1\n" << current_state.W_1 << std::endl;
            // std::cout << "current_state.W_2\n" << current_state.W_2 << std::endl;
            // std::cout << "current_state.W_3\n" << current_state.W_3 << std::endl;
        

            auto L2_norm = compute_L2_norm(dW_0, dW_1, dW_2, dW_3);
            iteration.push_back(it);
            Residuals.push_back({L2_norm(0), L2_norm(1), L2_norm(2), L2_norm(3)});

            auto end_time = std::chrono::high_resolution_clock::now(); // End timer
            std::chrono::duration<double> elapsed = end_time - start;
            iteration_times.push_back(elapsed.count());

            std::cout << "Iteration: " << it << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";

            auto [C_l, C_d, C_m] = compute_coeff();

            std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";

            // Check for convergence
            if (L2_norm(0) < convergence_tol) {
                break;
            }
        }
    }
    else {
        Eigen::ArrayXXd R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3;
        for (int it = 0; it < it_max; it++) {
            current_state.run_even();
            // Initialize Rd0
            current_state.update_Rd0();

            W0_0 = current_state.W_0(seqy, seqx);
            W1_0 = current_state.W_1(seqy, seqx);
            W2_0 = current_state.W_2(seqy, seqx);
            W3_0 = current_state.W_3(seqy, seqx);            
            dt = compute_dt();

            // Stage 1
            Eigen::ArrayXXd& Rd0_0 = current_state.Rd_0;
            Eigen::ArrayXXd& Rd1_0 = current_state.Rd_1;
            Eigen::ArrayXXd& Rd2_0 = current_state.Rd_2;
            Eigen::ArrayXXd& Rd3_0 = current_state.Rd_3;
    
            dW_0 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);           
            dW_1 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            dW_2 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            dW_3 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);
             
            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(dW_0, dW_1, dW_2, dW_3);

            
            current_state.W_0(seqy, seqx) = W0_0 - a1*R_star_star_0;
            current_state.W_1(seqy, seqx) = W1_0 - a1*R_star_star_1;
            current_state.W_2(seqy, seqx) = W2_0 - a1*R_star_star_2;
            current_state.W_3(seqy, seqx) = W3_0 - a1*R_star_star_3;
                  
            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 2
            Rd0_0 = current_state.Rd_0;
            Rd1_0 = current_state.Rd_1;
            Rd2_0 = current_state.Rd_2;
            Rd3_0 = current_state.Rd_3;

            dW_0 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);
            dW_1 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            dW_2 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            dW_3 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);

            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(dW_0, dW_1, dW_2, dW_3);

            current_state.W_0(seqy, seqx) = W0_0 - a2*R_star_star_0;
            current_state.W_1(seqy, seqx) = W1_0 - a2*R_star_star_1;
            current_state.W_2(seqy, seqx) = W2_0 - a2*R_star_star_2;
            current_state.W_3(seqy, seqx) = W3_0 - a2*R_star_star_3;
        
            current_state.update_conservative_variables();
            current_state.run_even();

            // Stage 3
            Rd20_0 = b3*current_state.Rd_0 + (1-b3)*current_state.Rd0_0;
            Rd20_1 = b3*current_state.Rd_1 + (1-b3)*current_state.Rd0_1;
            Rd20_2 = b3*current_state.Rd_2 + (1-b3)*current_state.Rd0_2;
            Rd20_3 = b3*current_state.Rd_3 + (1-b3)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd20_0;
            current_state.Rd0_1 = Rd20_1;
            current_state.Rd0_2 = Rd20_2;
            current_state.Rd0_3 = Rd20_3;

            dW_0 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd20_0);
            dW_1 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd20_1);           
            dW_2 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd20_2);           
            dW_3 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd20_3);
            
            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(dW_0, dW_1, dW_2, dW_3);

            current_state.W_0(seqy, seqx) = W0_0 - a3*R_star_star_0;
            current_state.W_1(seqy, seqx) = W1_0 - a3*R_star_star_1;
            current_state.W_2(seqy, seqx) = W2_0 - a3*R_star_star_2;
            current_state.W_3(seqy, seqx) = W3_0 - a3*R_star_star_3;

            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 4
            Rd20_0 = current_state.Rd0_0;
            Rd20_1 = current_state.Rd0_1;
            Rd20_2 = current_state.Rd0_2;
            Rd20_3 = current_state.Rd0_3;

            dW_0 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd20_0);
            dW_1 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd20_1);            
            dW_2 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd20_2);           
            dW_3 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd20_3);

            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(dW_0, dW_1, dW_2, dW_3);

            current_state.W_0(seqy, seqx) = W0_0 - a4*R_star_star_0;
            current_state.W_1(seqy, seqx) = W1_0 - a4*R_star_star_1;
            current_state.W_2(seqy, seqx) = W2_0 - a4*R_star_star_2;
            current_state.W_3(seqy, seqx) = W3_0 - a4*R_star_star_3;

            current_state.update_conservative_variables();
            current_state.run_even();

            // Stage 5
            Rd42_0 = b5*current_state.Rd_0 + (1-b5)*current_state.Rd0_0;
            Rd42_1 = b5*current_state.Rd_1 + (1-b5)*current_state.Rd0_1;
            Rd42_2 = b5*current_state.Rd_2 + (1-b5)*current_state.Rd0_2;
            Rd42_3 = b5*current_state.Rd_3 + (1-b5)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd42_0;
            current_state.Rd0_1 = Rd42_1;
            current_state.Rd0_2 = Rd42_2;
            current_state.Rd0_3 = Rd42_3;

            dW_0 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd42_0);
            dW_1 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd42_1);           
            dW_2 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd42_2);           
            dW_3 = dt/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd42_3);
            
            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(dW_0, dW_1, dW_2, dW_3);

            current_state.W_0(seqy, seqx) = W0_0 - a5*R_star_star_0;
            current_state.W_1(seqy, seqx) = W1_0 - a5*R_star_star_1;
            current_state.W_2(seqy, seqx) = W2_0 - a5*R_star_star_2;
            current_state.W_3(seqy, seqx) = W3_0 - a5*R_star_star_3;

            current_state.update_conservative_variables();
        

            auto L2_norm = compute_L2_norm(dW_0, dW_1, dW_2, dW_3);
            iteration.push_back(it);
            Residuals.push_back({L2_norm(0), L2_norm(1), L2_norm(2), L2_norm(3)});

            auto end_time = std::chrono::high_resolution_clock::now(); // End timer
            std::chrono::duration<double> elapsed = end_time - start;
            iteration_times.push_back(elapsed.count());

            std::cout << "Iteration: " << it << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";

            auto [C_l, C_d, C_m] = compute_coeff();

            std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";

            // Check for convergence
            if (L2_norm(0) < convergence_tol && L2_norm(1) < convergence_tol && L2_norm(2) < convergence_tol && L2_norm(3) < convergence_tol) {
                break;
            }
        }
    }


    return {current_state.W_0, current_state.W_1, current_state.W_2, current_state.W_3, Residuals, iteration_times};
}

std::tuple<double, double, double> TemporalDiscretization::compute_coeff() {
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;

    auto seqx = Eigen::seq(2, current_state.ncells_x-3);    
    Eigen::ArrayXXd p_wall = 0.5*(3*current_state.p_cells(2, seqx) - current_state.p_cells(3, seqx));
    double Fx = (p_wall*current_state.nx_x(2, seqx)*current_state.Ds_x(2, seqx)).sum();
    double Fy = (p_wall*current_state.nx_y(2, seqx)*current_state.Ds_x(2, seqx)).sum();

    Eigen::ArrayXXd x_mid = 0.5*(current_state.x(0, Eigen::seq(0, x.cols()-2)) + current_state.x(0, Eigen::seq(1, x.cols()-1)));
    Eigen::ArrayXXd y_mid = 0.5*(current_state.y(0, Eigen::seq(0, x.cols()-2)) + current_state.y(0, Eigen::seq(1, x.cols()-1)));
    double M = (current_state.p_cells(2, seqx)*(-(x_mid-x_ref)*current_state.nx_y(2, seqx) + (y_mid-y_ref)*current_state.nx_x(2, seqx))*current_state.Ds_x(2, seqx)).sum();

    double L = Fy*std::cos(current_state.alpha) - Fx*std::sin(current_state.alpha);
    double D = Fy*std::sin(current_state.alpha) + Fx*std::cos(current_state.alpha);

    double C_l = L/(0.5*rho*(u*u+v*v)*c);
    double C_d = D/(0.5*rho*(u*u+v*v)*c);
    double C_m = M/(0.5*rho*(u*u+v*v)*c*c);

    return {C_l, C_d, C_m};
}
