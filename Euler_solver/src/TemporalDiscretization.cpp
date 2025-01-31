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


TemporalDiscretization::TemporalDiscretization(Eigen::ArrayXXd& x,
                            Eigen::ArrayXXd& y,
                            double rho,
                            double u,
                            double v,
                            double E,
                            double T,
                            double p,
                            double T_ref,
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
      T_ref(T_ref),
      U_ref(U_ref),
      current_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, T_ref, U_ref),
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

std::tuple<SpatialDiscretization, std::vector<std::vector<double>>> TemporalDiscretization::RungeKutta(int it_max) {
    double convergence_tol = 1e-11;
    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;


    current_state.run_even();
    // Initialize Rd0
    current_state.update_Rd0();

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

    auto seqy = Eigen::seq(2, current_state.ncells_y-3);
    auto seqx = Eigen::seq(2, current_state.ncells_x-3);

    if (res_smoothing == 0) {
        for (int it = 0; it < it_max; it++) {
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

            Res_0 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);
            dW_0 = -a1*dt*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            dW_1 = -a1*dt*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            dW_2 = -a1*dt*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);
            dW_3 = -a1*dt*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
                  
            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 2
            Rd0_0 = current_state.Rd_0;
            Rd1_0 = current_state.Rd_1;
            Rd2_0 = current_state.Rd_2;
            Rd3_0 = current_state.Rd_3;

            Res_0 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);
            dW_0 = -a2*dt*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            dW_1 = -a2*dt*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            dW_2 = -a2*dt*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);
            dW_3 = -a2*dt*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
        
            current_state.update_conservative_variables();
            current_state.run_even();
            
            // Stage 3
            // std::cout << "Stage 3" << std::endl;
            Rd20_0 = b3*current_state.Rd_0 + (1-b3)*current_state.Rd0_0;
            Rd20_1 = b3*current_state.Rd_1 + (1-b3)*current_state.Rd0_1;
            Rd20_2 = b3*current_state.Rd_2 + (1-b3)*current_state.Rd0_2;
            Rd20_3 = b3*current_state.Rd_3 + (1-b3)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd20_0;
            current_state.Rd0_1 = Rd20_1;
            current_state.Rd0_2 = Rd20_2;
            current_state.Rd0_3 = Rd20_3;

            Res_0 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd20_0);
            dW_0 = -a3*dt*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd20_1);
            dW_1 = -a3*dt*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd20_2);
            dW_2 = -a3*dt*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd20_3);
            dW_3 = -a3*dt*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 4
            Rd20_0 = current_state.Rd0_0;
            Rd20_1 = current_state.Rd0_1;
            Rd20_2 = current_state.Rd0_2;
            Rd20_3 = current_state.Rd0_3;

            Res_0 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd20_0);
            dW_0 = -a4*dt*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd20_1);
            dW_1 = -a4*dt*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd20_2);
            dW_2 = -a4*dt*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd20_3);
            dW_3 = -a4*dt*Res_3;
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

            Res_0 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd42_0);
            dW_0 = -a5*dt*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd42_1);
            dW_1 = -a5*dt*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd42_2);
            dW_2 = -a5*dt*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd42_3);
            dW_3 = -a5*dt*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_odd();
        

            auto L2_norm = compute_L2_norm(dW_0, dW_1, dW_2, dW_3);
            iteration.push_back(it);
            Residuals.push_back({L2_norm(0), L2_norm(1), L2_norm(2), L2_norm(3)});

            std::cout << "Iteration: " << it << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";

            auto [C_l, C_d, C_m] = compute_coeff();

            std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";

            // Check for convergence
            if (L2_norm(0) < convergence_tol && L2_norm(1) < convergence_tol && L2_norm(2) < convergence_tol && L2_norm(3) < convergence_tol) {
                break;
            }
        }
    }
    else {
        for (int it = 0; it < it_max; it++) {
            Eigen::ArrayXXd W0_0 = current_state.W_0(seqy, seqx);
            Eigen::ArrayXXd W1_0 = current_state.W_1(seqy, seqx);
            Eigen::ArrayXXd W2_0 = current_state.W_2(seqy, seqx);
            Eigen::ArrayXXd W3_0 = current_state.W_3(seqy, seqx);            
            Eigen::ArrayXXd dt = compute_dt();

            // Stage 1
            Eigen::ArrayXXd& Rd0_0 = current_state.Rd_0;
            Eigen::ArrayXXd& Rd1_0 = current_state.Rd_1;
            Eigen::ArrayXXd& Rd2_0 = current_state.Rd_2;
            Eigen::ArrayXXd& Rd3_0 = current_state.Rd_3;

            Eigen::ArrayXXd Res_0 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_0 - Rd0_0);
            dW_0 = -a1*dt*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Eigen::ArrayXXd Res_1 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_1 - Rd1_0);
            dW_1 = -a1*dt*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Eigen::ArrayXXd Res_2 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_2 - Rd2_0);
            dW_2 = -a1*dt*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Eigen::ArrayXXd Res_3 = 1/current_state.OMEGA(seqy, seqx)*(current_state.Rc_3 - Rd3_0);
            dW_3 = -a1*dt*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
                  
            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 2
        }
    }


    return {current_state, Residuals};
}

std::tuple<double, double, double> TemporalDiscretization::compute_coeff() {
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;

    auto seqx = Eigen::seq(2, current_state.ncells_x-3);    
    double Fx = (current_state.p_cells(2, seqx)*current_state.nx_x(2, seqx)*current_state.Ds_x(2, seqx)).sum();
    double Fy = (current_state.p_cells(2, seqx)*current_state.nx_y(2, seqx)*current_state.Ds_x(2, seqx)).sum();

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
