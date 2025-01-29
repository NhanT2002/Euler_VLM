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
      current_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, T_ref, U_ref),
      sigma(sigma),
      k2_coeff(k2_coeff),
      k4_coeff(k4_coeff),
      res_smoothing(res_smoothing) {}

Eigen::ArrayXXd TemporalDiscretization::compute_dt() const {

    Eigen::ArrayXXd dt_array = sigma*current_state.OMEGA/(current_state.Lambda_I + current_state.Lambda_J);

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

    Eigen::ArrayXXd dW_0, dW_1, dW_2, dW_3;
    std::vector<std::vector<double>> Residuals;
    std::vector<int> iteration;
    
    Residuals = std::vector<std::vector<double>>{};
    iteration = std::vector<int>{};

    auto seqy = Eigen::seq(2, current_state.ncells_y-3);
    auto seqx = Eigen::seq(2, current_state.ncells_x-3);

    for (int it = 0; it < it_max; it++) {
        if (res_smoothing == 0) {
            Eigen::ArrayXXd W0_0 = current_state.W_0(seqy, seqx);
            Eigen::ArrayXXd W1_0 = current_state.W_1(seqy, seqx);
            Eigen::ArrayXXd W2_0 = current_state.W_2(seqy, seqx);
            Eigen::ArrayXXd W3_0 = current_state.W_3(seqy, seqx);

            // Stage 1
            Eigen::ArrayXXd dt = compute_dt();
            Eigen::ArrayXXd& Rd0_0 = current_state.Rd_0;
            Eigen::ArrayXXd& Rd1_0 = current_state.Rd_1;
            Eigen::ArrayXXd& Rd2_0 = current_state.Rd_2;
            Eigen::ArrayXXd& Rd3_0 = current_state.Rd_3;

            Eigen::ArrayXXd Res_0 = 1/current_state.OMEGA*(current_state.Rc_0 - Rd0_0);
            dW_0 = -a1*dt(seqy, seqx)*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Eigen::ArrayXXd Res_1 = 1/current_state.OMEGA*(current_state.Rc_1 - Rd1_0);
            dW_1 = -a1*dt(seqy, seqx)*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Eigen::ArrayXXd Res_2 = 1/current_state.OMEGA*(current_state.Rc_2 - Rd2_0);
            dW_2 = -a1*dt(seqy, seqx)*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Eigen::ArrayXXd Res_3 = 1/current_state.OMEGA*(current_state.Rc_3 - Rd3_0);
            dW_3 = -a1*dt(seqy, seqx)*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
        
            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 2
            dt = compute_dt();
            Rd0_0 = current_state.Rd_0;
            Rd1_0 = current_state.Rd_1;
            Rd2_0 = current_state.Rd_2;
            Rd3_0 = current_state.Rd_3;

            Res_0 = 1/current_state.OMEGA*(current_state.Rc_0 - Rd0_0);
            dW_0 = -a2*dt(seqy, seqx)*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA*(current_state.Rc_1 - Rd1_0);
            dW_1 = -a2*dt(seqy, seqx)*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA*(current_state.Rc_2 - Rd2_0);
            dW_2 = -a2*dt(seqy, seqx)*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA*(current_state.Rc_3 - Rd3_0);
            dW_3 = -a2*dt(seqy, seqx)*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;
        
            current_state.update_conservative_variables();
            current_state.run_even();
            
            // Stage 3
            dt = compute_dt();
            Eigen::ArrayXXd Rd20_0 = b3*current_state.Rd_0 + (1-b3)*current_state.Rd0_0;
            Eigen::ArrayXXd Rd20_1 = b3*current_state.Rd_1 + (1-b3)*current_state.Rd0_1;
            Eigen::ArrayXXd Rd20_2 = b3*current_state.Rd_2 + (1-b3)*current_state.Rd0_2;
            Eigen::ArrayXXd Rd20_3 = b3*current_state.Rd_3 + (1-b3)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd20_0;
            current_state.Rd0_1 = Rd20_1;
            current_state.Rd0_2 = Rd20_2;
            current_state.Rd0_3 = Rd20_3;

            Res_0 = 1/current_state.OMEGA*(current_state.Rc_0 - Rd20_0);
            dW_0 = -a3*dt(seqy, seqx)*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA*(current_state.Rc_1 - Rd20_1);
            dW_1 = -a3*dt(seqy, seqx)*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA*(current_state.Rc_2 - Rd20_2);
            dW_2 = -a3*dt(seqy, seqx)*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA*(current_state.Rc_3 - Rd20_3);
            dW_3 = -a3*dt(seqy, seqx)*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_odd();

            // Stage 4
            dt = compute_dt();
            Rd20_0 = current_state.Rd_0;
            Rd20_1 = current_state.Rd_1;
            Rd20_2 = current_state.Rd_2;
            Rd20_3 = current_state.Rd_3;

            Res_0 = 1/current_state.OMEGA*(current_state.Rc_0 - Rd20_0);
            dW_0 = -a4*dt(seqy, seqx)*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA*(current_state.Rc_1 - Rd20_1);
            dW_1 = -a4*dt(seqy, seqx)*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA*(current_state.Rc_2 - Rd20_2);
            dW_2 = -a4*dt(seqy, seqx)*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA*(current_state.Rc_3 - Rd20_3);
            dW_3 = -a4*dt(seqy, seqx)*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_even();

            // Stage 5
            dt = compute_dt();
            Eigen::ArrayXXd Rd42_0 = b5*current_state.Rd_0 + (1-b5)*current_state.Rd0_0;
            Eigen::ArrayXXd Rd42_1 = b5*current_state.Rd_1 + (1-b5)*current_state.Rd0_1;
            Eigen::ArrayXXd Rd42_2 = b5*current_state.Rd_2 + (1-b5)*current_state.Rd0_2;
            Eigen::ArrayXXd Rd42_3 = b5*current_state.Rd_3 + (1-b5)*current_state.Rd0_3;

            current_state.Rd0_0 = Rd42_0;
            current_state.Rd0_1 = Rd42_1;
            current_state.Rd0_2 = Rd42_2;
            current_state.Rd0_3 = Rd42_3;

            Res_0 = 1/current_state.OMEGA*(current_state.Rc_0 - Rd42_0);
            dW_0 = -a5*dt(seqy, seqx)*Res_0;
            current_state.W_0(seqy, seqx) = W0_0 + dW_0;

            Res_1 = 1/current_state.OMEGA*(current_state.Rc_1 - Rd42_1);
            dW_1 = -a5*dt(seqy, seqx)*Res_1;
            current_state.W_1(seqy, seqx) = W1_0 + dW_1;

            Res_2 = 1/current_state.OMEGA*(current_state.Rc_2 - Rd42_2);
            dW_2 = -a5*dt(seqy, seqx)*Res_2;
            current_state.W_2(seqy, seqx) = W2_0 + dW_2;

            Res_3 = 1/current_state.OMEGA*(current_state.Rc_3 - Rd42_3);
            dW_3 = -a5*dt(seqy, seqx)*Res_3;
            current_state.W_3(seqy, seqx) = W3_0 + dW_3;

            current_state.update_conservative_variables();
            current_state.run_odd();
        }

        auto L2_norm = compute_L2_norm(dW_0, dW_1, dW_2, dW_3);
        iteration.push_back(it);
        Residuals.push_back({L2_norm(0), L2_norm(1), L2_norm(2), L2_norm(3)});

        std::cout << "Iteration: " << it << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << std::endl;

        // Check for convergence
        if (L2_norm(0) < convergence_tol && L2_norm(1) < convergence_tol && L2_norm(2) < convergence_tol && L2_norm(3) < convergence_tol) {
            break;
        }
    }


    return {current_state, Residuals};
}
