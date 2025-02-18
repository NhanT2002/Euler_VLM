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

void multigrid_halo(Eigen::ArrayXXd& array) {
    int im1 = array.cols() - 3;
    int im2 = array.cols() - 4;
    int ip1 = 2;
    int ip2 = 3;

    array.col(1) = array.col(im1);
    array.col(0) = array.col(im2);
    array.col(array.cols()-2) = array.col(ip1);
    array.col(array.cols()-1) = array.col(ip2);
}

Eigen::VectorXd triDiagonal(const Eigen::VectorXd& a, const Eigen::VectorXd& b, const Eigen::VectorXd& c, const Eigen::VectorXd& d) {

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

Multigrid::Multigrid(SpatialDiscretization& h_state, double sigma, int res_smoothing, double k2_coeff, double k4_coeff)
    : h_state(h_state), sigma(sigma), res_smoothing(res_smoothing), k2_coeff(k2_coeff), k4_coeff(k4_coeff), multigrid_convergence(false) {
        h_state.run_even();
        h_state.update_Rd0();
    }

SpatialDiscretization Multigrid::mesh_restriction(SpatialDiscretization& h_state) {
    int ny_2h = (h_state.nvertex_y+1)/2;
    int nx_2h = (h_state.nvertex_x+1)/2;

    Eigen::ArrayXXd x_2h(ny_2h, nx_2h);
    Eigen::ArrayXXd y_2h(ny_2h, nx_2h);

    for (int j = 0; j < (h_state.nvertex_y-1)/2+1; j++) {
        for (int i = 0; i < (h_state.nvertex_x-1)/2+1; i++) {
            x_2h(j, i) = h_state.x(2*j, 2*i);
            y_2h(j, i) = h_state.y(2*j, 2*i);
        }
    }

    // // Verify that the mesh is correct
    // write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xy");
    // write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xy");

    SpatialDiscretization h2_state(x_2h, y_2h, h_state.rho, h_state.u, h_state.v, h_state.E, h_state.T, h_state.p, h_state.k2_coeff, h_state.k4_coeff, h_state.Mach, h_state.U_ref);
        
    h2_state.run_even();
    h2_state.update_Rd0();

    return h2_state;
}

void Multigrid::restriction(SpatialDiscretization& h_state, SpatialDiscretization& h2_state) {
    
    int ny_2h = (h_state.nvertex_y+1)/2;
    int nx_2h = (h_state.nvertex_x+1)/2;

    auto seqy = Eigen::seq(2, h_state.ncells_y-4, 2);
    auto seqyp1 = Eigen::seq(3, h_state.ncells_y-3, 2);
    auto seqx = Eigen::seq(2, h_state.ncells_x-4, 2);
    auto seqxp1 = Eigen::seq(3, h_state.ncells_x-3, 2);

    Eigen::ArrayXXd h2_OMEGA = (h_state.OMEGA(seqy, seqx) + h_state.OMEGA(seqy, seqxp1) + h_state.OMEGA(seqyp1, seqx) + h_state.OMEGA(seqyp1, seqxp1));
    h2_state.W_0(Eigen::seq(2, h2_state.ncells_y-3), Eigen::seq(2, h2_state.ncells_x-3)) = (h_state.W_0(seqy, seqx)*h_state.OMEGA(seqy, seqx) + 
                                                                                            h_state.W_0(seqy, seqxp1)*h_state.OMEGA(seqy, seqxp1) + 
                                                                                            h_state.W_0(seqyp1, seqx)*h_state.OMEGA(seqyp1, seqx) + 
                                                                                            h_state.W_0(seqyp1, seqxp1)*h_state.OMEGA(seqyp1, seqxp1))/h2_OMEGA;
    h2_state.W_1(Eigen::seq(2, h2_state.ncells_y-3), Eigen::seq(2, h2_state.ncells_x-3)) = (h_state.W_1(seqy, seqx)*h_state.OMEGA(seqy, seqx) + 
                                                                                            h_state.W_1(seqy, seqxp1)*h_state.OMEGA(seqy, seqxp1) + 
                                                                                            h_state.W_1(seqyp1, seqx)*h_state.OMEGA(seqyp1, seqx) + 
                                                                                            h_state.W_1(seqyp1, seqxp1)*h_state.OMEGA(seqyp1, seqxp1))/h2_OMEGA;
    h2_state.W_2(Eigen::seq(2, h2_state.ncells_y-3), Eigen::seq(2, h2_state.ncells_x-3)) = (h_state.W_2(seqy, seqx)*h_state.OMEGA(seqy, seqx) + 
                                                                                            h_state.W_2(seqy, seqxp1)*h_state.OMEGA(seqy, seqxp1) + 
                                                                                            h_state.W_2(seqyp1, seqx)*h_state.OMEGA(seqyp1, seqx) + 
                                                                                            h_state.W_2(seqyp1, seqxp1)*h_state.OMEGA(seqyp1, seqxp1))/h2_OMEGA;
    h2_state.W_3(Eigen::seq(2, h2_state.ncells_y-3), Eigen::seq(2, h2_state.ncells_x-3)) = (h_state.W_3(seqy, seqx)*h_state.OMEGA(seqy, seqx) + 
                                                                                            h_state.W_3(seqy, seqxp1)*h_state.OMEGA(seqy, seqxp1) + 
                                                                                            h_state.W_3(seqyp1, seqx)*h_state.OMEGA(seqyp1, seqx) + 
                                                                                            h_state.W_3(seqyp1, seqxp1)*h_state.OMEGA(seqyp1, seqxp1))/h2_OMEGA;

    h2_state.update_conservative_variables();
    h2_state.run_even();
    h2_state.update_Rd0();


    // std::cout << "h2 rho_cells\n" << h2_state.rho_cells << std::endl;
    // std::cout << "h rho_cells\n" << h_state.rho_cells << std::endl;
    // std::cout << "h2 u_cells\n" << h2_state.u_cells << std::endl;
    // std::cout << "h u_cells\n" << h_state.u_cells << std::endl;
    // std::cout << "h2 v_cells\n" << h2_state.v_cells << std::endl;
    // std::cout << "h v_cells\n" << h_state.v_cells << std::endl;
    // std::cout << "h2 E_cells\n" << h2_state.E_cells << std::endl;
    // std::cout << "h E_cells\n" << h_state.E_cells << std::endl;
    // std::cout << "h2 p_cells\n" << h2_state.p_cells << std::endl;
    // std::cout << "h p_cells\n" << h_state.p_cells << std::endl;

    seqy = Eigen::seq(0, h_state.ncells_y-6, 2);
    seqx = Eigen::seq(0, h_state.ncells_x-6, 2);
    seqyp1 = Eigen::seq(1, h_state.ncells_y-5, 2);
    seqxp1 = Eigen::seq(1, h_state.ncells_x-5, 2);

    h2_state.restriction_operator_0 = h_state.Rc_0(seqy, seqx) - h_state.Rd0_0(seqy, seqx) + h_state.forcing_function_0(seqy, seqx) +
                                        h_state.Rc_0(seqy, seqxp1) - h_state.Rd0_0(seqy, seqxp1) + h_state.forcing_function_0(seqy, seqxp1) +
                                        h_state.Rc_0(seqyp1, seqx) - h_state.Rd0_0(seqyp1, seqx) + h_state.forcing_function_0(seqyp1, seqx) +
                                        h_state.Rc_0(seqyp1, seqxp1) - h_state.Rd0_0(seqyp1, seqxp1) + h_state.forcing_function_0(seqyp1, seqxp1);
    h2_state.restriction_operator_1 = h_state.Rc_1(seqy, seqx) - h_state.Rd0_1(seqy, seqx) + h_state.forcing_function_1(seqy, seqx) +
                                        h_state.Rc_1(seqy, seqxp1) - h_state.Rd0_1(seqy, seqxp1) + h_state.forcing_function_1(seqy, seqxp1) +
                                        h_state.Rc_1(seqyp1, seqx) - h_state.Rd0_1(seqyp1, seqx) + h_state.forcing_function_1(seqyp1, seqx) +
                                        h_state.Rc_1(seqyp1, seqxp1) - h_state.Rd0_1(seqyp1, seqxp1) + h_state.forcing_function_1(seqyp1, seqxp1);
    h2_state.restriction_operator_2 = h_state.Rc_2(seqy, seqx) - h_state.Rd0_2(seqy, seqx) + h_state.forcing_function_2(seqy, seqx) +
                                        h_state.Rc_2(seqy, seqxp1) - h_state.Rd0_2(seqy, seqxp1) + h_state.forcing_function_2(seqy, seqxp1) +
                                        h_state.Rc_2(seqyp1, seqx) - h_state.Rd0_2(seqyp1, seqx) + h_state.forcing_function_2(seqyp1, seqx) +
                                        h_state.Rc_2(seqyp1, seqxp1) - h_state.Rd0_2(seqyp1, seqxp1) + h_state.forcing_function_2(seqyp1, seqxp1);
    h2_state.restriction_operator_3 = h_state.Rc_3(seqy, seqx) - h_state.Rd0_3(seqy, seqx) + h_state.forcing_function_3(seqy, seqx) +
                                        h_state.Rc_3(seqy, seqxp1) - h_state.Rd0_3(seqy, seqxp1) + h_state.forcing_function_3(seqy, seqxp1) +
                                        h_state.Rc_3(seqyp1, seqx) - h_state.Rd0_3(seqyp1, seqx) + h_state.forcing_function_3(seqyp1, seqx) +
                                        h_state.Rc_3(seqyp1, seqxp1) - h_state.Rd0_3(seqyp1, seqxp1) + h_state.forcing_function_3(seqyp1, seqxp1);

    // Compute forcing function


    h2_state.forcing_function_0 = h2_state.restriction_operator_0 - (h2_state.Rc_0 - h2_state.Rd0_0);
    h2_state.forcing_function_1 = h2_state.restriction_operator_1 - (h2_state.Rc_1 - h2_state.Rd0_1);
    h2_state.forcing_function_2 = h2_state.restriction_operator_2 - (h2_state.Rc_2 - h2_state.Rd0_2);
    h2_state.forcing_function_3 = h2_state.restriction_operator_3 - (h2_state.Rc_3 - h2_state.Rd0_3);

    
    // Store initial interpolated solution
    h2_state.W2h_0 = h2_state.W_0;
    h2_state.W2h_1 = h2_state.W_1;
    h2_state.W2h_2 = h2_state.W_2;
    h2_state.W2h_3 = h2_state.W_3;
}

Eigen::ArrayXXd Multigrid::compute_dt(SpatialDiscretization& h_state) {

    auto seqy = Eigen::seq(2, h_state.ncells_y-3);
    auto seqx = Eigen::seq(2, h_state.ncells_x-3);
    Eigen::ArrayXXd dt_array = sigma*h_state.OMEGA(seqy, seqx)/(h_state.Lambda_I(seqy, seqx) + h_state.Lambda_J(seqy, seqx));
    // std::cout << "dt_array\n" << dt_array << std::endl;

    return dt_array;
}

Eigen::Array<double, 4, 1> Multigrid::compute_L2_norm(const Eigen::ArrayXXd &dW_0, const Eigen::ArrayXXd &dW_1, const Eigen::ArrayXXd &dW_2, const Eigen::ArrayXXd &dW_3) {
    Eigen::Array<double, 4, 1> L2_norms;

    L2_norms(0) = std::sqrt(((dW_0*dW_0).sum()/dW_0.size()));
    L2_norms(1) = std::sqrt(((dW_1*dW_1).sum()/dW_1.size()));
    L2_norms(2) = std::sqrt(((dW_2*dW_2).sum()/dW_2.size()));
    L2_norms(3) = std::sqrt(((dW_3*dW_3).sum()/dW_3.size()));

    return L2_norms;
}

std::tuple<double, double, double> Multigrid::compute_coeff(SpatialDiscretization& current_state) {
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;

    auto seqx = Eigen::seq(2, current_state.ncells_x-3);    
    Eigen::ArrayXXd p_wall = 0.125*(15*current_state.p_cells(2, seqx) - 10*current_state.p_cells(3, seqx) + 3*current_state.p_cells(4, seqx));
    double Fx = (p_wall*current_state.nx_x(2, seqx)*current_state.Ds_x(2, seqx)).sum();
    double Fy = (p_wall*current_state.nx_y(2, seqx)*current_state.Ds_x(2, seqx)).sum();

    Eigen::ArrayXXd x_mid = 0.5*(current_state.x(0, Eigen::seq(0, current_state.x.cols()-2)) + current_state.x(0, Eigen::seq(1, current_state.x.cols()-1)));
    Eigen::ArrayXXd y_mid = 0.5*(current_state.y(0, Eigen::seq(0, current_state.x.cols()-2)) + current_state.y(0, Eigen::seq(1, current_state.x.cols()-1)));
    double M = (current_state.p_cells(2, seqx)*(-(x_mid-x_ref)*current_state.nx_y(2, seqx) + (y_mid-y_ref)*current_state.nx_x(2, seqx))*current_state.Ds_x(2, seqx)).sum();

    double L = Fy*std::cos(current_state.alpha) - Fx*std::sin(current_state.alpha);
    double D = Fy*std::sin(current_state.alpha) + Fx*std::cos(current_state.alpha);

    double C_l = L/(0.5*current_state.rho*(current_state.u*current_state.u+current_state.v*current_state.v)*c);
    double C_d = D/(0.5*current_state.rho*(current_state.u*current_state.u+current_state.v*current_state.v)*c);
    double C_m = M/(0.5*current_state.rho*(current_state.u*current_state.u+current_state.v*current_state.v)*c*c);

    return {C_l, C_d, C_m};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> Multigrid::compute_abc(SpatialDiscretization& current_state) {
    auto seqy = Eigen::seq(2, current_state.ncells_y-3);
    auto seqx = Eigen::seq(2, current_state.ncells_x-3);
    int n = (current_state.ncells_x-4)*(current_state.ncells_y-4);

    double rr = 2.0;
    Eigen::ArrayXXd r = current_state.Lambda_J(seqy, seqx)/current_state.Lambda_I(seqy, seqx);
    Eigen::ArrayXXd eps_I = (0.25*((rr/(1+0.125*r)).square()-1)).max(0.0);
    Eigen::ArrayXXd eps_J = (0.25*((rr/(1+0.125/r)).square()-1)).max(0.0); 

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

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> Multigrid::R_star(SpatialDiscretization& current_state, const Eigen::ArrayXXd& dW_0, const Eigen::ArrayXXd& dW_1, const Eigen::ArrayXXd& dW_2, const Eigen::ArrayXXd& dW_3) {
    Eigen::VectorXd d_0, d_1, d_2, d_3;
    Eigen::VectorXd R_star_0, R_star_1, R_star_2, R_star_3;
    Eigen::VectorXd R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3;

    d_0 = dW_0.transpose().reshaped();
    d_1 = dW_1.transpose().reshaped();
    d_2 = dW_2.transpose().reshaped();
    d_3 = dW_3.transpose().reshaped();

    auto [a_I, b_I, c_I, a_J, b_J, c_J] = Multigrid::compute_abc(current_state);
    R_star_0 = triDiagonal(a_I, b_I, c_I, d_0);
    R_star_1 = triDiagonal(a_I, b_I, c_I, d_1);
    R_star_2 = triDiagonal(a_I, b_I, c_I, d_2);
    R_star_3 = triDiagonal(a_I, b_I, c_I, d_3);
    

    R_star_star_0 = triDiagonal(a_J, b_J, c_J, R_star_0.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());
    R_star_star_1 = triDiagonal(a_J, b_J, c_J, R_star_1.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());
    R_star_star_2 = triDiagonal(a_J, b_J, c_J, R_star_2.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());
    R_star_star_3 = triDiagonal(a_J, b_J, c_J, R_star_3.reshaped(dW_0.rows(), dW_0.cols()).transpose().reshaped());

    return {R_star_star_0.reshaped(dW_0.rows(), dW_0.cols()).array(), R_star_star_1.reshaped(dW_0.rows(), dW_0.cols()).array(), R_star_star_2.reshaped(dW_0.rows(), dW_0.cols()).array(), R_star_star_3.reshaped(dW_0.rows(), dW_0.cols()).array()};
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, std::vector<std::vector<double>>> Multigrid::restriction_timestep(SpatialDiscretization& h_state, 
                                                                                                                    int it_max, 
                                                                                                                    int current_iteration) {
    double convergence_tol = 1e-11;
    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;                                                                                                                    
    multigrid_convergence = false;


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

    auto seqy = Eigen::seq(2, h_state.ncells_y-3);
    auto seqx = Eigen::seq(2, h_state.ncells_x-3);


    if (res_smoothing == 0) {
        for (int it = 0; it < it_max; it++) {
            W0_0 = h_state.W_0(seqy, seqx);
            W1_0 = h_state.W_1(seqy, seqx);
            W2_0 = h_state.W_2(seqy, seqx);
            W3_0 = h_state.W_3(seqy, seqx);            
            dt = compute_dt(h_state);

            // Stage 1
            Eigen::ArrayXXd& Rd0_0 = h_state.Rd_0;
            Eigen::ArrayXXd& Rd1_0 = h_state.Rd_1;
            Eigen::ArrayXXd& Rd2_0 = h_state.Rd_2;
            Eigen::ArrayXXd& Rd3_0 = h_state.Rd_3;

            // std::cout << "first stage : h_state.Rc_0 - Rd0_0 + h_state.forcing_function_0\n" << h_state.Rc_0 - Rd0_0 + h_state.forcing_function_0 << std::endl;
            dW_0 = -a1*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd0_0 + h_state.forcing_function_0);
            h_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a1*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd1_0 + h_state.forcing_function_1);
            h_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a1*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd2_0 + h_state.forcing_function_2);
            h_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a1*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd3_0 + h_state.forcing_function_3);
            h_state.W_3(seqy, seqx) = W3_0 + dW_3;
                  
            h_state.update_conservative_variables();
            h_state.run_odd();

            // Stage 2
            Rd0_0 = h_state.Rd_0;
            Rd1_0 = h_state.Rd_1;
            Rd2_0 = h_state.Rd_2;
            Rd3_0 = h_state.Rd_3;

            dW_0 = -a2*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd0_0 + h_state.forcing_function_0);
            h_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a2*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd1_0 + h_state.forcing_function_1);
            h_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a2*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd2_0 + h_state.forcing_function_2);
            h_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a2*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd3_0 + h_state.forcing_function_3);
            h_state.W_3(seqy, seqx) = W3_0 + dW_3;
        
            h_state.update_conservative_variables();
            h_state.run_even();
            
            // Stage 3
            Rd20_0 = b3*h_state.Rd_0 + (1-b3)*h_state.Rd0_0;
            Rd20_1 = b3*h_state.Rd_1 + (1-b3)*h_state.Rd0_1;
            Rd20_2 = b3*h_state.Rd_2 + (1-b3)*h_state.Rd0_2;
            Rd20_3 = b3*h_state.Rd_3 + (1-b3)*h_state.Rd0_3;

            h_state.Rd0_0 = Rd20_0;
            h_state.Rd0_1 = Rd20_1;
            h_state.Rd0_2 = Rd20_2;
            h_state.Rd0_3 = Rd20_3;

            dW_0 = -a3*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd20_0 + h_state.forcing_function_0);
            h_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a3*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd20_1 + h_state.forcing_function_1);
            h_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a3*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd20_2 + h_state.forcing_function_2);
            h_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a3*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd20_3 + h_state.forcing_function_3);
            h_state.W_3(seqy, seqx) = W3_0 + dW_3;

            h_state.update_conservative_variables();
            h_state.run_odd();

            // Stage 4
            Rd20_0 = h_state.Rd0_0;
            Rd20_1 = h_state.Rd0_1;
            Rd20_2 = h_state.Rd0_2;
            Rd20_3 = h_state.Rd0_3;

            dW_0 = -a4*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd20_0 + h_state.forcing_function_0);
            h_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a4*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd20_1 + h_state.forcing_function_1);
            h_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a4*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd20_2 + h_state.forcing_function_2);
            h_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a4*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd20_3 + h_state.forcing_function_3);
            h_state.W_3(seqy, seqx) = W3_0 + dW_3;

            h_state.update_conservative_variables();
            h_state.run_even();

            // Stage 5
            Rd42_0 = b5*h_state.Rd_0 + (1-b5)*h_state.Rd0_0;
            Rd42_1 = b5*h_state.Rd_1 + (1-b5)*h_state.Rd0_1;
            Rd42_2 = b5*h_state.Rd_2 + (1-b5)*h_state.Rd0_2;
            Rd42_3 = b5*h_state.Rd_3 + (1-b5)*h_state.Rd0_3;

            h_state.Rd0_0 = Rd42_0;
            h_state.Rd0_1 = Rd42_1;
            h_state.Rd0_2 = Rd42_2;
            h_state.Rd0_3 = Rd42_3;

            dW_0 = -a5*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd42_0 + h_state.forcing_function_0);
            h_state.W_0(seqy, seqx) = W0_0 + dW_0;

            dW_1 = -a5*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd42_1 + h_state.forcing_function_1);
            h_state.W_1(seqy, seqx) = W1_0 + dW_1;

            dW_2 = -a5*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd42_2 + h_state.forcing_function_2);
            h_state.W_2(seqy, seqx) = W2_0 + dW_2;

            dW_3 = -a5*dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd42_3 + h_state.forcing_function_3);
            h_state.W_3(seqy, seqx) = W3_0 + dW_3;

            h_state.update_conservative_variables();
            h_state.run_odd();
        

            auto L2_norm = compute_L2_norm(dW_0, dW_1, dW_2, dW_3);
            iteration.push_back(it);
            Residuals.push_back({L2_norm(0), L2_norm(1), L2_norm(2), L2_norm(3)});
            

            if (current_iteration == 0) {
                auto [C_l, C_d, C_m] = compute_coeff(h_state);           
                std::cout << "Iteration: " << it << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";
                std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";
            }
            else if (current_iteration > 0) {
                auto [C_l, C_d, C_m] = compute_coeff(h_state);         
                std::cout << "Iteration: " << current_iteration << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";
                std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";
            }
            

            

            // Check for convergence
            if (L2_norm(0) < convergence_tol && L2_norm(1) < convergence_tol && L2_norm(2) < convergence_tol && L2_norm(3) < convergence_tol) {
                multigrid_convergence = true;
                break;
            }
        }
    }
    else {
        Eigen::ArrayXXd R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3;
        for (int it = 0; it < it_max; it++) {
            W0_0 = h_state.W_0(seqy, seqx);
            W1_0 = h_state.W_1(seqy, seqx);
            W2_0 = h_state.W_2(seqy, seqx);
            W3_0 = h_state.W_3(seqy, seqx);            
            dt = compute_dt(h_state);

            // Stage 1
            Eigen::ArrayXXd& Rd0_0 = h_state.Rd_0;
            Eigen::ArrayXXd& Rd1_0 = h_state.Rd_1;
            Eigen::ArrayXXd& Rd2_0 = h_state.Rd_2;
            Eigen::ArrayXXd& Rd3_0 = h_state.Rd_3;
    
            dW_0 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd0_0 + h_state.forcing_function_0);           
            dW_1 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd1_0 + h_state.forcing_function_1);
            dW_2 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd2_0 + h_state.forcing_function_2);
            dW_3 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd3_0 + h_state.forcing_function_3);
             
            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(h_state, dW_0, dW_1, dW_2, dW_3);

            
            h_state.W_0(seqy, seqx) = W0_0 - a1*R_star_star_0;
            h_state.W_1(seqy, seqx) = W1_0 - a1*R_star_star_1;
            h_state.W_2(seqy, seqx) = W2_0 - a1*R_star_star_2;
            h_state.W_3(seqy, seqx) = W3_0 - a1*R_star_star_3;
                  
            h_state.update_conservative_variables();
            h_state.run_odd();

            // Stage 2
            Rd0_0 = h_state.Rd_0;
            Rd1_0 = h_state.Rd_1;
            Rd2_0 = h_state.Rd_2;
            Rd3_0 = h_state.Rd_3;

            dW_0 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd0_0 + h_state.forcing_function_0);
            dW_1 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd1_0 + h_state.forcing_function_1);
            dW_2 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd2_0 + h_state.forcing_function_2);
            dW_3 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd3_0 + h_state.forcing_function_3);

            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(h_state, dW_0, dW_1, dW_2, dW_3);

            h_state.W_0(seqy, seqx) = W0_0 - a2*R_star_star_0;
            h_state.W_1(seqy, seqx) = W1_0 - a2*R_star_star_1;
            h_state.W_2(seqy, seqx) = W2_0 - a2*R_star_star_2;
            h_state.W_3(seqy, seqx) = W3_0 - a2*R_star_star_3;
        
            h_state.update_conservative_variables();
            h_state.run_even();

            // Stage 3
            Rd20_0 = b3*h_state.Rd_0 + (1-b3)*h_state.Rd0_0;
            Rd20_1 = b3*h_state.Rd_1 + (1-b3)*h_state.Rd0_1;
            Rd20_2 = b3*h_state.Rd_2 + (1-b3)*h_state.Rd0_2;
            Rd20_3 = b3*h_state.Rd_3 + (1-b3)*h_state.Rd0_3;

            h_state.Rd0_0 = Rd20_0;
            h_state.Rd0_1 = Rd20_1;
            h_state.Rd0_2 = Rd20_2;
            h_state.Rd0_3 = Rd20_3;

            dW_0 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd20_0 + h_state.forcing_function_0);
            dW_1 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd20_1 + h_state.forcing_function_1);           
            dW_2 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd20_2 + h_state.forcing_function_2);           
            dW_3 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd20_3 + h_state.forcing_function_3);
            
            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(h_state, dW_0, dW_1, dW_2, dW_3);

            h_state.W_0(seqy, seqx) = W0_0 - a3*R_star_star_0;
            h_state.W_1(seqy, seqx) = W1_0 - a3*R_star_star_1;
            h_state.W_2(seqy, seqx) = W2_0 - a3*R_star_star_2;
            h_state.W_3(seqy, seqx) = W3_0 - a3*R_star_star_3;

            h_state.update_conservative_variables();
            h_state.run_odd();

            // Stage 4
            Rd20_0 = h_state.Rd0_0;
            Rd20_1 = h_state.Rd0_1;
            Rd20_2 = h_state.Rd0_2;
            Rd20_3 = h_state.Rd0_3;

            dW_0 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd20_0 + h_state.forcing_function_0);
            dW_1 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd20_1 + h_state.forcing_function_1);            
            dW_2 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd20_2 + h_state.forcing_function_2);           
            dW_3 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd20_3 + h_state.forcing_function_3);

            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(h_state, dW_0, dW_1, dW_2, dW_3);

            h_state.W_0(seqy, seqx) = W0_0 - a4*R_star_star_0;
            h_state.W_1(seqy, seqx) = W1_0 - a4*R_star_star_1;
            h_state.W_2(seqy, seqx) = W2_0 - a4*R_star_star_2;
            h_state.W_3(seqy, seqx) = W3_0 - a4*R_star_star_3;

            h_state.update_conservative_variables();
            h_state.run_even();

            // Stage 5
            Rd42_0 = b5*h_state.Rd_0 + (1-b5)*h_state.Rd0_0;
            Rd42_1 = b5*h_state.Rd_1 + (1-b5)*h_state.Rd0_1;
            Rd42_2 = b5*h_state.Rd_2 + (1-b5)*h_state.Rd0_2;
            Rd42_3 = b5*h_state.Rd_3 + (1-b5)*h_state.Rd0_3;

            h_state.Rd0_0 = Rd42_0;
            h_state.Rd0_1 = Rd42_1;
            h_state.Rd0_2 = Rd42_2;
            h_state.Rd0_3 = Rd42_3;

            dW_0 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_0 - Rd42_0 + h_state.forcing_function_0);
            dW_1 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_1 - Rd42_1 + h_state.forcing_function_1);           
            dW_2 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_2 - Rd42_2 + h_state.forcing_function_2);           
            dW_3 = dt/h_state.OMEGA(seqy, seqx)*(h_state.Rc_3 - Rd42_3 + h_state.forcing_function_3);
            
            std::tie(R_star_star_0, R_star_star_1, R_star_star_2, R_star_star_3) = R_star(h_state, dW_0, dW_1, dW_2, dW_3);

            h_state.W_0(seqy, seqx) = W0_0 - a5*R_star_star_0;
            h_state.W_1(seqy, seqx) = W1_0 - a5*R_star_star_1;
            h_state.W_2(seqy, seqx) = W2_0 - a5*R_star_star_2;
            h_state.W_3(seqy, seqx) = W3_0 - a5*R_star_star_3;

            h_state.update_conservative_variables();
            h_state.run_odd();
        

            auto L2_norm = compute_L2_norm(dW_0, dW_1, dW_2, dW_3);
            iteration.push_back(it);
            Residuals.push_back({L2_norm(0), L2_norm(1), L2_norm(2), L2_norm(3)});

            if (current_iteration == 0) {
                auto [C_l, C_d, C_m] = compute_coeff(h_state);           
                std::cout << "Iteration: " << it << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";
                std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";
            }
            else if (current_iteration > 0) {
                auto [C_l, C_d, C_m] = compute_coeff(h_state);         
                std::cout << "Iteration: " << current_iteration << " : L2_norms: " << L2_norm(0) << " " << L2_norm(1) << " " << L2_norm(2) << " " << L2_norm(3) << " ";
                std::cout << "C_l: " << C_l << " C_d: " << C_d << " C_m: " << C_m << "\n";
            }

            // Check for convergence
            if (L2_norm(0) < convergence_tol && L2_norm(1) < convergence_tol && L2_norm(2) < convergence_tol && L2_norm(3) < convergence_tol) {
                multigrid_convergence = true;
                break;
            }
        }
    }

    return {h_state.W_0, h_state.W_1, h_state.W_2, h_state.W_3, Residuals};                                                              
}

void Multigrid::prolongation(SpatialDiscretization& h2_state, SpatialDiscretization& h_state) {

    h2_state.deltaW2h_0 = h2_state.W_0 - h2_state.W2h_0;
    h2_state.deltaW2h_1 = h2_state.W_1 - h2_state.W2h_1;
    h2_state.deltaW2h_2 = h2_state.W_2 - h2_state.W2h_2;
    h2_state.deltaW2h_3 = h2_state.W_3 - h2_state.W2h_3;


    auto seqy = Eigen::seq(2, h_state.ncells_y-4, 2);
    auto seqyp1 = Eigen::seq(3, h_state.ncells_y-3, 2);
    auto seqx = Eigen::seq(2, h_state.ncells_x-4, 2);
    auto seqxp1 = Eigen::seq(3, h_state.ncells_x-3, 2);

    auto h2_seqym1 = Eigen::seq(1, h2_state.ncells_y-4);
    auto h2_seqxm1 = Eigen::seq(1, h2_state.ncells_x-4);
    auto h2_seqy = Eigen::seq(2, h2_state.ncells_y-3);
    auto h2_seqx = Eigen::seq(2, h2_state.ncells_x-3);
    auto h2_seqyp1 = Eigen::seq(3, h2_state.ncells_y-2);
    auto h2_seqxp1 = Eigen::seq(3, h2_state.ncells_x-2);

    h_state.prolongation_operator_0(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_0(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_0(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_0(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqyp1, h2_seqxp1));

    h_state.prolongation_operator_1(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_1(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_1(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_1(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqyp1, h2_seqxp1));

    h_state.prolongation_operator_2(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_2(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_2(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_2(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqyp1, h2_seqxp1));

    h_state.prolongation_operator_3(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_3(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_3(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_3(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqyp1, h2_seqxp1));

    // std::cout << "h2 deltaW2h_0\n" << h2_state.deltaW2h_0 << std::endl;
    // std::cout << "h prolongation_operator_0\n" << h_state.prolongation_operator_0 << std::endl;


    // Compute W_h_+
    h_state.W_0 += h_state.prolongation_operator_0;
    h_state.W_1 += h_state.prolongation_operator_1;
    h_state.W_2 += h_state.prolongation_operator_2;
    h_state.W_3 += h_state.prolongation_operator_3;
}

void Multigrid::prolongation_smooth(SpatialDiscretization& h2_state, SpatialDiscretization& h_state) {
    // Coarse grid correction
    h2_state.deltaW2h_0 = h2_state.W_0 - h2_state.W2h_0;
    h2_state.deltaW2h_1 = h2_state.W_1 - h2_state.W2h_1;
    h2_state.deltaW2h_2 = h2_state.W_2 - h2_state.W2h_2;
    h2_state.deltaW2h_3 = h2_state.W_3 - h2_state.W2h_3;


    auto seqy = Eigen::seq(2, h_state.ncells_y-4, 2);
    auto seqyp1 = Eigen::seq(3, h_state.ncells_y-3, 2);
    auto seqx = Eigen::seq(2, h_state.ncells_x-4, 2);
    auto seqxp1 = Eigen::seq(3, h_state.ncells_x-3, 2);

    auto h2_seqym1 = Eigen::seq(1, h2_state.ncells_y-4);
    auto h2_seqxm1 = Eigen::seq(1, h2_state.ncells_x-4);
    auto h2_seqy = Eigen::seq(2, h2_state.ncells_y-3);
    auto h2_seqx = Eigen::seq(2, h2_state.ncells_x-3);
    auto h2_seqyp1 = Eigen::seq(3, h2_state.ncells_y-2);
    auto h2_seqxp1 = Eigen::seq(3, h2_state.ncells_x-2);

    h_state.prolongation_operator_0(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_0(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_0(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_0(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_0(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_0(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_0(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_0(h2_seqyp1, h2_seqxp1));

    h_state.prolongation_operator_1(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_1(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_1(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_1(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_1(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_1(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_1(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_1(h2_seqyp1, h2_seqxp1));

    h_state.prolongation_operator_2(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_2(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_2(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_2(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_2(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_2(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_2(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_2(h2_seqyp1, h2_seqxp1));

    h_state.prolongation_operator_3(seqy, seqx) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqym1, h2_seqxm1));
    h_state.prolongation_operator_3(seqy, seqxp1) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqym1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqym1, h2_seqxp1));
    h_state.prolongation_operator_3(seqyp1, seqx) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxm1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqyp1, h2_seqxm1));
    h_state.prolongation_operator_3(seqyp1, seqxp1) = 0.0625*(9*h2_state.deltaW2h_3(h2_seqy, h2_seqx) + 
                                                          3*h2_state.deltaW2h_3(h2_seqy, h2_seqxp1) + 
                                                          3*h2_state.deltaW2h_3(h2_seqyp1, h2_seqx) + 
                                                          h2_state.deltaW2h_3(h2_seqyp1, h2_seqxp1));

    // std::cout << "h_state.prologation_operator_0(2, 2) : " << h_state.prolongation_operator_0(Eigen::seq(0, 100), Eigen::seq(0, 100)) << std::endl;

    auto seqy_h = Eigen::seq(2, h_state.ncells_y-3);
    auto seqx_h = Eigen::seq(2, h_state.ncells_x-3);

    auto [deltaW2h_star_star_0, deltaW2h_star_star_1, deltaW2h_star_star_2, deltaW2h_star_star_3] = R_star(h_state, h_state.prolongation_operator_0(seqy_h, seqx_h), h_state.prolongation_operator_1(seqy_h, seqx_h), h_state.prolongation_operator_2(seqy_h, seqx_h), h_state.prolongation_operator_3(seqy_h, seqx_h));

    h_state.prolongation_operator_0(seqy_h, seqx_h) = deltaW2h_star_star_0;
    h_state.prolongation_operator_0.row(0) = h_state.prolongation_operator_0.row(2);
    h_state.prolongation_operator_0.row(1) = h_state.prolongation_operator_0.row(2);
    h_state.prolongation_operator_0.row(h_state.ncells_y-1) = h_state.prolongation_operator_0.row(h_state.ncells_y-3);
    h_state.prolongation_operator_0.row(h_state.ncells_y-2) = h_state.prolongation_operator_0.row(h_state.ncells_y-3);
    multigrid_halo(h_state.prolongation_operator_0);

    h_state.prolongation_operator_1(seqy_h, seqx_h) = deltaW2h_star_star_1;
    h_state.prolongation_operator_1.row(0) = h_state.prolongation_operator_1.row(2);
    h_state.prolongation_operator_1.row(1) = h_state.prolongation_operator_1.row(2);
    h_state.prolongation_operator_1.row(h_state.ncells_y-1) = h_state.prolongation_operator_1.row(h_state.ncells_y-3);
    h_state.prolongation_operator_1.row(h_state.ncells_y-2) = h_state.prolongation_operator_1.row(h_state.ncells_y-3);
    multigrid_halo(h_state.prolongation_operator_1);

    h_state.prolongation_operator_2(seqy_h, seqx_h) = deltaW2h_star_star_2;
    h_state.prolongation_operator_2.row(0) = h_state.prolongation_operator_2.row(2);
    h_state.prolongation_operator_2.row(1) = h_state.prolongation_operator_2.row(2);
    h_state.prolongation_operator_2.row(h_state.ncells_y-1) = h_state.prolongation_operator_2.row(h_state.ncells_y-3);
    h_state.prolongation_operator_2.row(h_state.ncells_y-2) = h_state.prolongation_operator_2.row(h_state.ncells_y-3);
    multigrid_halo(h_state.prolongation_operator_2);

    h_state.prolongation_operator_3(seqy_h, seqx_h) = deltaW2h_star_star_3;
    h_state.prolongation_operator_3.row(0) = h_state.prolongation_operator_3.row(2);
    h_state.prolongation_operator_3.row(1) = h_state.prolongation_operator_3.row(2);
    h_state.prolongation_operator_3.row(h_state.ncells_y-1) = h_state.prolongation_operator_3.row(h_state.ncells_y-3);
    h_state.prolongation_operator_3.row(h_state.ncells_y-2) = h_state.prolongation_operator_3.row(h_state.ncells_y-3);
    multigrid_halo(h_state.prolongation_operator_3);


    // Compute W_h_+
    h_state.W_0 += h_state.prolongation_operator_0;
    h_state.W_1 += h_state.prolongation_operator_1;
    h_state.W_2 += h_state.prolongation_operator_2;
    h_state.W_3 += h_state.prolongation_operator_3;
}