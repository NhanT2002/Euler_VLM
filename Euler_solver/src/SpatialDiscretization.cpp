#include "SpatialDiscretization.h"
#include "vector_helper.h"
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>

SpatialDiscretization::SpatialDiscretization(const Eigen::ArrayXXd& x,
                                             const Eigen::ArrayXXd& y,
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
    nvertex_y = x.rows();
    nvertex_x = x.cols();
    ncells_y = nvertex_y + 3; // nvertex_y - 1 + 4 for dummy cells
    ncells_x = nvertex_x + 3; // nvertex_x - 1 + 4 for halo cells
    alpha = std::atan2(v, u);

    OMEGA.resize(ncells_y, ncells_x);
    sx_x.resize(ncells_y, ncells_x);
    sx_y.resize(ncells_y, ncells_x);
    sy_x.resize(ncells_y, ncells_x);
    sy_y.resize(ncells_y, ncells_x);
    Ds_x.resize(ncells_y, ncells_x); 
    Ds_y.resize(ncells_y, ncells_x); 
    nx_x.resize(ncells_y, ncells_x);
    nx_y.resize(ncells_y, ncells_x); 
    ny_x.resize(ncells_y, ncells_x); 
    ny_y.resize(ncells_y, ncells_x); 

    rho_cells.resize(ncells_y, ncells_x); 
    u_cells.resize(ncells_y, ncells_x); 
    v_cells.resize(ncells_y, ncells_x); 
    E_cells.resize(ncells_y, ncells_x); 
    p_cells.resize(ncells_y, ncells_x); 
    W_0.resize(ncells_y, ncells_x);
    W_1.resize(ncells_y, ncells_x); 
    W_2.resize(ncells_y, ncells_x);
    W_3.resize(ncells_y, ncells_x);
    Rc_0.resize(ncells_y, ncells_x); 
    Rc_1.resize(ncells_y, ncells_x); 
    Rc_2.resize(ncells_y, ncells_x); 
    Rc_3.resize(ncells_y, ncells_x); 
    Rd_0.resize(ncells_y, ncells_x); 
    Rd_1.resize(ncells_y, ncells_x); 
    Rd_2.resize(ncells_y, ncells_x); 
    Rd_3.resize(ncells_y, ncells_x); 
    Rd0_0.resize(ncells_y, ncells_x); 
    Rd0_1.resize(ncells_y, ncells_x);
    Rd0_2.resize(ncells_y, ncells_x);
    Rd0_3.resize(ncells_y, ncells_x);

    fluxx_0.resize(ncells_y, ncells_x);
    fluxx_1.resize(ncells_y, ncells_x); 
    fluxx_2.resize(ncells_y, ncells_x);
    fluxx_3.resize(ncells_y, ncells_x);
    fluxy_0.resize(ncells_y, ncells_x); 
    fluxy_1.resize(ncells_y, ncells_x); 
    fluxy_2.resize(ncells_y, ncells_x); 
    fluxy_3.resize(ncells_y, ncells_x); 
    
    dissipx_0.resize(ncells_y, ncells_x);
    dissipx_1.resize(ncells_y, ncells_x); 
    dissipx_2.resize(ncells_y, ncells_x); 
    dissipx_3.resize(ncells_y, ncells_x);
    dissipy_0.resize(ncells_y, ncells_x); 
    dissipy_1.resize(ncells_y, ncells_x); 
    dissipy_2.resize(ncells_y, ncells_x); 
    dissipy_3.resize(ncells_y, ncells_x); 
    eps2_x.resize(ncells_y, ncells_x);
    eps2_y.resize(ncells_y, ncells_x);
    eps4_x.resize(ncells_y, ncells_x);
    eps4_y.resize(ncells_y, ncells_x);
    Lambda_I.resize(ncells_y, ncells_x);
    Lambda_J.resize(ncells_y, ncells_x);

    restriction_operator_0.resize(ncells_y, ncells_x);
    restriction_operator_1.resize(ncells_y, ncells_x);
    restriction_operator_2.resize(ncells_y, ncells_x);
    restriction_operator_3.resize(ncells_y, ncells_x);
    forcing_function_0.resize(ncells_y, ncells_x);
    forcing_function_1.resize(ncells_y, ncells_x);
    forcing_function_2.resize(ncells_y, ncells_x);
    forcing_function_3.resize(ncells_y, ncells_x);
    prolongation_operator_0.resize(ncells_y, ncells_x);
    prolongation_operator_1.resize(ncells_y, ncells_x);
    prolongation_operator_2.resize(ncells_y, ncells_x);
    prolongation_operator_3.resize(ncells_y, ncells_x);
    deltaW2h_0.resize(ncells_y, ncells_x);
    deltaW2h_1.resize(ncells_y, ncells_x);
    deltaW2h_2.resize(ncells_y, ncells_x);
    deltaW2h_3.resize(ncells_y, ncells_x);

    for (int j=2; j<ncells_y-2; j++) {
        for (int i=2; i<ncells_x-2; i++) {
            int jj = j - 2;
            int ii = i - 2;
            int jjp1 = jj + 1;
            int iip1 = ii + 1;    

            const double &x1 = x(jj, ii);
            const double &x2 = x(jj, iip1);
            const double &x3 = x(jjp1, iip1);
            const double &x4 = x(jjp1, ii);
            const double &y1 = y(jj, ii);
            const double &y2 = y(jj, iip1);
            const double &y3 = y(jjp1, iip1);
            const double &y4 = y(jjp1, ii);

            OMEGA(j, i) = 0.5 * ((x1-x3)*(y2-y4) + (x4-x2)*(y1-y3));

            double Sx_x = y2 - y1;
            double Sx_y = x1 - x2;
            double Sy_x = y1 - y4;
            double Sy_y = x4 - x1;

            Ds_x(j, i) = std::sqrt(Sx_x*Sx_x + Sx_y*Sx_y);
            Ds_y(j, i) = std::sqrt(Sy_x*Sy_x + Sy_y*Sy_y);

            nx_x(j, i) = Sx_x / Ds_x(j, i);
            nx_y(j, i) = Sx_y / Ds_x(j, i);
            ny_x(j, i) = Sy_x / Ds_y(j, i);
            ny_y(j, i) = Sy_y / Ds_y(j, i);

            rho_cells(j, i) = rho;
            u_cells(j, i) = u;
            v_cells(j, i) = v;
            E_cells(j, i) = E;
            p_cells(j, i) = p;
        }
    }

    // Far field normal
    int j = ncells_y - 2; // Compute normal vector of x face in first farfield dummy cell
    int jj = j - 2;
    std::cout << "j: " << j << std::endl;
    for (int i=2; i<ncells_x-2; i++) {
        const double &x1 = x(jj, i-2);
        const double &x2 = x(jj, i-1); // i-2+1
        const double &y1 = y(jj, i-2);
        const double &y2 = y(jj, i-1); // i-2+1

        double Sx_x = y2 - y1;
        double Sx_y = x1 - x2;

        Ds_x(j, i) = std::sqrt(Sx_x*Sx_x + Sx_y*Sx_y);

        nx_x(j, i) = Sx_x / Ds_x(j, i);
        nx_y(j, i) = Sx_y / Ds_x(j, i);
    }

}

void SpatialDiscretization::update_W() {
    W_0 = rho_cells;
    W_1 = rho_cells * u_cells;
    W_2 = rho_cells * v_cells;
    W_3 = rho_cells * E_cells;
}

void SpatialDiscretization::compute_dummy_cells() {
    // Solid wall
    Eigen::Array<double, 1, Eigen::Dynamic> V = nx_x(2, Eigen::all)*u_cells(2, Eigen::all) + nx_y(2, Eigen::all)*v_cells(2, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> u_dummy = u_cells(2, Eigen::all) - 2*V*nx_x(2, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> v_dummy = v_cells(2, Eigen::all) - 2*V*nx_y(2, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> p_dummy = p_cells(2, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> rho_dummy = rho_cells(2, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> E_dummy = p_dummy/((1.4-1)*rho_dummy) + 0.5*(u_dummy*u_dummy + v_dummy*v_dummy);
    
    rho_cells.row(1) = rho_dummy;
    u_cells.row(1) = u_dummy;
    v_cells.row(1) = v_dummy;
    E_cells.row(1) = E_dummy;

    rho_cells.row(0) = rho_dummy;
    u_cells.row(0) = u_dummy;
    v_cells.row(0) = v_dummy;
    E_cells.row(0) = E_dummy;
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

