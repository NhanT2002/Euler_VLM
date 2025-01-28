#include "SpatialDiscretization.h"
#include "vector_helper.h"
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>

void halo(Eigen::ArrayXXd& array) {
    int im1 = array.cols() - 3;
    int im2 = array.cols() - 4;
    int ip1 = 2;
    int ip2 = 3;

    array.col(1) = array.col(im1);
    array.col(0) = array.col(im2);
    array.col(array.cols()-2) = array.col(ip1);
    array.col(array.cols()-1) = array.col(ip2);
}

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
    ncells_domain_y = nvertex_y - 1; // number of cells in real domain (without dummys) 
    ncells_domain_x = nvertex_x - 1; // number of cells in real domain (without dummys)
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

    Ds_x_avg.resize(ncells_y, ncells_x);
    Ds_y_avg.resize(ncells_y, ncells_x);
    nx_x_avg.resize(ncells_y, ncells_x);
    nx_y_avg.resize(ncells_y, ncells_x);
    ny_x_avg.resize(ncells_y, ncells_x);
    ny_y_avg.resize(ncells_y, ncells_x);
    

    rho_cells.resize(ncells_y, ncells_x); 
    u_cells.resize(ncells_y, ncells_x); 
    v_cells.resize(ncells_y, ncells_x); 
    E_cells.resize(ncells_y, ncells_x); 
    p_cells.resize(ncells_y, ncells_x); 
    W_0.resize(ncells_y, ncells_x);
    W_1.resize(ncells_y, ncells_x); 
    W_2.resize(ncells_y, ncells_x);
    W_3.resize(ncells_y, ncells_x);

    Rc_0.resize(ncells_domain_y, ncells_domain_x); 
    Rc_1.resize(ncells_domain_y, ncells_domain_x); 
    Rc_2.resize(ncells_domain_y, ncells_domain_x); 
    Rc_3.resize(ncells_domain_y, ncells_domain_x); 
    Rd_0.resize(ncells_domain_y, ncells_domain_x); 
    Rd_1.resize(ncells_domain_y, ncells_domain_x); 
    Rd_2.resize(ncells_domain_y, ncells_domain_x); 
    Rd_3.resize(ncells_domain_y, ncells_domain_x); 
    Rd0_0.resize(ncells_domain_y, ncells_domain_x); 
    Rd0_1.resize(ncells_domain_y, ncells_domain_x);
    Rd0_2.resize(ncells_domain_y, ncells_domain_x);
    Rd0_3.resize(ncells_domain_y, ncells_domain_x);

    fluxx_0.resize(ncells_domain_y, ncells_domain_x);
    fluxx_1.resize(ncells_domain_y, ncells_domain_x); 
    fluxx_2.resize(ncells_domain_y, ncells_domain_x);
    fluxx_3.resize(ncells_domain_y, ncells_domain_x);
    fluxy_0.resize(ncells_domain_y, ncells_domain_x); 
    fluxy_1.resize(ncells_domain_y, ncells_domain_x); 
    fluxy_2.resize(ncells_domain_y, ncells_domain_x); 
    fluxy_3.resize(ncells_domain_y, ncells_domain_x); 
    
    dissipx_0.resize(ncells_domain_y, ncells_domain_x);
    dissipx_1.resize(ncells_domain_y, ncells_domain_x); 
    dissipx_2.resize(ncells_domain_y, ncells_domain_x); 
    dissipx_3.resize(ncells_domain_y, ncells_domain_x);
    dissipy_0.resize(ncells_domain_y, ncells_domain_x); 
    dissipy_1.resize(ncells_domain_y, ncells_domain_x); 
    dissipy_2.resize(ncells_domain_y, ncells_domain_x); 
    dissipy_3.resize(ncells_domain_y, ncells_domain_x); 
    eps2_x.resize(ncells_domain_y, ncells_domain_x);
    eps2_y.resize(ncells_domain_y, ncells_domain_x);
    eps4_x.resize(ncells_domain_y, ncells_domain_x);
    eps4_y.resize(ncells_domain_y, ncells_domain_x);
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
    // Halo cells
    halo(OMEGA);
    halo(Ds_x);
    halo(Ds_y);
    halo(sx_x);
    halo(sx_y);
    halo(sy_x);
    halo(sy_y);
    halo(nx_x);
    halo(nx_y);
    halo(ny_x);
    halo(ny_y);


    // Dummys same geometry as closest cell
    OMEGA.row(0) = OMEGA.row(2);
    OMEGA.row(1) = OMEGA.row(2);
    OMEGA.row(ncells_y-2) = OMEGA.row(ncells_y-3);
    OMEGA.row(ncells_y-1) = OMEGA.row(ncells_y-3);

    nx_x.row(0) = nx_x.row(2);
    nx_x.row(1) = nx_x.row(2);
    nx_y.row(0) = nx_y.row(2);
    nx_y.row(1) = nx_y.row(2);
    // nx_x.row(ncells_y-2) compute below

    ny_x.row(0) = ny_x.row(2);
    ny_x.row(1) = ny_x.row(2);
    ny_x.row(ncells_y-2) = ny_x.row(ncells_y-3);
    ny_x.row(ncells_y-1) = ny_x.row(ncells_y-3);
    ny_y.row(0) = ny_y.row(2);
    ny_y.row(1) = ny_y.row(2);
    ny_y.row(ncells_y-2) = ny_y.row(ncells_y-3);
    ny_y.row(ncells_y-1) = ny_y.row(ncells_y-3);


    Ds_x.row(0) = Ds_x.row(2);
    Ds_x.row(1) = Ds_x.row(2);
    Ds_x.row(ncells_y-2) = Ds_x.row(ncells_y-3);
    Ds_x.row(ncells_y-1) = Ds_x.row(ncells_y-3);
    Ds_y.row(0) = Ds_y.row(2);
    Ds_y.row(1) = Ds_y.row(2);
    Ds_y.row(ncells_y-2) = Ds_y.row(ncells_y-3);
    Ds_y.row(ncells_y-1) = Ds_y.row(ncells_y-3);

    // Compute normal vector of x face in first farfield dummy cell
    int j = ncells_y - 2;
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
        Ds_x(j+1, i) = std::sqrt(Sx_x*Sx_x + Sx_y*Sx_y);

        nx_x(j, i) = Sx_x / Ds_x(j, i);
        nx_y(j, i) = Sx_y / Ds_x(j, i);
        nx_x(j+1, i) = Sx_x / Ds_x(j, i);
        nx_y(j+1, i) = Sx_y / Ds_x(j, i);
    }

    // Compute average cell geometry
    auto seq_y = Eigen::seq(1, ncells_y-2);
    auto seq_yp1 = Eigen::seq(2, ncells_y-1);
    auto seq_x = Eigen::seq(1, ncells_x-2);
    auto seq_xp1 = Eigen::seq(2, ncells_x-1);
    Ds_x_avg(seq_y, seq_x) = 0.5*(Ds_x(seq_yp1, seq_x) + Ds_x(seq_y, seq_x));
    Ds_y_avg(seq_y, seq_x) = 0.5*(Ds_y(seq_y, seq_xp1) + Ds_y(seq_y, seq_x));
    nx_x_avg(seq_y, seq_x) = 0.5*(nx_x(seq_yp1, seq_x) + nx_x(seq_y, seq_x));
    nx_y_avg(seq_y, seq_x) = 0.5*(nx_y(seq_yp1, seq_x) + nx_y(seq_y, seq_x));
    ny_x_avg(seq_y, seq_x) = 0.5*(ny_x(seq_y, seq_xp1) + ny_x(seq_y, seq_x));
    ny_y_avg(seq_y, seq_x) = 0.5*(ny_y(seq_y, seq_xp1) + ny_y(seq_y, seq_x));


}

void SpatialDiscretization::update_W() {
    W_0 = rho_cells;
    W_1 = rho_cells * u_cells;
    W_2 = rho_cells * v_cells;
    W_3 = rho_cells * E_cells;
}

void SpatialDiscretization::update_halo() {
    int im1 = ncells_x - 3;
    int im2 = ncells_x - 4;
    int ip1 = 2;
    int ip2 = 3;

    rho_cells.col(1) = rho_cells.col(im1);
    u_cells.col(1) = u_cells.col(im1);
    v_cells.col(1) = v_cells.col(im1);
    E_cells.col(1) = E_cells.col(im1);
    p_cells.col(1) = p_cells.col(im1);

    rho_cells.col(0) = rho_cells.col(im2);
    u_cells.col(0) = u_cells.col(im2);
    v_cells.col(0) = v_cells.col(im2);
    E_cells.col(0) = E_cells.col(im2);
    p_cells.col(0) = p_cells.col(im2);

    rho_cells.col(ncells_x-2) = rho_cells.col(ip1);
    u_cells.col(ncells_x-2) = u_cells.col(ip1);
    v_cells.col(ncells_x-2) = v_cells.col(ip1);
    E_cells.col(ncells_x-2) = E_cells.col(ip1);
    p_cells.col(ncells_x-2) = p_cells.col(ip1);

    rho_cells.col(ncells_x-1) = rho_cells.col(ip2);
    u_cells.col(ncells_x-1) = u_cells.col(ip2);
    v_cells.col(ncells_x-1) = v_cells.col(ip2);
    E_cells.col(ncells_x-1) = E_cells.col(ip2);
    p_cells.col(ncells_x-1) = p_cells.col(ip2);

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
    p_cells.row(1) = p_dummy;

    rho_cells.row(0) = rho_dummy;
    u_cells.row(0) = u_dummy;
    v_cells.row(0) = v_dummy;
    E_cells.row(0) = E_dummy;
    p_cells.row(0) = p_dummy;

    // Far field
    int j_last_cells = ncells_y - 3;
    int j = j_last_cells + 1;
    int jj = j + 1;
    Eigen::Array<double, 1, Eigen::Dynamic> c_cells = (1.4*p_cells(j_last_cells, Eigen::all)/rho_cells(j_last_cells, Eigen::all)).sqrt(); // speed of sound
    Eigen::Array<double, 1, Eigen::Dynamic> M_cells = (u_cells(j_last_cells, Eigen::all).square() + v_cells(j_last_cells, Eigen::all).square()).sqrt()/c_cells; // Mach number
    Eigen::Array<double, 1, Eigen::Dynamic> nx_cells = -1*nx_x(j, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> ny_cells = -1*nx_y(j, Eigen::all);

    for (int i=2; i<ncells_x-2; i++) {
        double& rho_d = rho_cells(j_last_cells, i);
        double& u_d = u_cells(j_last_cells, i);
        double& v_d = v_cells(j_last_cells, i);
        double& E_d = E_cells(j_last_cells, i);
        double& p_d = p_cells(j_last_cells, i);
        double& c = c_cells(0, i);
        double& M = M_cells(0, i);
        double& nx = nx_cells(0, i);
        double& ny = ny_cells(0, i);

        if (u_d*nx + v_d*ny > 0) { // out of cell
            if (M >= 1) { // supersonic
                rho_cells(j, i) = rho_cells(j_last_cells, i);
                u_cells(j, i) = u_cells(j_last_cells, i);
                v_cells(j, i) = v_cells(j_last_cells, i);
                E_cells(j, i) = E_cells(j_last_cells, i);
                p_cells(j, i) = p_cells(j_last_cells, i);

                rho_cells(jj, i) = rho_cells(j_last_cells, i);
                u_cells(jj, i) = u_cells(j_last_cells, i);
                v_cells(jj, i) = v_cells(j_last_cells, i);
                E_cells(jj, i) = E_cells(j_last_cells, i);
                p_cells(jj, i) = p_cells(j_last_cells, i);
            } 
            else { // subsonic
                double p_b = this->p;
                double rho_b = rho_d + (p_b - p_d)/(c*c);
                double u_b = u_d + nx*(p_d - p_b)/(rho_d*c);
                double v_b = v_d + ny*(p_d - p_b)/(rho_d*c);
                double E_b = p_b/(rho_b*(1.4-1)) + 0.5*(u_b*u_b + v_b*v_b);

                rho_cells(j, i) = 2*rho_b - rho_d;
                u_cells(j, i) = (2*(u_b*rho_b) - (u_d*rho_d))/rho_cells(j, i);
                v_cells(j, i) = (2*(v_b*rho_b) - (v_d*rho_d))/rho_cells(j, i);
                E_cells(j, i) = (2*(E_b*rho_b) - (E_d*rho_d))/rho_cells(j, i);
                p_cells(j, i) = (1.4-1)*rho_cells(j, i)*(E_cells(j, i) - 0.5*(u_cells(j, i)*u_cells(j, i) + v_cells(j, i)*v_cells(j, i)));

                rho_cells(jj, i) = rho_cells(j, i);
                u_cells(jj, i) = u_cells(j, i);
                v_cells(jj, i) = v_cells(j, i);
                E_cells(jj, i) = E_cells(j, i);
                p_cells(jj, i) = p_cells(j, i);

                }
        }
        else { // in cell
            if (M >=1) { // supersonic
                rho_cells(j, i) = this->rho;
                u_cells(j, i) = this->u;
                v_cells(j, i) = this->v;
                E_cells(j, i) = this->E;
                p_cells(j, i) = this->p;

                rho_cells(jj, i) = rho_cells(j, i);
                u_cells(jj, i) = u_cells(j, i);
                v_cells(jj, i) = v_cells(j, i);
                E_cells(jj, i) = E_cells(j, i);
                p_cells(jj, i) = p_cells(j, i);
            }
            else { // subsonic
                double p_b = 0.5*(this->p + p_d - rho_d*c*(nx*(this->u - u_d) + ny*(this->v - v_d)));
                double rho_b = this->rho + (p_b - this->p)/(c*c);
                double u_b = this->u - nx*(this->p - p_b)/(rho_d*c);
                double v_b = this->v - ny*(this->p - p_b)/(rho_d*c);
                double E_b = p_b/(rho_b*(1.4-1)) + 0.5*(u_b*u_b + v_b*v_b);

                rho_cells(j, i) = 2*rho_b - rho_d;
                u_cells(j, i) = (2*(u_b*rho_b) - (u_d*rho_d))/rho_cells(j, i);
                v_cells(j, i) = (2*(v_b*rho_b) - (v_d*rho_d))/rho_cells(j, i);
                E_cells(j, i) = (2*(E_b*rho_b) - (E_d*rho_d))/rho_cells(j, i);
                p_cells(j, i) = (1.4-1)*rho_cells(j, i)*(E_cells(j, i) - 0.5*(u_cells(j, i)*u_cells(j, i) + v_cells(j, i)*v_cells(j, i)));

                rho_cells(jj, i) = rho_cells(j, i);
                u_cells(jj, i) = u_cells(j, i);
                v_cells(jj, i) = v_cells(j, i);
                E_cells(jj, i) = E_cells(j, i);
                p_cells(jj, i) = p_cells(j, i);
            }
        }
    }
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> SpatialDiscretization::FcDs(const Eigen::ArrayXXd& rhoo, const Eigen::ArrayXXd& uu, const Eigen::ArrayXXd& vv, const Eigen::ArrayXXd& EE, const Eigen::ArrayXXd& pp, 
                                            const Eigen::ArrayXXd& nx, const Eigen::ArrayXXd& ny, const Eigen::ArrayXXd& Ds) {
    Eigen::ArrayXXd V = nx*uu + ny*vv;
    Eigen::ArrayXXd H = EE + pp/rhoo;

    return {rhoo*V*Ds, (rhoo*vv*V + nx*pp)*Ds, (rhoo*vv*V + ny*pp)*Ds, rhoo*H*V*Ds};
}

void SpatialDiscretization::compute_flux() {
    auto seqy = Eigen::seq(2, ncells_y-3);
    auto seqx = Eigen::seq(2, ncells_x-3);
    auto seqy_m1 = Eigen::seq(1, ncells_y-4);
    auto seqx_m1 = Eigen::seq(1, ncells_x-4);

    // y direction
    Eigen::ArrayXXd avg_rho = 0.5*(rho_cells(seqy, seqx) + rho_cells(seqy_m1, seqx));
    Eigen::ArrayXXd avg_u = 0.5*(u_cells(seqy, seqx) + u_cells(seqy_m1, seqx));
    Eigen::ArrayXXd avg_v = 0.5*(v_cells(seqy, seqx) + v_cells(seqy_m1, seqx));
    Eigen::ArrayXXd avg_E = 0.5*(E_cells(seqy, seqx) + E_cells(seqy_m1, seqx));
    Eigen::ArrayXXd avg_p = 0.5*(p_cells(seqy, seqx) + p_cells(seqy_m1, seqx));
    auto [Fcy0, Fcy1, Fcy2, Fcy3] = FcDs(avg_rho, avg_u, avg_v, avg_E, avg_p, nx_x(seqy, seqx), nx_y(seqy, seqx), Ds_x(seqy, seqx));

    fluxy_0 = Fcy0;
    fluxy_1 = Fcy1;
    fluxy_2 = Fcy2;
    fluxy_3 = Fcy3;

    // x direction
    avg_rho = 0.5*(rho_cells(seqy, seqx) + rho_cells(seqy, seqx_m1));
    avg_u = 0.5*(u_cells(seqy, seqx) + u_cells(seqy, seqx_m1));
    avg_v = 0.5*(v_cells(seqy, seqx) + v_cells(seqy, seqx_m1));
    avg_E = 0.5*(E_cells(seqy, seqx) + E_cells(seqy, seqx_m1));
    avg_p = 0.5*(p_cells(seqy, seqx) + p_cells(seqy, seqx_m1));
    auto [Fcx0, Fcx1, Fcx2, Fcx3] = FcDs(avg_rho, avg_u, avg_v, avg_E, avg_p, ny_x(seqy, seqx), ny_y(seqy, seqx), Ds_y(seqy, seqx));
    fluxx_0 = Fcx0;
    fluxx_1 = Fcx1;
    fluxx_2 = Fcx2;
    fluxx_3 = Fcx3;

}

void SpatialDiscretization::compute_lambda() {
    Eigen::ArrayXXd c_cells = 1.4*p_cells/rho_cells.sqrt();
    Lambda_I = ((ny_x_avg*u_cells + ny_y_avg*v_cells).abs() + c_cells)*Ds_y_avg;
    Lambda_J = ((nx_x_avg*u_cells + nx_y_avg*v_cells).abs() + c_cells)*Ds_x_avg;
    halo(Lambda_I);
    halo(Lambda_J);
}

void SpatialDiscretization::run_odd() {
    SpatialDiscretization::compute_dummy_cells();
    SpatialDiscretization::compute_lambda();
    SpatialDiscretization::compute_flux();
    SpatialDiscretization::compute_R_c();
}

void SpatialDiscretization::run_even() {
    SpatialDiscretization::compute_dummy_cells();
    SpatialDiscretization::compute_lambda();
    SpatialDiscretization::compute_flux();
    SpatialDiscretization::compute_dissipation();
    SpatialDiscretization::compute_R_c();
    SpatialDiscretization::compute_R_d();
}

