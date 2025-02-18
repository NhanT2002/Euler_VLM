#ifndef SPATIALDISCRETIZATION_H
#define SPATIALDISCRETIZATION_H

#include <vector>
#include <Eigen/Dense>

class SpatialDiscretization {
public:

    Eigen::ArrayXXd OMEGA; // Volume of cell
    Eigen::ArrayXXd sx_x; // x face x component
    Eigen::ArrayXXd sx_y; // x face y component
    Eigen::ArrayXXd sy_x; // y face x component
    Eigen::ArrayXXd sy_y; // y face y component
    Eigen::ArrayXXd Ds_x; // x face length
    Eigen::ArrayXXd Ds_y; // y face length    
    Eigen::ArrayXXd nx_x; // x face normal x component    
    Eigen::ArrayXXd nx_y; // x face normal y component    
    Eigen::ArrayXXd ny_x; // y face normal x component    
    Eigen::ArrayXXd ny_y; // y face normal y component
    
    Eigen::ArrayXXd Ds_x_avg; // average x face of the cell
    Eigen::ArrayXXd Ds_y_avg; // average y face of the cell
    Eigen::ArrayXXd nx_x_avg; // average x face normal x component
    Eigen::ArrayXXd nx_y_avg; // average x face normal y component
    Eigen::ArrayXXd ny_x_avg; // average y face normal x component
    Eigen::ArrayXXd ny_y_avg; // average y face normal y component
    Eigen::ArrayXXd sx_x_avg; // average x face normal x component
    Eigen::ArrayXXd sx_y_avg; // average x face normal y component
    Eigen::ArrayXXd sy_x_avg; // average y face normal x component
    Eigen::ArrayXXd sy_y_avg; // average y face normal y component


    Eigen::ArrayXXd rho_cells; // Density
    Eigen::ArrayXXd u_cells; // x velocity
    Eigen::ArrayXXd v_cells; // y velocity
    Eigen::ArrayXXd E_cells; // Energy
    Eigen::ArrayXXd p_cells; // Pressure

    Eigen::ArrayXXd W_0; // rho
    Eigen::ArrayXXd W_1; // rho*u
    Eigen::ArrayXXd W_2; // rho*v
    Eigen::ArrayXXd W_3; // rho*E
    Eigen::ArrayXXd Rc_0; 
    Eigen::ArrayXXd Rc_1; 
    Eigen::ArrayXXd Rc_2; 
    Eigen::ArrayXXd Rc_3; 
    Eigen::ArrayXXd Rd_0; 
    Eigen::ArrayXXd Rd_1; 
    Eigen::ArrayXXd Rd_2; 
    Eigen::ArrayXXd Rd_3; 
    Eigen::ArrayXXd Rd0_0; 
    Eigen::ArrayXXd Rd0_1;
    Eigen::ArrayXXd Rd0_2;
    Eigen::ArrayXXd Rd0_3;

    Eigen::ArrayXXd fluxx_0; // x flux rho component
    Eigen::ArrayXXd fluxx_1; // x flux x momentum component
    Eigen::ArrayXXd fluxx_2; // x flux y momentum component
    Eigen::ArrayXXd fluxx_3; // x flux energy component
    Eigen::ArrayXXd fluxy_0; // y flux rho component
    Eigen::ArrayXXd fluxy_1; // y flux x momentum component
    Eigen::ArrayXXd fluxy_2; // y flux y momentum component
    Eigen::ArrayXXd fluxy_3; // y flux energy component
    
    Eigen::ArrayXXd dissipx_0; // x dissipation
    Eigen::ArrayXXd dissipx_1; // x dissipation
    Eigen::ArrayXXd dissipx_2; // x dissipation
    Eigen::ArrayXXd dissipx_3; // x dissipation
    Eigen::ArrayXXd dissipy_0; // y dissipation
    Eigen::ArrayXXd dissipy_1; // y dissipation
    Eigen::ArrayXXd dissipy_2; // y dissipation
    Eigen::ArrayXXd dissipy_3; // y dissipation
    // Eigen::ArrayXXd eps2_x;
    // Eigen::ArrayXXd eps2_y;
    // Eigen::ArrayXXd eps4_x;
    // Eigen::ArrayXXd eps4_y;
    Eigen::ArrayXXd Lambda_I;
    Eigen::ArrayXXd Lambda_J;

    Eigen::ArrayXXd restriction_operator_0;
    Eigen::ArrayXXd restriction_operator_1;
    Eigen::ArrayXXd restriction_operator_2;
    Eigen::ArrayXXd restriction_operator_3;
    Eigen::ArrayXXd forcing_function_0;
    Eigen::ArrayXXd forcing_function_1;
    Eigen::ArrayXXd forcing_function_2;
    Eigen::ArrayXXd forcing_function_3;
    Eigen::ArrayXXd prolongation_operator_0;
    Eigen::ArrayXXd prolongation_operator_1;
    Eigen::ArrayXXd prolongation_operator_2;
    Eigen::ArrayXXd prolongation_operator_3;
    Eigen::ArrayXXd deltaW2h_0;
    Eigen::ArrayXXd deltaW2h_1;
    Eigen::ArrayXXd deltaW2h_2;
    Eigen::ArrayXXd deltaW2h_3;
    Eigen::ArrayXXd W2h_0, W2h_1, W2h_2, W2h_3;
    

    Eigen::ArrayXXd x, y;
    double rho, u, v, E, T, p, k2_coeff, k4_coeff;
    double Mach, U_ref;
    int nvertex_y, nvertex_x, ncells_y, ncells_x, ncells_domain_y, ncells_domain_x;
    double alpha;

    SpatialDiscretization(const Eigen::ArrayXXd& x,
                          const Eigen::ArrayXXd& y,
                          double rho,
                          double u,
                          double v,
                          double E,
                          double T,
                          double p,
                          double k2_coeff,
                          double k4_coeff,
                          double Mach,
                          double U_ref);
    
    // Default constructor to initialize empty variables
    SpatialDiscretization()
        : rho(0), u(0), v(0), E(0), T(0), p(0), k2_coeff(0), k4_coeff(0),
        Mach(0), U_ref(0), alpha(0) {}

    // Custom copy constructor
    SpatialDiscretization(const SpatialDiscretization& other)
        : x(other.x), y(other.y), rho(other.rho), u(other.u), v(other.v),
          E(other.E), T(other.T), p(other.p), k2_coeff(other.k2_coeff),
          k4_coeff(other.k4_coeff), Mach(other.Mach), U_ref(other.U_ref), alpha(other.alpha) {}

    // Custom assignment operator
    SpatialDiscretization& operator=(const SpatialDiscretization& other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
            rho = other.rho;
            u = other.u;
            v = other.v;
            E = other.E;
            T = other.T;
            p = other.p;
            k2_coeff = other.k2_coeff;
            k4_coeff = other.k4_coeff;
            Mach = other.Mach;
            U_ref = other.U_ref;
            alpha = other.alpha;
        }
        return *this;
    }

    void update_W();

    void update_halo();

    void update_conservative_variables();

    void compute_dummy_cells();

    void compute_flux();

    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> FcDs(const Eigen::ArrayXXd& rhoo, const Eigen::ArrayXXd& uu, const Eigen::ArrayXXd& vv, const Eigen::ArrayXXd& EE, const Eigen::ArrayXXd& pp, 
                                            const Eigen::ArrayXXd& nx, const Eigen::ArrayXXd& ny, const Eigen::ArrayXXd& Ds);

    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> FcDs_wall(const Eigen::ArrayXXd& pp, const Eigen::ArrayXXd& nx, const Eigen::ArrayXXd& ny, const Eigen::ArrayXXd& Ds);


    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> compute_epsilon(const Eigen::ArrayXXd& p_Im1, const Eigen::ArrayXXd& p, const Eigen::ArrayXXd& p_Ip1, const Eigen::ArrayXXd& p_Ip2, double k2 = 1.0/4.0, double k4 = 1.0/64.0);

    void compute_lambda();

    void compute_dissipation();

    void compute_Rc();

    void compute_Rd();

    void update_Rd0();

    void run_odd();

    void run_even();

    std::tuple<double, double, double> compute_coeff();
};



#endif //SPATIALDISCRETIZATION_H
