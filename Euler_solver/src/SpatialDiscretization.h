#ifndef SPATIALDISCRETIZATION_H
#define SPATIALDISCRETIZATION_H

#include <vector>

class SpatialDiscretization {
public:


    std::vector<std::vector<double>> OMEGA;
    std::vector<std::vector<std::vector<std::vector<double>>>> s;
    std::vector<std::vector<std::vector<double>>> Ds;
    std::vector<std::vector<std::vector<std::vector<double>>>> n;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<std::vector<double>>> R_c;
    std::vector<std::vector<std::vector<double>>> R_d;
    std::vector<std::vector<std::vector<double>>> R_d0;
    std::vector<std::vector<std::vector<double>>> restriction_operator;
    std::vector<std::vector<std::vector<double>>> prolongation_operator;
    std::vector<std::vector<std::vector<double>>> deltaW_2h;
    std::vector<std::vector<std::vector<std::vector<double>>>> flux;
    std::vector<std::vector<std::vector<std::vector<double>>>> D;
    std::vector<std::vector<std::vector<double>>> eps_2;
    std::vector<std::vector<std::vector<double>>> eps_4;
    std::vector<std::vector<double>> Lambda_I;
    std::vector<std::vector<double>> Lambda_J;
    std::vector<std::vector<std::vector<double>>> Lambda_S;

    std::vector<std::vector<double>> x, y;
    double rho, u, v, E, T, p, k2_coeff, k4_coeff;
    double T_ref, U_ref;
    int ny, nx;
    double alpha;

    SpatialDiscretization(const std::vector<std::vector<double>>& x,
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
                          double U_ref);

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
            T_ref = other.T_ref;
            U_ref = other.U_ref;
        }
        return *this;
    }

    std::tuple<double, double, double, double, double, double> conservative_variable_from_W(const std::vector<double>& W) const;

    void compute_dummy_cells();

    std::vector<double> FcDs(const std::vector<double>& W, const std::vector<double>& n, const double& Ds) const;

    double Lambdac(const std::vector<double>& W, const std::vector<double>& n, const double& Ds) const;

    void compute_Fc_DeltaS();
    std::tuple<double, double> compute_epsilon(const std::vector<double>& W_Im1, const std::vector<double>& W_I,
                                              const std::vector<double>& W_Ip1, const std::vector<double>& W_Ip2,
                                              double k2 = 1.0/4.0, double k4 = 1.0/64.0) const;

    void compute_lambda();

    void compute_dissipation();

    void compute_R_c();

    void compute_R_d();

    void run_odd();

    void run_even();
};



#endif //SPATIALDISCRETIZATION_H
