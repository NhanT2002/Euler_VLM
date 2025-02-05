//
// Created by hieun on 10/15/2024.
//

#ifndef TEMPORALDISCRETIZATION_H
#define TEMPORALDISCRETIZATION_H


#include <string>
#include "SpatialDiscretization.h"
#include <vector>

class TemporalDiscretization{
public:
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
    double rho, u, v, E, T, p;
    double T_ref, U_ref;

    SpatialDiscretization current_state;
    double sigma, k2_coeff, k4_coeff;
    int res_smoothing;

    TemporalDiscretization(const std::vector<std::vector<double>>& x,
                           const std::vector<std::vector<double>>& y,
                           const double& rho,
                           const double& u,
                           const double& v,
                           const double& E,
                           const double& T,
                           const double& p,
                           const double& T_ref,
                           const double& U_ref,
                           double sigma = 0.5,
                           int res_smoothing = 1,
                           double k2_coeff = 1.0,
                           double k4_coeff = 1.0);

    std::vector<std::vector<double>> compute_dt() const;

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> compute_abc() const;

    std::tuple<double, double> compute_eps(const std::vector<double>& W_IJ,
                  const double& OMEGA,
                  const std::vector<double>& n1,
                  const std::vector<double>& n2,
                  const std::vector<double>& n3,
                  const std::vector<double>& n4,
                  const double& Ds1,
                  const double& Ds2,
                  const double& Ds3,
                  const double& Ds4,
                  double psi = 0.125,
                  double rr = 2.) const;

    static std::vector<double> compute_L2_norm(const std::vector<std::vector<std::vector<double>>> &residuals);

    static void save_checkpoint(const std::vector<std::vector<std::vector<double>>>& q,
                         const std::vector<int>& iteration,
                         const std::vector<std::vector<double>>& Residuals,
                         const std::string& file_name = "checkpoint.txt");

    static std::tuple<std::vector<std::vector<std::vector<double>>>,std::vector<int>,
           std::vector<std::vector<double>>> load_checkpoint(const std::string& file_name);

    std::tuple<double, double, double> compute_coeff();

    std::tuple<std::vector<std::vector<std::vector<double>>>,
               std::vector<std::vector<std::vector<double>>>,
               std::vector<std::vector<double>>> RungeKutta(int it_max = 20000);

    void run();
};



#endif //TEMPORALDISCRETIZATION_H
