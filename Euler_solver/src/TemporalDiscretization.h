//
// Created by hieun on 10/15/2024.
//

#ifndef TEMPORALDISCRETIZATION_H
#define TEMPORALDISCRETIZATION_H


#include <string>
#include "SpatialDiscretization.h"
#include <vector>
#include <Eigen/Dense>

class TemporalDiscretization{
public:
    Eigen::ArrayXXd& x;
    Eigen::ArrayXXd& y;
    double rho, u, v, E, T, p;
    double Mach, U_ref;

    SpatialDiscretization current_state;
    double sigma, k2_coeff, k4_coeff;
    int res_smoothing;

    TemporalDiscretization(Eigen::ArrayXXd& x,
                            Eigen::ArrayXXd& y,
                            double rho,
                            double u,
                            double v,
                            double E,
                            double T,
                            double p,
                            double Mach,
                            double U_ref,
                            double sigma = 0.5,
                            int res_smoothing = 1,
                            double k2_coeff = 1.0,
                            double k4_coeff = 1.0);

    

    

    Eigen::ArrayXXd compute_dt() const;

    void Res() const;

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> compute_abc();

    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> R_star(const Eigen::ArrayXXd& dW_0, const Eigen::ArrayXXd& dW_1, const Eigen::ArrayXXd& dW_2, const Eigen::ArrayXXd& dW_3);

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

    Eigen::Array<double, 4, 1> compute_L2_norm(const Eigen::ArrayXXd &dW_0, const Eigen::ArrayXXd &dW_1, const Eigen::ArrayXXd &dW_2, const Eigen::ArrayXXd &dW_3);

    // static void save_checkpoint(const std::vector<std::vector<std::vector<double>>>& q,
    //                      const std::vector<int>& iteration,
    //                      const std::vector<std::vector<double>>& Residuals,
    //                      const std::string& file_name = "checkpoint.txt");

    // static std::tuple<std::vector<std::vector<std::vector<double>>>,std::vector<int>,
    //        std::vector<std::vector<double>>> load_checkpoint(const std::string& file_name);

    std::tuple<double, double, double> compute_coeff();

    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, std::vector<std::vector<double>>, std::vector<double>> RungeKutta(int it_max = 20000);

    void run();
};

std::vector<std::vector<double>> reshapeColumnWise(const std::vector<std::vector<double>>& input, int ny, int nx);



#endif //TEMPORALDISCRETIZATION_H
