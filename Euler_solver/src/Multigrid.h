#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "SpatialDiscretization.h"
#include <vector>

class Multigrid {
public:
    SpatialDiscretization h_state;
    double sigma;
    int res_smoothing;
    double k2_coeff;
    double k4_coeff;
    bool multigrid_convergence;

    // Constructor
    Multigrid(SpatialDiscretization& h_state, double sigma = 0.5, int res_smoothing = 1, double k2_coeff = 1.0, double k4_coeff = 1.0);

    // Mesh restriction
    SpatialDiscretization mesh_restriction(SpatialDiscretization& h_state);

    // Fine to coarse grid
    void restriction(SpatialDiscretization& h_state, SpatialDiscretization& h2_state);

    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, std::vector<std::vector<double>>> restriction_timestep(SpatialDiscretization& h_state, 
                                                                                                                    int it_max, 
                                                                                                                    int current_iteration=0);

    Eigen::ArrayXXd compute_dt(SpatialDiscretization& current_state);

    Eigen::Array<double, 4, 1> compute_L2_norm(const Eigen::ArrayXXd &dW_0, const Eigen::ArrayXXd &dW_1, const Eigen::ArrayXXd &dW_2, const Eigen::ArrayXXd &dW_3);

    std::tuple<double, double, double> compute_coeff(SpatialDiscretization& current_state);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> compute_abc(SpatialDiscretization& current_state);

    std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> R_star(SpatialDiscretization& current_state, const Eigen::ArrayXXd& dW_0, const Eigen::ArrayXXd& dW_1, const Eigen::ArrayXXd& dW_2, const Eigen::ArrayXXd& dW_3);


    // Coarse to fine grid
    void prolongation(SpatialDiscretization& h2_state, SpatialDiscretization& h_state);

    void prolongation_smooth(SpatialDiscretization& h2_state, SpatialDiscretization& h_state);
};

#endif // MULTIGRID_H
