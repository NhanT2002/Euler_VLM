#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "TemporalDiscretization.h"
#include "SpatialDiscretization.h"
#include <vector>

class Multigrid : public TemporalDiscretization {
public:
    SpatialDiscretization h_state;
    double sigma;
    int res_smoothing;
    double k2_coeff;
    double k4_coeff;
    bool multigrid_convergence;

    // Constructor
    explicit Multigrid(SpatialDiscretization& h_state, double sigma = 0.5, int res_smoothing = 1, double k2_coeff = 1.0, double k4_coeff = 1.0);

    // Mesh restriction
    SpatialDiscretization mesh_restriction(SpatialDiscretization& h_state);

    // Fine to coarse grid
    void restriction(SpatialDiscretization& h_state, SpatialDiscretization& h2_state);

    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> restriction_timestep(SpatialDiscretization& h_state, 
                                                                                                                    int it_max, 
                                                                                                                    int current_iteration=0, 
                                                                                                                    std::vector<double> multigrid_first_residual={});

    std::vector<std::vector<double>> compute_dt(SpatialDiscretization& current_state);

    std::tuple<double, double, double> compute_coeff(SpatialDiscretization& current_state);

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> compute_abc(SpatialDiscretization& current_state);

    // Coarse to fine grid
    void prolongation(SpatialDiscretization& h2_state, SpatialDiscretization& h_state);
};

#endif // MULTIGRID_H
