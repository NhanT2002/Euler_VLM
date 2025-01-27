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

    // Constructor
    explicit Multigrid(SpatialDiscretization& h_state, double sigma = 0.5, int res_smoothing = 1, double k2_coeff = 1.0, double k4_coeff = 1.0);

    // Fine to coarse grid
    SpatialDiscretization restriction(SpatialDiscretization& h_state);

    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> restriction_timestep(SpatialDiscretization& h_state, int it_max);

    std::vector<std::vector<double>> compute_dt(SpatialDiscretization& current_state);

    std::tuple<double, double, double> compute_coeff(SpatialDiscretization& current_state);

    // Coarse to fine grid
    void prolongation(SpatialDiscretization& h2_state, SpatialDiscretization& h_state);
};

#endif // MULTIGRID_H
