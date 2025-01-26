#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "TemporalDiscretization.h"
#include "SpatialDiscretization.h"

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

    SpatialDiscretization restriction_timestep(int it_max);

    // Coarse to fine grid
    SpatialDiscretization prolongation(SpatialDiscretization& h2_state, std::vector<std::vector<double>>& x_h, std::vector<std::vector<double>>& y_h);
};

#endif // MULTIGRID_H
