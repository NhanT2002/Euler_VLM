#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "TemporalDiscretization.h"
#include "SpatialDiscretization.h"

class Multigrid : public TemporalDiscretization {
public:
    SpatialDiscretization h_state;

    // Constructor
    explicit Multigrid(SpatialDiscretization& h_state);

    // Fine to coarse grid
    SpatialDiscretization restriction(SpatialDiscretization& h_state);

    // Coarse to fine grid
    SpatialDiscretization prolongation(SpatialDiscretization& h2_state, std::vector<std::vector<double>>& x_h, std::vector<std::vector<double>>& y_h);
};

#endif // MULTIGRID_H
