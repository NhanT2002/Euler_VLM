#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "TemporalDiscretization.h"
#include "SpatialDiscretization.h"

class Multigrid {
public :
    SpatialDiscretization h_state;
    Multigrid(SpatialDiscretization h_state);

    // Fine to coarse grid
    SpatialDiscretization restriction(SpatialDiscretization h_state);

    // Coarse to fine grid
    SpatialDiscretization prolongation(SpatialDiscretization h_state);
}


#endif //MULTIGRID_H