#include "Multigrid.h"
#include "read_PLOT3D.h"

Multigrid::Multigrid(SpatialDiscretization h_state) : h_state(h_state) {}

// Fine to coarse grid
SpatialDiscretization Multigrid::restriction(SpatialDiscretization h_state) {
    
    // New mesh
    std::vector<std::vector<double>> x_2h, y_2h;
    for (int j = 0; j < (h_state.ny - 1)/2; ++j) {
        for (int i = 0; i < (hstate.nx - 1)/2; ++i) {

            const double &x1 = x[j][i];
            const double &x2 = x[j][i + 2];
            const double &x3 = x[j + 2][i + 2];
            const double &x4 = x[j + 2][i];
            const double &y1 = y[j][i];
            const double &y2 = y[j][i + 2];
            const double &y3 = y[j + 2][i + 2];
            const double &y4 = y[j + 2][i];


        }
    }
    // write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xyz");
    // write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xyz");

    return SpatialDiscretization 2h_state(x_2h, y_2h, h_state.rho, h_state.u, h_state.v, h_state.E, h_state.T, h_state.p, h_state.k2_coeff, h_state.k4_coeff, h_state.T_ref, h_state.U_ref);


}