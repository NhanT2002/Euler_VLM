#include "Multigrid.h"
#include "read_PLOT3D.h"

// Constructor implementation
Multigrid::Multigrid(SpatialDiscretization& h_state) : h_state(h_state) {}

// Restriction implementation (fine to coarse grid)
SpatialDiscretization Multigrid::restriction(SpatialDiscretization& h_state) {
    std::vector<std::vector<double>> x_2h(h_state.ny / 2 + 1, std::vector<double>(h_state.nx / 2 + 1));
    std::vector<std::vector<double>> y_2h(h_state.ny / 2 + 1, std::vector<double>(h_state.nx / 2 + 1));
    for (int j = 0; j < (h_state.ny - 1)/2 + 1; ++j) {
        for (int i = 0; i < (h_state.nx - 1)/2 + 1; ++i) {
            x_2h[j][i] = h_state.x[2 * j][2 * i];
            y_2h[j][i] = h_state.y[2 * j][2 * i];
        }
    }

    write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xy");
    write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xy");

    SpatialDiscretization h2_state(x_2h, y_2h, h_state.rho, h_state.u, h_state.v, h_state.E,
                                    h_state.T, h_state.p, h_state.k2_coeff, h_state.k4_coeff,
                                    h_state.T_ref, h_state.U_ref);

    return h2_state;
}
