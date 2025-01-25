#include "Multigrid.h"
#include "read_PLOT3D.h"
#include "vector_helper.h"
#include <iostream>
#include <vector>



// Constructor implementation
Multigrid::Multigrid(SpatialDiscretization& h_state) : h_state(h_state) {}

// Restriction implementation (fine to coarse grid)
SpatialDiscretization Multigrid::restriction(SpatialDiscretization& h_state) {
    int ny_2h = (h_state.ny+1) / 2;
    int nx_2h = (h_state.nx+1) / 2;
    std::vector<std::vector<double>> x_2h(ny_2h, std::vector<double>(nx_2h));
    std::vector<std::vector<double>> y_2h(ny_2h, std::vector<double>(nx_2h));
    for (int j = 0; j < (h_state.ny - 1)/2 + 1; ++j) {
        for (int i = 0; i < (h_state.nx - 1)/2 + 1; ++i) {
            x_2h[j][i] = h_state.x[2 * j][2 * i];
            y_2h[j][i] = h_state.y[2 * j][2 * i];
        }
    }

    // Verify the mesh restriction
    write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xy");
    write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xy");

    SpatialDiscretization h2_state(x_2h, y_2h, h_state.rho, h_state.u, h_state.v, h_state.E,
                                    h_state.T, h_state.p, h_state.k2_coeff, h_state.k4_coeff,
                                    h_state.T_ref, h_state.U_ref);

    
    for (int j=2; j < ny_2h-1+2; ++j) {
        for (int i=0; i < nx_2h-1; ++i) {
            // Transfer operators for the cell-centered scheme
            std::cout << j << " " << i << std::endl;
            std::cout << "h_state W" << std::endl;
            printVector(h_state.W[2*j][2*i]);
            printVector(h_state.W[2*j][2*i+1]);
            printVector(h_state.W[2*+1][2*i+1]);
            printVector(h_state.W[2*j+1][2*i]);
            h2_state.W[j][i] = vector_scale(1/(h_state.OMEGA[2*j][2*i] + h_state.OMEGA[2*j][2*i+1] + h_state.OMEGA[2*j+1][2*i+1] + h_state.OMEGA[2*j+1][2*i]),
                                vector_add(vector_add(vector_scale(h_state.OMEGA[2*j][2*i], h_state.W[2*j][2*i]), vector_scale(h_state.OMEGA[2*j][2*i+1], h_state.W[2*j][2*i+1])), 
                                vector_add(vector_scale(h_state.OMEGA[2*j+1][2*i+1], h_state.W[2*j+1][2*i+1]), vector_scale(h_state.OMEGA[2*j+1][2*i], h_state.W[2*j+1][2*i]))));
            std::cout << "h2_state W" << std::endl;
            printVector(h2_state.W[j][i]);
            std::cout << " " << std::endl;
            
            // Restriction operator
            std::cout << "h_state residual" << std::endl;
            printVector(h_state.R_c[2*(j-2)][2*i]);
            printVector(h_state.R_c[2*(j-2)][2*i+1]);
            printVector(h_state.R_c[2*(j-2)+1][2*i+1]);
            printVector(h_state.R_c[2*(j-2)+1][2*i]);
            h2_state.restriction_operator[(j-2)][i] = vector_add(vector_add(vector_subtract(h_state.R_c[2*(j-2)][2*i], h_state.R_d[2*(j-2)][2*i]), 
                                                                        vector_subtract(h_state.R_c[2*(j-2)][2*i+1], h_state.R_d[2*(j-2)][2*i+1])),
                                                            vector_add(vector_subtract(h_state.R_c[2*(j-2)+1][2*i+1], h_state.R_d[2*(j-2)+1][2*i+1]), 
                                                                        vector_subtract(h_state.R_c[2*(j-2)+1][2*i], h_state.R_d[2*(j-2)+1][2*i])));
            std::cout << "h2_state residual" << std::endl;
            printVector(h2_state.restriction_operator[(j-2)][i]);
            std::cout << " " << std::endl;
        }
    }

    


    return h2_state;
}


