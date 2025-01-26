#include "Multigrid.h"
#include "read_PLOT3D.h"
#include "vector_helper.h"
#include <iostream>
#include <vector>



// Constructor definition
Multigrid::Multigrid(SpatialDiscretization& h_state)
    : TemporalDiscretization(h_state.x, h_state.y, h_state.rho, h_state.u, h_state.v,
                              h_state.E, h_state.T, h_state.p, h_state.T_ref,
                              h_state.U_ref),
      h_state(h_state) {}


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
    // write_PLOT3D_mesh(h_state.x, h_state.y, "mesh_h.xy");
    // write_PLOT3D_mesh(x_2h, y_2h, "mesh_2h.xy");

    SpatialDiscretization h2_state(x_2h, y_2h, h_state.rho, h_state.u, h_state.v, h_state.E,
                                    h_state.T, h_state.p, h_state.k2_coeff, h_state.k4_coeff,
                                    h_state.T_ref, h_state.U_ref);

    int nx = h2_state.nx;
    int ny = h2_state.ny;

    std::cout << "nx: " << nx << " ny: " << ny << std::endl;
    
    for (int j=2; j < ny_2h-1+2; ++j) {
        for (int i=0; i < nx_2h-1; ++i) {
            // Transfer operators for the cell-centered scheme
            // std::cout << j << " " << i << std::endl;
            // std::cout << "h_state W" << std::endl;
            // printVector(h_state.W[2*j][2*i]);
            // printVector(h_state.W[2*j][2*i+1]);
            // printVector(h_state.W[2*+1][2*i+1]);
            // printVector(h_state.W[2*j+1][2*i]);
            h2_state.W[j][i] = vector_scale(1/(h_state.OMEGA[2*j][2*i] + h_state.OMEGA[2*j][2*i+1] + h_state.OMEGA[2*j+1][2*i+1] + h_state.OMEGA[2*j+1][2*i]),
                                vector_add(vector_add(vector_scale(h_state.OMEGA[2*j][2*i], h_state.W[2*j][2*i]), vector_scale(h_state.OMEGA[2*j][2*i+1], h_state.W[2*j][2*i+1])), 
                                vector_add(vector_scale(h_state.OMEGA[2*j+1][2*i+1], h_state.W[2*j+1][2*i+1]), vector_scale(h_state.OMEGA[2*j+1][2*i], h_state.W[2*j+1][2*i]))));
            // std::cout << "h2_state W" << std::endl;
            // printVector(h2_state.W[j][i]);
            // std::cout << " " << std::endl;
            
            // Restriction operator
            // std::cout << "h_state residual" << std::endl;
            // printVector(vector_subtract(h_state.R_c[2*(j-2)][2*i], h_state.R_d[2*(j-2)][2*i]));
            // printVector(vector_subtract(h_state.R_c[2*(j-2)][2*i+1], h_state.R_d[2*(j-2)][2*i+1]));
            // printVector(vector_subtract(h_state.R_c[2*(j-2)+1][2*i+1], h_state.R_d[2*(j-2)+1][2*i+1]));
            // printVector(vector_subtract(h_state.R_c[2*(j-2)+1][2*i], h_state.R_d[2*(j-2)+1][2*i]));
            h2_state.restriction_operator[(j-2)][i] = vector_add(vector_add(vector_subtract(h_state.R_c[2*(j-2)][2*i], h_state.R_d[2*(j-2)][2*i]), 
                                                                        vector_subtract(h_state.R_c[2*(j-2)][2*i+1], h_state.R_d[2*(j-2)][2*i+1])),
                                                            vector_add(vector_subtract(h_state.R_c[2*(j-2)+1][2*i+1], h_state.R_d[2*(j-2)+1][2*i+1]), 
                                                                        vector_subtract(h_state.R_c[2*(j-2)+1][2*i], h_state.R_d[2*(j-2)+1][2*i])));
            // std::cout << "h2_state residual" << std::endl;
            // printVector(h2_state.restriction_operator[(j-2)][i]);
            // std::cout << " " << std::endl;
        }
    }

    


    return h2_state;
}

// Prolongation implementation (coarse to fine grid)
SpatialDiscretization Multigrid::prolongation(SpatialDiscretization& h2_state, std::vector<std::vector<double>>& x_h, std::vector<std::vector<double>>& y_h) {
    int ny_cell_2h = h2_state.ny - 1;
    int nx_cell_2h = h2_state.nx - 1;

    SpatialDiscretization h_state(x_h, y_h, h2_state.rho, h2_state.u, h2_state.v, h2_state.E,
                                    h2_state.T, h2_state.p, h2_state.k2_coeff, h2_state.k4_coeff,
                                    h2_state.T_ref, h2_state.U_ref);

    int nx = h_state.nx;
    int ny = h_state.ny;

    std::cout << "nx: " << nx << " ny: " << ny << std::endl;
    for (int j=2; j < ny; j += 2) {
        // upper right corner
        // std::cout << "upper right corner" << std::endl;
        for (int i=0; i < nx - 1; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = i/2;
            int j_I = j/2+1;
            int i_Im1 = (i/2 - 1 + nx_cell_2h) % nx_cell_2h;
            int j_Jm1 = j/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Im1: " << i_Im1 << " j_Jm1: " << j_Jm1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jm1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Im1]), h2_state.deltaW_2h[j_Jm1][i_Im1])));
        }
        // upper left corner
        // std::cout << "upper left corner" << std::endl;
        for (int i=1; i < nx; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = (i-1)/2;
            int j_I = j/2+1;
            int i_Ip1 = ((i+1)/2) % nx_cell_2h;
            int j_Jm1 = j/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Ip1: " << i_Ip1 << " j_Jm1: " << j_Jm1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jm1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Ip1]), h2_state.deltaW_2h[j_Jm1][i_Ip1])));
        }
    }
    for (int j=3; j < ny + 1; j += 2) {
        // lower right corner
        // std::cout << "lower right corner" << std::endl;
        for (int i=0; i < nx - 1; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = i/2;
            int j_I = (j+1)/2;
            int i_Im1 = (i/2 - 1 + nx_cell_2h) % nx_cell_2h;
            int j_Jp1 = (j+3)/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Im1: " << i_Im1 << " j_Jp1: " << j_Jp1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jp1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Im1]), h2_state.deltaW_2h[j_Jp1][i_Im1])));
        }
        // lower left corner
        // std::cout << "lower left corner" << std::endl;
        for (int i=1; i < nx; i += 2) {
            // std::cout << "j: " << j << " i: " << i << std::endl;
            int i_I = (i-1)/2;
            int j_I = (j+1)/2;
            int i_Ip1 = ((i+1)/2) % nx_cell_2h;
            int j_Jp1 = (j+3)/2;
            // std::cout << "i_I: " << i_I << " j_I: " << j_I << " i_Ip1: " << i_Ip1 << " j_Jp1: " << j_Jp1 << std::endl;
            h_state.prolongation_operator[j-2][i] = vector_scale(0.0625, vector_add(vector_add(vector_scale(9, h2_state.deltaW_2h[j_I][i_I]), vector_scale(3, h2_state.deltaW_2h[j_Jp1][i_I])),
                                                    vector_add(vector_scale(3, h2_state.deltaW_2h[j_I][i_Ip1]), h2_state.deltaW_2h[j_Jp1][i_Ip1])));
        }
    }
    
    return h_state;
}


