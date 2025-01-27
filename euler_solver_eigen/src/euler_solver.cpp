//
// Created by corde on 2025-01-24.
//


#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <chrono>

#include <omp.h>

using namespace std;
using namespace Eigen;

// using Eigen::MatrixxD;
// using Eigen::MatrixXd;



class Structured_FVM {
public:

    // Air properties
    const double gamma = 1.4;
    const double R = 286;

    // Flow properties
    const double mach = 0.5;
    const double alpha = 0.02181662;

    const double P_inf = 100000;
    const double T_inf = 300;

    const double rho_inf = P_inf/(R*T_inf);
    const double c_inf = sqrt(gamma*P_inf/rho_inf);

    const double u_inf = mach * c_inf * cos(alpha);
    const double v_inf = mach * c_inf * sin(alpha);
    const double E_inf = compute_E(P_inf, rho_inf, u_inf, v_inf);

    const double dyn_pressure = 0.5*rho_inf*(u_inf*u_inf + v_inf*v_inf);


    // JST scheme properties
    const double kappa2 = 1.0/2.0;
    const double kappa4 = 1.0/32;

    int ni_nodes = 0;
    int nj_nodes = 0;

    int ni_cells = 0;
    int nj_cells = 0;

    const double gamma_m1 = gamma - 1.0;



    chrono::duration<double> duration_flux;
    chrono::duration<double> duration_dependent;
    chrono::duration<double> duration_upsilon;
    chrono::duration<double> duration_lambda;
    chrono::duration<double> duration_dissipation;
    chrono::duration<double> duration_update_dt;
    chrono::duration<double> duration_residual;

    vector<int> face_i = {0, 1, 0, 0};  // only index 1 and 2 are of interest
    vector<int> face_j = {0, 0, 1, 0};

    // string filename = "";

    Tensor<double, 3> mesh_nodes;

    Tensor<double, 4> faces;        // i, j, k, (nx, ny, dS)
    Tensor<double, 4> avg_faces;        // i, j, k, (nx, ny, dS)
    Tensor<double, 2> volumes;      // i, j
    Tensor<double, 3> cv;           // Conservative variables
    Tensor<double, 3> dv;           // Dependent variables

    Tensor<double, 3> upsilon;
    Tensor<double, 3> epsilon2;
    Tensor<double, 3> epsilon4;

    Tensor<double, 3> lambda;

    Tensor<double, 2> dt;

    void load_mesh(const string filename) {

        ifstream mesh_file(filename);

        string str;
        getline(mesh_file, str);

        // number of blocks
        int block_number = stoi(str);


        getline(mesh_file, str, ' ');
        ni_nodes = stoi(str);

        getline(mesh_file, str);
        nj_nodes = stoi(str);

        cout << "Filename: " << filename << "\n";
        cout << "Number of blocks: " << block_number << "\n";
        cout << "Number of i nodes: " << ni_nodes << "\n";
        cout << "Number of j nodes: " << nj_nodes << endl;

        mesh_nodes = Tensor<double, 3>(ni_nodes, nj_nodes, 2);

        // vector<vector<vector<double>>> nodes(i_nodes, vector<vector<double>>(j_nodes, vector<double>(2, 0)));

        for (int j = 0; j < nj_nodes; j++) {
            for (int i = 0; i < ni_nodes; i++) {
                // cout << i << " " << j << endl;
                getline(mesh_file, str);
                mesh_nodes(i, j, 0) = stof(str);

                // cout << "node id: [" << i << ", " << j << "]" << endl;
            }
        }

        for (int j = 0; j < nj_nodes; j++) {
            for (int i = 0; i < ni_nodes; i++) {
                getline(mesh_file, str);
                mesh_nodes(i, j, 1) = stof(str);
            }
        }
    }

    void compute_geometry() {

        // Adds 2 layers of dummy cells at each boundary
        ni_cells = ni_nodes + 3;
        nj_cells = nj_nodes + 3;

        faces = Tensor<double, 4>(ni_cells, nj_cells, 4, 3);
        avg_faces = Tensor<double, 4>(ni_cells, nj_cells, 2, 3);
        volumes = Tensor<double, 2>(ni_cells, nj_cells);
        cv = Tensor<double, 3>(ni_cells, nj_cells, 4);
        dv = Tensor<double, 3>(ni_cells, nj_cells, 5);

        upsilon = Tensor<double, 3>(ni_cells, nj_cells, 2);
        epsilon2 = Tensor<double, 3>(ni_cells, nj_cells, 2);
        epsilon4 = Tensor<double, 3>(ni_cells, nj_cells, 2);
        lambda = Tensor<double, 3>(ni_cells, nj_cells, 3);

        dt = Tensor<double, 2>(ni_cells, nj_cells);
        // dt_broadcasted = Tensor<double, 3>(ni_cells, nj_cells, 4);
        // volumes_broadcasted = Tensor<double, 3>(ni_cells, nj_cells, 4);

        vector<vector<int>> indexes = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}};

        for (int j = 0; j < nj_nodes-1; j++) {
            for (int i = 0; i < ni_nodes-1; i++) {
                vector<double> x_coord(4, 0.0);
                vector<double> y_coord(4, 0.0);

                for (int k = 0; k < 4; k++) {

                    // int i_next = indexes[k+1][0] + i;
                    // int j_next = indexes[k+1][1] + j;

                    vector<int> current_index = {i, j};

                    int p0_i = indexes[k][0] + i;
                    int p0_j = indexes[k][1] + j;
                    int p1_i = indexes[k+1][0] + i;
                    int p1_j = indexes[k+1][1] + j;

                    double dx = mesh_nodes(p1_i, p1_j, 0) - mesh_nodes(p0_i, p0_j, 0);
                    double dy = mesh_nodes(p1_i, p1_j, 1) - mesh_nodes(p0_i, p0_j, 1);
                    double dS = sqrt(dx*dx + dy*dy);

                    if(dS == 0) {
                        cout << "Null normal at index: " << i << ", " << j << endl;
                    }

                    dx = dx / dS;
                    dy = dy / dS;

                    faces(i+2, j+2, k, 0) = dy;
                    faces(i+2, j+2, k, 1) = -dx;
                    faces(i+2, j+2, k, 2) = dS;

                    x_coord[k] = mesh_nodes(p0_i, p0_j, 0);
                    y_coord[k] = mesh_nodes(p0_i, p0_j, 1);
                }

                // Compute cell volumes
                volumes(i+2, j+2) = 0.5*((x_coord[0] - x_coord[2])*(y_coord[1] - y_coord[3]) + (x_coord[3] - x_coord[1])*(y_coord[0] - y_coord[2]));
            }
        }

        // Eigen::array<int, 3> new_shape = {ni_cells, nj_cells, 1}
        // auto reshaped = volumes_broadcasted.reshape(new_shape);



        // volumes_broadcasted =

        // Give dummy cells dimensions
        // TODO: maybe give corner cells volume to avoid NaN error
        for (int i = 2; i < ni_cells-2; i++) {
            faces.chip(i, 0).chip(0, 0) = faces.chip(i, 0).chip(2, 0);
            faces.chip(i, 0).chip(1, 0) = faces.chip(i, 0).chip(2, 0);
            volumes(i, 0) = volumes(i, 2);
            volumes(i, 1) = volumes(i, 2);

            faces.chip(i, 0).chip(nj_cells-1, 0) = faces.chip(i, 0).chip(nj_cells-3, 0);
            faces.chip(i, 0).chip(nj_cells-2, 0) = faces.chip(i, 0).chip(nj_cells-3, 0);
            volumes(i, nj_cells-1) = volumes(i, nj_cells-3);
            volumes(i, nj_cells-2) = volumes(i, nj_cells-3);
        }

        for (int j = 2; j < nj_cells-2; j++) {
            faces.chip(0, 0).chip(j, 0) = faces.chip(ni_cells-4, 0).chip(j, 0);
            faces.chip(1, 0).chip(j, 0) = faces.chip(ni_cells-3, 0).chip(j, 0);
            volumes(0, j) = volumes(ni_cells-4, j);
            volumes(1, j) = volumes(ni_cells-3, j);

            faces.chip(ni_cells-1, 0).chip(j, 0) = faces.chip(3, 0).chip(j, 0);
            faces.chip(ni_cells-2, 0).chip(j, 0) = faces.chip(2, 0).chip(j, 0);
            volumes(ni_cells-1, j) = volumes(3, j);
            volumes(ni_cells-2, j) = volumes(2, j);
        }

        // Computes averages
        for (int i = 0; i < ni_cells; i++) {
            for (int j = 0; j < nj_cells; j++) {
                for (int k = 0; k < 2; k++) {
                    avg_faces(i, j, k, 0) = 0.5*(faces(i, j, k, 0) - faces(i, j, k+2, 0));
                    avg_faces(i, j, k, 1) = 0.5*(faces(i, j, k, 1) - faces(i, j, k+2, 1));
                    avg_faces(i, j, k, 2) = 0.5*(faces(i, j, k, 2) + faces(i, j, k+2, 2));
                }
            }
        }

        // cout << "volumes" << endl;
        // cout << volumes << endl;
        // cout << "" << endl;
    }

    void initialize_cells() {

        Tensor<double, 1> init_values(4);
        init_values.setValues({rho_inf, rho_inf*u_inf, rho_inf*v_inf, rho_inf*E_inf});

        for (int i = 0; i < ni_cells; i++) {
            for (int j = 0; j < nj_cells; j++) {
                cv.chip(i, 0).chip(j, 0) = init_values;
            }
        }
    }

    void update_all_dependents() {

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2)
        for (int i = 2; i < ni_cells-2; i++) {
            for (int j = 2; j < nj_cells-2; j++) {
                update_dependent(i, j);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        duration_dependent += (end_time - start_time);
    }

    void update_dependent(const int i, const int j) {

        double rho = cv(i, j, 0);
        double u = cv(i, j, 1)/rho;
        double v = cv(i, j, 2)/rho;
        double E = cv(i, j, 3)/rho;

        double pressure = compute_pressure(rho, E, u, v);
        double c = compute_c(pressure, rho);

        dv(i, j, 0) = u;
        dv(i, j, 1) = v;
        dv(i, j, 2) = E;
        dv(i, j, 3) = pressure;
        dv(i, j, 4) = c;
    }


    Tensor<double, 3> compute_flux() {

        auto start_time = chrono::high_resolution_clock::now();

        Tensor<double, 3> flux(ni_cells, nj_cells, 4);
        // flux.setZero();

        Tensor<double, 4> flux_faces(ni_cells, nj_cells, 2, 4);

        //
        #pragma omp parallel for collapse(2) schedule(static, 64)
        for (int i = 2; i < ni_cells-2; i++) {
            for (int j = 2; j < nj_cells-2; j++) {
                for (int k = 1; k < 3; k++) {
                    // Neighboring cell index
                    int ii = face_i[k] + i;
                    int jj = face_j[k] + j;

                    double nx = faces(i, j, k, 0);
                    double ny = faces(i, j, k, 1);
                    double dS = faces(i, j, k, 2);

                    // Average of conservative var.
                    // Tensor<double, 1> cv_face = 0.5*(cv.chip(i, 0).chip(j, 0) + cv.chip(ii, 0).chip(jj, 0));

                    double rho = 0.5*(cv(i, j, 0) + cv(ii, jj, 0));
                    double rhou = 0.5*(cv(i, j, 1) + cv(ii, jj, 1));
                    double rhov = 0.5*(cv(i, j, 2) + cv(ii, jj, 2));
                    double rhoE = 0.5*(cv(i, j, 3) + cv(ii, jj, 3));
                    // cout << cv_face << endl;

                    double u = rhou/rho;
                    double v = rhov/rho;
                    double E = rhoE/rho;

                    double pressure = compute_pressure(rho, E, u, v);
                    double H = E + pressure/rho;
                    double V = compute_V(u, v, nx, ny);

                    // Tensor<double, 1> flux_face(4);
                    // flux_face.setValues({(rho*V)*dS, (rhou*V + nx*pressure)*dS, (rhov*V + ny*pressure)*dS, (rho*H*V)*dS});

                    // double flux0 = rho*V*dS;
                    // double flux1 = (rhou*V + nx*pressure)*dS;
                    // double flux2 = (rhov*V + ny*pressure)*dS;
                    // double flux3 = rho*H*V*dS;

                    // TODO: directly compute flux values instead of creating temporary variables
                    flux_faces(i, j, k-1, 0) = rho*V*dS;
                    flux_faces(i, j, k-1, 1) = (rhou*V + nx*pressure)*dS;
                    flux_faces(i, j, k-1, 2) = (rhov*V + ny*pressure)*dS;
                    flux_faces(i, j, k-1, 3) = rho*H*V*dS;

                    // flux(i, j, 0) += flux0;
                    // flux(i, j, 1) += flux1;
                    // flux(i, j, 2) += flux2;
                    // flux(i, j, 3) += flux3;
                    //
                    // flux(ii, jj, 0) -= flux0;
                    // flux(ii, jj, 1) -= flux1;
                    // flux(ii, jj, 2) -= flux2;
                    // flux(ii, jj, 3) -= flux3;
                }
            }
        }

        // cout << "flux:" << endl;
        // cout << flux.chip(0, 2) << endl;

        // Solid wall (airfoil)
        #pragma omp parallel for
        for (int i = 2; i < ni_cells-2; i++) {

            double nx = faces(i, 2, 0, 0);
            double ny = faces(i, 2, 0, 1);
            double dS = faces(i, 2, 0, 2);

            // Pressure interpolation
            double wall_pressure = 0.5*(3*dv(i, 2, 3) - dv(i, 3, 3));

            flux_faces(i, 1, 1, 0) = 0.0;
            flux_faces(i, 1, 1, 1) = -nx*wall_pressure*dS;
            flux_faces(i, 1, 1, 2) = -ny*wall_pressure*dS;
            flux_faces(i, 1, 1, 3) = 0.0;
        }

        // Coordinate cut
        #pragma omp parallel for
        for (int j = 2; j < nj_cells-2; j++) {

            int i = 1;
            int k = 1;

            // Neighboring cell index
            int ii = 2; // face_i[k] + i;
            int jj = j; //face_j[k] + j;W

            double nx = faces(i, j, k, 0);
            double ny = faces(i, j, k, 1);
            double dS = faces(i, j, k, 2);

            double rho = 0.5*(cv(i, j, 0) + cv(ii, jj, 0));
            double rhou = 0.5*(cv(i, j, 1) + cv(ii, jj, 1));
            double rhov = 0.5*(cv(i, j, 2) + cv(ii, jj, 2));
            double rhoE = 0.5*(cv(i, j, 3) + cv(ii, jj, 3));

            double u = rhou/rho;
            double v = rhov/rho;
            double E = rhoE/rho;

            double pressure = compute_pressure(rho, E, u, v);
            double H = E + pressure/rho;
            double V = compute_V(u, v, nx, ny);

            // Tensor<double, 1> flux_face(4);
            // flux_face.setValues({(rho*V)*dS, (rhou*V + nx*pressure)*dS, (rhov*V + ny*pressure)*dS, (rho*H*V)*dS});

            double flux0 = rho*V*dS;
            double flux1 = (rhou*V + nx*pressure)*dS;
            double flux2 = (rhov*V + ny*pressure)*dS;
            double flux3 = rho*H*V*dS;

            flux_faces(i, j, 0, 0) = flux0;
            flux_faces(i, j, 0, 1) = flux1;
            flux_faces(i, j, 0, 2) = flux2;
            flux_faces(i, j, 0, 3) = flux3;

            // flux(ii, jj, 0) -= flux0;
            // flux(ii, jj, 1) -= flux1;
            // flux(ii, jj, 2) -= flux2;
            // flux(ii, jj, 3) -= flux3;
        }

        #pragma omp parallel for collapse(3) schedule(static, 64)
        for (int i = 2; i < ni_cells-2; i++) {
            for (int j = 2; j < nj_cells-2; j++) {
                for (int l = 0; l < 4; l++) {
                    double flux_i = flux_faces(i, j, 0, l) - flux_faces(i-1, j, 0, l);
                    double flux_j = flux_faces(i, j, 1, l) - flux_faces(i, j-1, 1, l);
                    flux(i, j, l) = flux_i + flux_j;
                }
            }
        }

        //cout << flux.chip(0, 2) << endl;
        auto end_time = chrono::high_resolution_clock::now();
        duration_flux += (end_time - start_time);

        return flux;
    }

    void update_all_upsilon() {

        auto start_time = chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 1; i < nj_cells-1; i++) {
            for(int j = 1; j < nj_cells-1; j++) {
                for(int k = 1; k < 3; k++) {

                    // TODO: compute these indexes in advance?
                    int p1i = face_i[k] + i;
                    int p1j = face_j[k] + j;

                    int m1i = -face_i[k] + i;
                    int m1j = -face_j[k] + j;

                    // Was faster when directly accessing the tensor?
                    double p1_pressure = dv(p1i, p1j, 3);
                    double pressure = 2.0*dv(i, j, 3);
                    double m1_pressure = dv(m1i, m1j, 3);

                    double dd_pressure = p1_pressure - pressure + m1_pressure;

                    upsilon(i, j, k-1) = abs(dd_pressure)/(p1_pressure + pressure + m1_pressure);
                }
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        duration_upsilon += (end_time - start_time);
    }

    void update_all_lambda() {

        auto start_time = chrono::high_resolution_clock::now();

        // TODO: lambda can be computed with the dependent variables
        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 1; i < nj_cells-1; i++) {
            for(int j = 1; j < nj_cells-1; j++) {

                double u = dv(i, j, 0);
                double v = dv(i, j, 1);
                double c = dv(i, j, 4);

                // indexes for avg_faces --> 0: j-direction, 1: i-direction
                double Vi = compute_V(u, v, avg_faces(i, j, 1, 0), avg_faces(i, j, 1, 1));
                double lambda_i = (abs(Vi) + c)*avg_faces(i, j, 1, 2);

                double Vj = compute_V(u, v, avg_faces(i, j, 0, 0), avg_faces(i, j, 0, 1));
                double lambda_j = (abs(Vj) + c)*avg_faces(i, j, 0, 2);

                // index corrected to be compatible with dissipation computation
                // lambda(i, j, 0) = lambda_i;
                // lambda(i, j, 1) = lambda_j;
                lambda(i, j, 2) = lambda_i + lambda_j;
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        duration_lambda += (end_time - start_time);
    }

    Tensor<double, 3> compute_dissipation() {

        auto start_time = chrono::high_resolution_clock::now();

        Tensor<double, 3> dissip(ni_cells, nj_cells, 4);
        Tensor<double, 4> dissip_faces(ni_cells, nj_cells, 2, 4);
        // dissip.setZero();

        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 1; i < nj_cells-2; i++) {
            for(int j = 1; j < nj_cells-2; j++) {
                for(int k = 1; k < 3; k++) {

                    // TODO: precompute index values
                    int p1i = face_i[k] + i;
                    int p1j = face_j[k] + j;
                    int p2i = 2*face_i[k] + i;  // TODO: vector with already 2* and -1* values
                    int p2j = 2*face_j[k] + j;
                    int m1i = -face_i[k] + i;
                    int m1j = -face_j[k] + j;

                    double lambda_s = 0.5*(lambda(i, j, 2) + lambda(p1i, p1j, 2));

                    // TODO: if dissipation cannot be parallelized, epsilon computation can be
                    double eps2 = kappa2*max(upsilon(i, j, k-1), upsilon(p1i, p1j, k-1));
                    double eps4 = max(0.0, kappa4 - eps2);

                    for(int l = 0; l < 4; l++) {
                        double order2 = cv(p1i, p1j, l) - cv(i, j, l);
                        double order4 = cv(p2i, p2j, l) - 3.0*cv(p1i, p1j, l) + 3.0*cv(i, j, l) - cv(m1i, m1j, l);
                        // double local_dissip = lambda_s*(eps2*order2 - eps4*order4);

                        dissip_faces(i, j, k-1, l) = lambda_s*(eps2*order2 - eps4*order4);
                        // dissip(i, j, l) += local_dissip;
                        // dissip(p1i, p1j, l) -= local_dissip;
                    }
                }
            }
        }

        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 2; i < nj_cells-2; i++) {
            for(int j = 2; j < nj_cells-2; j++) {
                for(int l = 0; l < 4; l++) {
                    double dissip_i = dissip_faces(i, j, 0, l) - dissip_faces(i-1, j, 0, l);
                    double dissip_j = dissip_faces(i, j, 1, l) - dissip_faces(i, j-1, 1, l);
                    dissip(i, j, l) = dissip_i + dissip_j;
                }
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        duration_dissipation += (end_time - start_time);

        return dissip;
    }

    void update_boundary() {

        // solid wall (airfoil)
        for(int i = 2; i < ni_cells-2; i++) {
            cv.chip(i, 0).chip(1, 0) = 2.0*cv.chip(i, 0).chip(2, 0) - cv.chip(i, 0).chip(3, 0);
            cv.chip(i, 0).chip(0, 0) = 3.0*cv.chip(i, 0).chip(2, 0) - 2.0*cv.chip(i, 0).chip(3, 0);

            update_dependent(i, 1);
            update_dependent(i, 0);

            // Pressure is required to compute flux at solidwall
            update_dependent(i, 2);
            update_dependent(i, 3);
        }

        // far-field
        for(int i = 2; i < ni_cells-2; i++) {

            double nx = faces(i, nj_cells-3, 2, 0);
            double ny = faces(i, nj_cells-3, 2, 1);

            // nj_cells-3 cells dv are not updated when only computing flux
            update_dependent(i, nj_cells-3);

            // TODO: all the values below should already be computed when calling update_dependent()
            double rho_d = cv(i, nj_cells-3, 0);
            double u_d = cv(i, nj_cells-3, 1)/rho_d;
            double v_d = cv(i, nj_cells-3, 2)/rho_d;
            double E_d = cv(i, nj_cells-3, 3)/rho_d;
            double V_d = compute_V(u_d, v_d, nx, ny);
            // double pressure_cv = compute_pressure(rho_d, E_d, u_d, v_d);

            double pressure_d = compute_pressure(rho_d, E_d, u_d, v_d);
            double c_d = compute_c(pressure_d, rho_d);

            // double pressure_d = dv[i][farfield_j][0]; // pressure within domain

            double pressure_b = 0;
            double rho_b = 0;
            double u_b = 0;
            double v_b = 0;
            double E_b = 0;

            // cout << "u: " << u_d << " - nx: " << nx << endl;

            // outflow
            if(V_d > 0) {

                //cout << "outflow" << endl;
                // TODO: add supersonic outflow

                // subsonic outflow
                pressure_b = P_inf;
                rho_b = rho_d + (pressure_b - pressure_d)/(c_d*c_d);
                u_b = u_d + nx*(pressure_d - pressure_b)/(rho_d*c_d);
                v_b = v_d + ny*(pressure_d - pressure_b)/(rho_d*c_d);
            }

            // inflow
            else {
                // TODO: add supersonic inflow

                // cout << "inflow" << endl;

                // subsonic inflow
                pressure_b = 0.5*(P_inf + pressure_d - rho_d*c_d*(nx*(u_inf - u_d) + ny*(v_inf - v_d)));
                rho_b = rho_inf + (pressure_b - P_inf)/(c_d*c_d);
                u_b = u_inf - nx*(P_inf - pressure_b)/(rho_d*c_d);
                v_b = v_inf - ny*(P_inf - pressure_b)/(rho_d*c_d);
            }

            E_b = compute_E(pressure_b, rho_b, u_b, v_b);

            cv(i, nj_cells-1, 0) = rho_b;
            cv(i, nj_cells-1, 1) = rho_b*u_b;
            cv(i, nj_cells-1, 2) = rho_b*v_b;
            cv(i, nj_cells-1, 3) = rho_b*E_b;
            update_dependent(i, nj_cells-1);


            cv.chip(i, 0).chip(nj_cells-2, 0) = 2.0*cv.chip(i, 0).chip(nj_cells-1, 0) - cv.chip(i, 0).chip(nj_cells-3, 0);
            update_dependent(i, nj_cells-2);
        }

        // coordinate cut
        for (int j = 2; j < nj_cells-2; j++) {

            cv.chip(0, 0).chip(j, 0) = cv.chip(ni_cells-4, 0).chip(j, 0);
            cv.chip(1, 0).chip(j, 0) = cv.chip(ni_cells-3, 0).chip(j, 0);

            cv.chip(ni_cells-1, 0).chip(j, 0) = cv.chip(3, 0).chip(j, 0);
            cv.chip(ni_cells-2, 0).chip(j, 0) = cv.chip(2, 0).chip(j, 0);

            update_dependent(1, j);
            update_dependent(0, j);
            update_dependent(ni_cells-2, j);
            update_dependent(ni_cells-1, j);
        }

    }

    void compute_dt() {

        double sigma = 3.6;

        // TODO: remove 'volumes' term here and in apply_dt()
        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 2; i < ni_cells-2; i++) {
            for(int j = 2; j < nj_cells-2; j++) {

                dt(i, j) = sigma*volumes(i, j)/lambda(i, j, 2);
            }
        }
    }


    double compute_pressure(const double rho, const double E, const double u, const double v) {
        return (gamma - 1.0)*rho*(E - (u*u + v*v)/2); // TODO: use gamma_m1
    }

    double compute_c(const double pressure, const double rho) {
        return sqrt(gamma*pressure/rho);
    }

    double compute_E(const double pressure, const double rho, const double u, const double v) {
        return pressure/((gamma - 1.0)*rho) + 0.5*(u*u + v*v);    // TODO: use gamma_m1
    }

    double compute_V(const double u, const double v, const double nx, const double ny) {
        return u*nx + v*ny;
    }

    // TODO: move this code to a new class

    void apply_dt(const Tensor<double, 3>& cv_0, const double alpha, const Tensor<double, 3>& Rc, const Tensor<double, 3>& Rd) {

        auto start_time = chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 2; i < ni_cells-2; i++) {
            for(int j = 2; j < nj_cells-2; j++) {

                // TODO: precompute dt(i, j)/volumes(i, j) since this operation is computed: ni_cells * nj_cells * 4 times in a full runge-kutta step
                const double step = alpha*dt(i, j)/volumes(i, j);

                for(int k = 0; k < 4; k++) {
                    cv(i, j, k) = cv_0(i, j, k) - step*(Rc(i, j, k) - Rd(i, j, k));
                    // cv(i, j, k) = cv_0(i, j, k) - alpha*dt(i, j)/volumes(i, j)*(Rc(i, j, k) - Rd(i, j, k));
                }
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        duration_update_dt += (end_time - start_time);
    }

    Tensor<double, 3> compute_Rd_mid(const Tensor<double, 3>& Rd0, const Tensor<double, 3>& Rd1, const double beta0, const double beta1) {

        Tensor<double, 3> Rd_mid(ni_cells, nj_cells, 4);

        const double one_minus_beta1 = 1.0 - beta1;

        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 2; i < ni_cells-2; i++) {
            for(int j = 2; j < nj_cells-2; j++) {
                for(int k = 0; k < 4; k++) {
                    Rd_mid(i, j, k) = beta0*Rd0(i, j, k) + one_minus_beta1*Rd1(i, j, k);
                }
            }
        }

        return Rd_mid;
    }

    double compute_l2_residual(const Tensor<double, 3>& cv_0, const Tensor<double, 3>& cv_1, const double l2_ref=1.0) {

        auto start_time = chrono::high_resolution_clock::now();

        double residual = 0.0;

        #pragma omp parallel for collapse(2) schedule(dynamic, 128)
        for(int i = 2; i < ni_cells-2; i++ ) {
            for(int j = 2; j < nj_cells-2; j++) {
                residual = residual + pow(cv_1(i, j, 0) - cv_0(i, j, 0), 2);
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        duration_residual += (end_time - start_time);

        return residual;
    }

    double compute_coeff() {

        double Fx = 0.0;
        double Fy = 0.0;

        for (int i = 2; i < ni_cells-2; i++) {

            update_dependent(i, 2);
            update_dependent(i, 3);
            double wall_pressure = 0.5*(3*dv(i, 2, 3) - dv(i, 3, 3));

            Fx += wall_pressure*faces(i, 2, 0, 0)*faces(i, 2, 0, 2);
            Fy += wall_pressure*faces(i, 2, 0, 1)*faces(i, 2, 0, 2);

        }

        double lift = Fy*cos(alpha) - Fx*sin(alpha);

        return lift/dyn_pressure;
    }

};


int main(int argc, char* argv[]) {

    auto start_time = std::chrono::system_clock::now();


    cout << "Available cores: " << omp_get_num_procs() << "\n";

    const int num_thread = 8;
    // omp_set_dynamic(0);
    omp_set_num_threads(num_thread);

    cout << "Set threads: " << num_thread << "\n";

    Structured_FVM fvm;

    // string filename = "naca0012_9.xyz";
    // string filename = "naca0012_32.xyz";
    // string filename = "naca0012_64.xyz";
    // string filename = "naca0012_256.xyz";
    string filename = "naca0012_512.xyz";

    fvm.load_mesh(filename);
    fvm.compute_geometry();
    fvm.initialize_cells();

    // problem dimensions
    int ni_cells = fvm.ni_cells;
    int nj_cells = fvm.nj_cells;
    int n_cv = 4;

    // Declare the tensors outside the loop to avoid repeated allocations.
    // Fixed-size tensors for the given problem dimensions
    Eigen::Tensor<double, 3> cv_0(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rc_0(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rc_1(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rc_2(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rc_3(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rc_4(ni_cells, nj_cells, n_cv);

    Eigen::Tensor<double, 3> Rd_0(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rd_2(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rd_20(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rd_4(ni_cells, nj_cells, n_cv);
    Eigen::Tensor<double, 3> Rd_42(ni_cells, nj_cells, n_cv);

    // Hybrid Runge-Kutta coefficients
    vector<double> alphas = {0.25, 0.1667, 0.3750, 0.50, 1.0};
    vector<double> betas = {1.0, 0.0, 0.56, 0.0, 0.44};

    double res0 = 1.0;

    int max_iter = 500;

    for(int iter = 0; iter < max_iter; iter++) {

        cv_0 = fvm.cv;

        cout << "Iter: " << iter << " / " << max_iter;

        fvm.update_all_dependents();
        fvm.update_boundary();
        fvm.update_all_upsilon();
        fvm.update_all_lambda();

        fvm.compute_dt();

        Rc_0 = fvm.compute_flux();
        Rd_0 = fvm.compute_dissipation();

        // stage 1
        fvm.apply_dt(cv_0, alphas[0], Rc_0, Rd_0);
        fvm.update_boundary();

        Rc_1 = fvm.compute_flux();

        // stage 2
        fvm.apply_dt(cv_0, alphas[1], Rc_1, Rd_0);

        fvm.update_all_dependents();
        fvm.update_boundary();
        fvm.update_all_upsilon();
        fvm.update_all_lambda();

        Rc_2 = fvm.compute_flux();
        Rd_2 = fvm.compute_dissipation();

        Rd_20 = fvm.compute_Rd_mid(Rd_2, Rd_0, betas[2], betas[2]);

        // stage 3
        fvm.apply_dt(cv_0, alphas[2], Rc_2, Rd_20);
        fvm.update_boundary();

        Rc_3 = fvm.compute_flux();

        // stage 4
        fvm.apply_dt(cv_0, alphas[3], Rc_3, Rd_20);

        fvm.update_all_dependents();
        fvm.update_boundary();
        fvm.update_all_upsilon();
        // fvm.update_all_epsilon();
        fvm.update_all_lambda();

        // spatial_disc.update_cv(cv_4);
        Rc_4 = fvm.compute_flux();
        Rd_4 = fvm.compute_dissipation();
        Rd_42 = fvm.compute_Rd_mid(Rd_4, Rd_20, betas[4], betas[4]);

        //stage 5
        fvm.apply_dt(cv_0, alphas[4], Rc_4, Rd_42);

        double res = fvm.compute_l2_residual(cv_0, fvm.cv, res0);

        if (iter == 0) {
            res0 = res;
        }

        cout << " - res: " << res;

        cout << " - CL = " << fvm.compute_coeff();

        cout << "\n";
    }

    auto end_time = std::chrono::system_clock::now();

    chrono::duration<double> elapsed_time = end_time - start_time;

    cout << "Solver duration: " << elapsed_time.count() << " seconds" << endl;
    cout << "End of code." << endl;

    cout << "--- PERF ---\n";
    cout << "Flux: " << fvm.duration_flux.count() << " seconds\n";
    cout << "DV: " << fvm.duration_dependent.count() << " seconds\n";
    cout << "Upsilon: " << fvm.duration_upsilon.count() << " seconds\n";
    cout << "Lambda: " << fvm.duration_lambda.count() << " seconds\n";
    cout << "Dissipation: " << fvm.duration_dissipation.count() << " seconds\n";
    cout << "Update dt: " << fvm.duration_update_dt.count() << " seconds\n";
    cout << "Residual: " << fvm.duration_residual.count() << " seconds\n";


    auto profiled_duration = fvm.duration_flux.count() + fvm.duration_dependent.count() + fvm.duration_upsilon.count() + fvm.duration_lambda.count() + fvm.duration_dissipation.count() + fvm.duration_update_dt.count() + fvm.duration_residual.count();
    cout << "Total profiled: " << profiled_duration << " seconds\n";

    return 0;
}
