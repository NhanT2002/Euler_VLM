#include <iostream>
#include <filesystem>
#include <cstdlib> // For std::system
#include <string>
#include <vector>
#include <string>
#include <tuple>
#include <fstream>   // For file handling
#include <iomanip>
#include <sstream>   // For string stream
#include <stdexcept> // For exception handling
#include <iostream>  // For printing errors (optional)
#include <cmath>
#include <map>
#include <algorithm> // For std::sort

namespace fs = std::filesystem;

std::tuple<double, double, double, double, double> conservative_variable_from_W(const std::vector<double>& W) {
    // Implement the conversion from W to (rho, u, v, E)
    double rho = W[0];
    double u = W[1] / rho;
    double v = W[2] / rho;
    double E = W[3] / rho;
    double qq = u*u+v*v;
    double p = (1.4-1)*rho*(E-qq/2);
    return std::make_tuple(rho, u, v, E, p);
}

class cells {
public:
    std::vector<std::vector<double>> x, y;
    int ny, nx;

    std::vector<std::vector<double>> OMEGA;
    std::vector<std::vector<std::vector<std::vector<double>>>> s;
    std::vector<std::vector<std::vector<double>>> Ds;
    std::vector<std::vector<std::vector<std::vector<double>>>> n;

    cells(const std::vector<std::vector<double>>& x,
                          const std::vector<std::vector<double>>& y);
};
cells::cells(const std::vector<std::vector<double>>& x,
                          const std::vector<std::vector<double>>& y)
    : x(x), y(y) {
    ny = static_cast<int>(y.size());
    nx = static_cast<int>(x[0].size());

    s.resize(ny - 1, std::vector(nx - 1, std::vector(2, std::vector<double>(2))));
    Ds.resize(ny - 1, std::vector(nx - 1, std::vector<double>(2)));
    n.resize(ny - 1, std::vector(nx - 1, std::vector(2, std::vector<double>(2))));

    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            const double &x1 = x[j][i];
            const double &x2 = x[j][i + 1];
            const double &x3 = x[j + 1][i + 1];
            const double &x4 = x[j + 1][i];
            const double &y1 = y[j][i];
            const double &y2 = y[j][i + 1];
            const double &y3 = y[j + 1][i + 1];
            const double &y4 = y[j + 1][i];

            // Set s and compute Ds using s values
            s[j][i][0] = {y2 - y1, x1 - x2};
            s[j][i][1] = {y1 - y4, x4 - x1};

            // Length of s vectors
            Ds[j][i][0] = std::hypot(s[j][i][0][0], s[j][i][0][1]);
            Ds[j][i][1] = std::hypot(s[j][i][1][0], s[j][i][1][1]);

            // Normal vectors
            n[j][i][0] = {s[j][i][0][0] / Ds[j][i][0], s[j][i][0][1] / Ds[j][i][0]};
            n[j][i][1] = {s[j][i][1][0] / Ds[j][i][1], s[j][i][1][1] / Ds[j][i][1]};
        }
    }
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> read_PLOT3D_mesh(const std::string& file_name) {

    std::ifstream file(file_name);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_name);
    }

    std::string line;
    // Read the first line to get grid dimensions
    if (!std::getline(file, line)) {
        throw std::runtime_error("Failed to read grid dimensions from file: " + file_name);
    }

    std::istringstream iss(line);
    int nx, ny;
    if (!(iss >> nx >> ny)) {
        throw std::runtime_error("Invalid grid dimensions format in file: " + file_name);
    }

    int total_points = nx * ny;

    // Initialize 1D arrays for x and y coordinates
    std::vector<double> x(total_points);
    std::vector<double> y(total_points);

    // Read the coordinates from the file
    for (int i = 0; i < total_points; ++i) {
        if (!(file >> x[i])) {
            throw std::runtime_error("Error reading x coordinates from file: " + file_name);
        }
    }

    for (int i = 0; i < total_points; ++i) {
        if (!(file >> y[i])) {
            throw std::runtime_error("Error reading y coordinates from file: " + file_name);
        }
    }

    // Reshape 1D x and y arrays into 2D arrays (vectors of vectors)
    std::vector<std::vector<double>> x_2d(ny, std::vector<double>(nx));
    std::vector<std::vector<double>> y_2d(ny, std::vector<double>(nx));

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            x_2d[j][i] = x[j * nx + i];
            y_2d[j][i] = y[j * nx + i];
        }
    }
    std::cout << "Grid dimensions: " << x_2d.size() << " x " << x_2d[0].size() << std::endl;

    return {x_2d, y_2d};
}

std::tuple<int, int, double, double, double, double, std::vector<std::vector<std::vector<double>>>> read_PLOT3D_solution(const std::string& solution_filename) {
    std::ifstream solution_file(solution_filename);
    if (!solution_file) {
        throw std::runtime_error("Could not open file: " + solution_filename);
    }

    int ni, nj;
    std::string line;

    // Read grid dimensions
    if (!std::getline(solution_file, line)) {
        throw std::runtime_error("Failed to read grid dimensions from file: " + solution_filename);
    }
    std::istringstream iss(line);
    if (!(iss >> ni >> nj)) {
        throw std::runtime_error("Invalid grid dimensions format in file: " + solution_filename);
    }

    // Read freestream conditions
    double mach, alpha, reyn, time;
    if (!std::getline(solution_file, line)) {
        throw std::runtime_error("Failed to read freestream conditions from file: " + solution_filename);
    }
    iss.clear();
    iss.str(line);
    if (!(iss >> mach >> alpha >> reyn >> time)) {
        throw std::runtime_error("Invalid freestream conditions format in file: " + solution_filename);
    }

    // Initialize the q array (nj, ni, 4)
    std::vector<std::vector<std::vector<double>>> q(nj, std::vector<std::vector<double>>(ni, std::vector<double>(4)));

    // Read flow variables
    for (int n = 0; n < 4; ++n) {  // Iterate over the 4 variables (density, x-momentum, y-momentum, energy)
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {  // Read in the reversed order: i first, then j
                if (!std::getline(solution_file, line)) {
                    throw std::runtime_error("Failed to read flow variable at (i, j): (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                }
                q[j][i][n] = std::stod(line);  // Convert string to double
            }
        }
    }

    // Return all parameters as a tuple
    return std::make_tuple(ni, nj, mach, alpha, reyn, time, q);
}

std::tuple<double, double, double> compute_coeff(cells cells, int nx, int ny, double mach, double alpha, std::vector<std::vector<std::vector<double>>> q) {
    double p_inf = 1E5;
    double T_inf = 300.0;
    double rho_inf = p_inf/(T_inf*287);

    double a = std::sqrt(1.4*p_inf/rho_inf);
    double Vitesse = mach*a;
    double u_inf = Vitesse*std::cos(alpha);
    double v_inf = Vitesse*std::sin(alpha);
    double E_inf = p_inf/((1.4-1)*rho_inf) + 0.5*std::pow(Vitesse, 2);

    double U_ref = std::sqrt(p_inf/rho_inf);

    double rho = 1.0;
    double u = u_inf/U_ref;
    double v = v_inf/U_ref;
    double E = E_inf/(U_ref*U_ref);
    double T = 1.0;
    double p = 1.0;
    
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;
    
    std::vector<std::vector<double>> q_airfoil(nx, std::vector<double>(5));
    for (int i = 0; i < nx; ++i) {
        auto [rho, u, v, E, p] = conservative_variable_from_W(q[0][i]);
            q_airfoil[i] = {rho, u, v, E, p};
        }


    double Fx = 0.0;
    double Fy = 0.0;
    double M = 0.0;
    for (int i = 0; i < nx - 1; ++i) {
        double p_mid = 0.5 * (q_airfoil[i][4] + q_airfoil[i+1][4]);
        Fx += p_mid*cells.n[0][i][0][0]*cells.Ds[0][i][0];
        Fy += p_mid*cells.n[0][i][0][1]*cells.Ds[0][i][0];
        
        double x_mid = 0.5*(cells.x[0][i] + cells.x[0][i+1]);
        double y_mid = 0.5*(cells.y[0][i] + cells.y[0][i+1]);
        M += p_mid*(-(x_mid-x_ref)*cells.n[0][i][0][1] + (y_mid-y_ref)*cells.n[0][i][0][0])*cells.Ds[0][i][0];
    }

    double L = Fy*std::cos(alpha) - Fx*std::sin(alpha);
    double D = Fy*std::sin(alpha) + Fx*std::cos(alpha);

    double C_L = L/(0.5*rho_inf*u_inf*u_inf*c);
    double C_D = D/(0.5*rho_inf*u_inf*u_inf*c);
    double C_M = M/(0.5*rho_inf*u_inf*u_inf*c*c);

    return std::tie<double>(C_L, C_D, C_M);
}

std::tuple<double, double, double, double, double> processFile(const fs::path& filePath, cells& cells) {
    std::ifstream inputFile(filePath);
    if (!inputFile) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return {};
    }

    std::cout << "Processing file: " << filePath.filename();
    auto [ni, nj, mach, alpha, reyn, time, q] = read_PLOT3D_solution(filePath);
    std::cout << ", Mach = " << mach << ", alpha = " << alpha / M_PI * 180.0; // Convert alpha from radians to degrees

    auto [C_L, C_D, C_M] = compute_coeff(cells, ni, nj, mach, alpha, q);
    std::cout << ", C_L = " << C_L << " C_D = " << C_D << " C_M = " << C_M << std::endl;

    inputFile.close();
    return {mach, alpha, C_L, C_D, C_M};
}

int main() {
    std::string mesh_file = "../mesh/x.6";
    auto [x, y] = read_PLOT3D_mesh(mesh_file);
    cells cells(x, y);

    const std::string outputDir = "output_files"; // Directory containing output files

    // Check if the directory exists
    if (!fs::exists(outputDir) || !fs::is_directory(outputDir)) {
        std::cerr << "Error: Directory " << outputDir << " does not exist." << std::endl;
        return 1;
    }

    // Map to store results grouped by Mach number
    std::map<double, std::vector<std::tuple<double, double, double, double>>> results;

    // Iterate over files in the directory
    for (const auto& entry : fs::directory_iterator(outputDir)) {
        if (entry.is_regular_file()) {
            auto [mach, alpha, C_L, C_D, C_M] = processFile(entry.path(), cells);
            results[mach].emplace_back(alpha, C_L, C_D, C_M); // Store alpha, C_L, C_D and C_M
        }
    }

    // Directory setup
    std::string databaseDir = "database";
    std::string meshName = fs::path(mesh_file).filename().string(); // Extract the file name with extension
    std::string databaseoutputDir = databaseDir + "/" + meshName;

    // Create directories if they don't exist
    fs::create_directories(databaseoutputDir);

    // Write CSV files for each Mach number
    for (auto& [mach, coeffs] : results) {
        // Sort the coefficients by alpha (first element in the tuple)
        std::sort(coeffs.begin(), coeffs.end(), [](const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs); // Compare alpha
        });

        // Create CSV file
        std::ostringstream filename;
        filename << databaseoutputDir << "/mach_" << std::fixed << std::setprecision(2) << mach << ".csv";
        std::ofstream outFile(filename.str());

        if (!outFile) {
            std::cerr << "Error: Could not create file " << filename.str() << std::endl;
            continue;
        }

        // Write header
        outFile << "Alpha (degrees),C_L,C_D,C_M\n";

        // Write data
        for (const auto& [alpha, C_L, C_D, C_M] : coeffs) {
            outFile << std::fixed 
                    << std::setprecision(2) << (alpha * 180.0 / M_PI) << "," // Alpha in degrees, 2 decimals
                    << std::setprecision(12) << C_L << ","                   // C_L
                    << C_D << ","                                           // C_D
                    << C_M << "\n";                                         // C_M
        }

        std::cout << "Written file: " << filename.str() << std::endl;
    }

    std::cout << "Processing completed." << std::endl;
    return 0;
}
