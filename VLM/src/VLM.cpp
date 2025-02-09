#include <iostream>
#include <fstream>
#include <iomanip> // std::setprecision
#include <tuple>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <Eigen/Dense>
# define M_PI 3.14159265358979323846 

typedef double T;
constexpr T EPS = std::numeric_limits<T>::epsilon();
using namespace std;
using namespace Eigen;

// Structure de Point en 3D
struct Point {
    double x, y, z;

    Point() : x(0), y(0), z(0) {}
    Point(double x, double y, double z) : x(x), y(y), z(z) {}

    // Opérateur pour additionner des points
    Point operator+(const Point& p) const {
        return Point(x + p.x, y + p.y, z + p.z);
    }

    // Opérateur pour soustraire des points
    Point operator-(const Point& p) const {
        return Point(x - p.x, y - p.y, z - p.z);
    }

    // Opérateur pour multiplier un scalaire à un point
    Point operator*(double scal) const {
        return Point(x * scal, y * scal, z * scal);
    }

    // Normaliser un point pour obtenir un vecteur unitaire
    Point normalize() const {
        double length = std::sqrt(x * x + y * y + z * z);
        return Point(x / length, y / length, z / length);
    }

    // Produit scalaire
    double dot(const Point& p) const {
        return x * p.x + y * p.y + z * p.z;
    }

    double longueur() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // Produit vectoriel
    Point cross(const Point& p) const {
        return Point(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
    }
};

// Structure de Panel en 3D
struct Panel {
    Point p1, p2, p3, p4;

    // Default constructor
    Panel() : p1(), p2(), p3(), p4() {}

    // Parameterized constructor
    Panel(Point p1, Point p2, Point p3, Point p4) : p1(p1), p2(p2), p3(p3), p4(p4) {}

    // Calcul de la normale d'un panneau
    Point normale(const Panel& p) const {
        Point A = p.p3 - p.p1;
        Point B = p.p2 - p.p4;
        return A.cross(B).normalize();
    }

    // Calcul du point de collocation 
    Point colloc(const Panel& p) const {
        Point col = p.p1 + p.p2 + p.p3 + p.p4;
        return col * 0.25;
    }

    // Calcul de l'aire du panneau
    double aire(const Panel& p) const {
        Point a = p.p2 - p.p1;
        Point b = p.p3 - p.p1;
        Point c = p.p4 - p.p1;
        Point d = a.cross(b);
        Point e = b.cross(c);
        return 0.5 * (std::sqrt(d.dot(d)) + std::sqrt(e.dot(e)));
    }

    // Calcul de la largeur d'un panneau
    double delta_y(const Panel& p) {
        Point a = p.p4 - p.p1;
        return std::sqrt(a.dot(a));
    }

    // Calcul de la largeur d'un panneau (vecteur)
    Point delta_y_vec(const Panel& p) {
        return p.p4 - p.p1;
    }

    // Calcul du panneau équivalent sur la demi-aile opposée
    Panel panelInv(const Panel& p) {
        Point a = p.p1;
        a.y *= -1;
        Point b = p.p2;
        b.y *= -1;
        Point c = p.p3;
        c.y *= -1;
        Point d = p.p4;
        d.y *= -1;
        return Panel(a, b, c, d);
    }
};







// Fonction pour calculer la circulation du vortex (Vortxl)
Point Vortxl(const Point& p, const Point& p1, const Point& p2, double gamma = 1) {
    Point r0 = p2 - p1;
    Point r1 = p - p1;
    Point r2 = p - p2;

    double r1_2 = std::sqrt(r1.dot(r1));
    double r2_2 = std::sqrt(r2.dot(r2));

    Point r1xr2 = r1.cross(r2);
    double r1xr2_2 = r1xr2.dot(r1xr2);

    double denom = 4 * M_PI * r1xr2_2;
    double terme1 = r0.dot(r1) / r1_2;
    double terme2 = r0.dot(r2) / r2_2;

    double K = gamma / denom * (terme1 - terme2);

    return Point(K * r1xr2.x, K * r1xr2.y, K * r1xr2.z);
}

// Fonction pour calculer le champ de vitesse (Voring)
Point Voring(const Point& p, const Panel& panel, double gamma = 1) {
    Point u1 = Vortxl(p, panel.p1, panel.p2);
    Point u2 = Vortxl(p, panel.p2, panel.p3);
    Point u3 = Vortxl(p, panel.p3, panel.p4);
    Point u4 = Vortxl(p, panel.p4, panel.p1);

    return Point{ u1.x + u2.x + u3.x + u4.x, 
        u1.y + u2.y + u3.y + u4.y, 
        u1.z + u2.z + u3.z + u4.z };
}

Point Voring2(const Point& p, const Panel& panel, double gamma = 1) {
    Point u1 = Vortxl(p, panel.p1, panel.p2);
    Point u3 = Vortxl(p, panel.p3, panel.p4);

    return Point{u1.x + u3.x, u1.y + u3.y, u1.z + u3.z};
}

// Fonction pour calculer le coefficient a
double coefA(const Point& p, const Panel& panel, const Point& n) {
    return Voring(p, panel).dot(n);
}

// Fonction pour calculer le coefficient a d'une aile avec le maillage d'une demi-aile
double coefA_sym(const Point& p, Panel& panel, const Point& n) {
    Panel panelSym = panel.panelInv(panel);
    Point aSym = Voring(p, panel) + Voring(p, panelSym);
    return aSym.dot(n);
}

// Fonction pour calculer le coefficient b
double coefB(const Point& p, const Panel& panel, const Point& n) {
    return Voring2(p, panel).dot(n);
}

// Fonction pour calculer le terme RHS
double funcRHS(const Point& U_inf, const Point& n) {
    return -U_inf.dot(n);
}


// Fonction pour créer un maillage temporaire
std::tuple<std::vector<Panel>, std::vector<Panel>, std::vector<Panel>> maillage(int ny, int nx, double span, double cord) {
    double dy = span / ny;
    double dx = cord / nx;
    ny += 1;
    nx += 1;
    double x = 0, y = 0, z = 0;
    
    std::vector<Point> points(nx * ny, Point());
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            points[i * ny + j] = Point(x, y, z);
            y += dy;
        }
        x += dx;
        y = 0;
    }


    std::vector<Point> points2(nx * ny, Point());
    std::vector<Point> points2W(ny, Point());

    for (int i = 0; i < nx * ny - ny; ++i) {
        points2[i] = (points[i + ny] - points[i]) * 0.25 + points[i];
    }

    for (int i = 0; i < ny; ++i) {
        points2[points2.size() - i - 1] = (points[points2.size() - i - 1] - points[points2.size() - i - 1 - ny]) * 0.25 + points[points2.size() - i - 1];
        points2W[ny - i - 1] = points2[points2.size() - i - 1] + Point(100 * (points[points2.size() - i - 1].x - points[points2.size() - i - 1 + ny].x),
            0,
            100 * (points[points2.size() - i - 1].z - points[points2.size() - i - 1 + ny].z));
    }

    ny -= 1;
    nx -= 1;

    // Créer les panneaux
    std::vector<Panel> panels(nx * ny);
    int dec1 = 0, dec2 = 1;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            Point p1 = points[i * ny + j + 1 + dec1];
            Point p2 = points[(i + 1) * ny + j + 1 + dec2];
            Point p3 = points[(i + 1) * ny + j + dec2];
            Point p4 = points[i * ny + j + dec1];
            panels[i * ny + j] = Panel(p1, p2, p3, p4);
        }
        dec1 += 1;
        dec2 += 1;
    }

    // Créer les panneaux 2 et panneauxW
    std::vector<Panel> panels2(nx * ny);
    std::vector<Panel> panelsW(ny);
    dec1 = 0;
    dec2 = 1;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            Point p1 = points2[i * ny + j + 1 + dec1];
            Point p2 = points2[(i + 1) * ny + j + 1 + dec2];
            Point p3 = points2[(i + 1) * ny + j + dec2];
            Point p4 = points2[i * ny + j + dec1];
            panels2[i * ny + j] = Panel(p1, p2, p3, p4);
        }
        dec1 += 1;
        dec2 += 1;
    }

    for (int i = 0; i < ny; ++i) {
        Point p1 = points2[points2.size() - i - 1];
        Point p2 = points2W[ny - i - 1];
        Point p3 = points2W[ny - i - 2];
        Point p4 = points2[points2.size() - i - 2];
        panelsW[ny - i - 1] = Panel(p1, p2, p3, p4);
    }

    // panels est formé des panneaux géométriques
    // panels2 est formé des panneaux de circulation
    // panelsW est formé des panneaux de sillage
    return { panels, panels2, panelsW };
}

void ecriture(const std::vector<Point>& p_colloc, const std::vector<double>& p_ij) {
    std::ofstream fichier("Pressions.csv");
    fichier << "x,y,z,Pression\n";
    for (size_t i = 0; i < p_ij.size(); ++i) {
        fichier << p_colloc[i].x << "," << p_colloc[i].y << "," << p_colloc[i].z << "," << p_ij[i] << "\n";
    }
}




int main()
{
    int ny = 20;  
    int nx = 2;  
    double AR = 7.28;  
    double alpha_input = 5.0;
    std::vector<double> alpha(ny, alpha_input);

    double Q_inf = 1.0;
    double rho = 1.0;
    double glisse = 0.0;


    std::vector<Panel> panels, panels2, panelsW;
    std::tie(panels, panels2, panelsW) = maillage(ny, nx, AR, 1.0);

    int n = panels.size();  // Nombre total de panneaux
    std::vector<Point> p_colloc(n);
    std::vector<Point> n_panel(n);

    // Vitesse de l'écoulement (U_inf) et initialisation des matrices
    //std::vector<std::vector<double>> a_ij(n, std::vector<double>(n, 0.0));
    //std::vector<std::vector<double>> b_ij(n, std::vector<double>(n, 0.0));
    //std::vector<double> RHS(n, 0.0);
    //std::vector<double> gamma(n, 0.0);  // Coefficients de circulation
    MatrixXd a_ij(n, n), b_ij(n, n);
    VectorXd RHS(n), gamma(n), w_ind(n);


    // Définir la vitesse à l'infini en fonction de alpha
    std::vector<Point> U_inf(ny, Point(0.0, 0.0, 0.0));  
    for (size_t i = 0; i < ny; ++i) {
        U_inf[i] = Point(Q_inf*std::cos(alpha[i] * M_PI / 180.0) * std::cos(glisse * M_PI / 180.0),
                         Q_inf * std::sin(glisse * M_PI / 180.0) * std::cos(alpha[i] * M_PI / 180.0),
                         Q_inf * std::sin(alpha[i] * M_PI / 180.0));
    }

    // Calcul des points de collocation et des normales
    for (int i = 0; i < n; ++i) {
        p_colloc[i] = panels[i].colloc(panels[i]);
        n_panel[i] = panels[i].normale(panels[i]);
    }

    // Calcul des coefficients a_ij et b_ij
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - ny; ++j) {
            a_ij(i, j) = coefA(p_colloc[i], panels[j], n_panel[j]);
            b_ij(i, j) = coefB(p_colloc[i], panels[j], n_panel[j]);
        }

        int indW = 0;
        for (int j = n - ny; j < n; ++j) {
            a_ij(i, j) = coefA(p_colloc[i], panels[j], n_panel[j]) + coefA(p_colloc[i], panelsW[indW], n_panel[j]);
            b_ij(i, j) = coefB(p_colloc[i], panels[j], n_panel[j]) + coefB(p_colloc[i], panelsW[indW], n_panel[j]);
            ++indW;
        }
    }

    // Calcul des termes RHS en fonction de la vitesse à l'infini
    for (int i = 0; i < n; ++i) {
        RHS(i) = funcRHS(U_inf[i % alpha.size()], n_panel[i]);
    }

    gamma = a_ij.colPivHouseholderQr().solve(RHS);
    w_ind = b_ij*gamma;


    std::vector<double> aire_ij(n, 0.0), dy_ij(n, 0.0), L_ij(n, 0.0), p_ij(n, 0.0), D_ij(n, 0.0);
    std::vector<Point> dy_ij_vec(n);
    double L = 0.0, D = 0.0, CL = 0.0, CD = 0.0, A = 0.0;

    for (int i = 0; i < ny; ++i) {
        aire_ij[i] = panels[i].aire(panels[i]);
        dy_ij[i] = panels[i].delta_y(panels[i]);
        dy_ij_vec[i] = panels[i].delta_y_vec(panels[i]);
        L_ij[i] = rho * gamma[i] * U_inf[i % ny].cross(dy_ij_vec[i]).longueur();
        p_ij[i] = L_ij[i] / aire_ij[i];
        D_ij[i] = -rho * gamma[i] * w_ind[i] * dy_ij[i];
    }

    for (int i = ny; i < n; ++i) {
        aire_ij[i] = panels[i].aire(panels[i]);
        dy_ij[i] = panels[i].delta_y(panels[i]);
        dy_ij_vec[i] = panels[i].delta_y_vec(panels[i]);
        L_ij[i] = rho * (gamma[i] - gamma[i - ny]) * U_inf[i % ny].cross(dy_ij_vec[i]).longueur();
        p_ij[i] = L_ij[i] / aire_ij[i];
        D_ij[i] = -rho / 2.0 * (gamma[i] - gamma[i - ny]) * w_ind[i] * dy_ij[i];
    }

    // Calcul des forces totales et des coefficients de portance et de traînée
    A = std::accumulate(aire_ij.begin(), aire_ij.end(), 0.0);  // 2 ailes, donc x2 ????????????????
    L = std::accumulate(L_ij.begin(), L_ij.end(), 0.0);
    D = std::accumulate(D_ij.begin(), D_ij.end(), 0.0);

    CL = L / (0.5 * rho * Q_inf * Q_inf * A);
    CD = D / (0.5 * rho * Q_inf * Q_inf * A);  

    // Calcul du coefficient de portance local (Cl)
    std::vector<double> Cl(ny, 0.0), aire_span(ny, 0.0);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            Cl[i] += L_ij[j * ny + i];
            aire_span[i] += aire_ij[j * ny + i];
        }
        Cl[i] /= (0.5 * rho * Q_inf * Q_inf * aire_span[i]);
    }

    std::cout << "CL = " << CL << " CD = " << CD << std::endl;

    return 0;
}