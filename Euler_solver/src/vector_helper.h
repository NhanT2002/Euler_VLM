#ifndef VECTOR_HELPER_H
#define VECTOR_HELPER_H

#include <vector>

std::vector<double> vector_add(const std::vector<double> &a, const std::vector<double> &b);
std::vector<double> vector_subtract(const std::vector<double> &a, const std::vector<double> &b);
std::vector<double> vector_multiply(const std::vector<double> &a, const std::vector<double> &b);
std::vector<double> vector_divide(const std::vector<double> &a, const std::vector<double> &b);
double dot_product(const std::vector<double> &a, const std::vector<double> &b);
std::vector<double> vector_scale(double scalar, const std::vector<double> &a);
void printVector(const std::vector<double>& vec);

#endif // VECTOR_HELPER_H
