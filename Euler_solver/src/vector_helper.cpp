#include "vector_helper.h"
#include <iostream>

// Function to add two vectors
std::vector<double> vector_add(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result; // Return by value
}

// Function to subtract two vectors
std::vector<double> vector_subtract(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result; // Return by value
}

// Function to multiply two vectors element-wise
std::vector<double> vector_multiply(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result; // Return by value
}

// Function to divide two vectors element-wise
std::vector<double> vector_divide(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] / b[i];
    }
    return result; // Return by value
}

// Function to compute the dot product of two vectors
double dot_product(const std::vector<double> &a, const std::vector<double> &b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result; // Return the dot product
}

// Function to scale a vector by a scalar
std::vector<double> vector_scale(double scalar, const std::vector<double> &a) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = scalar * a[i];
    }
    return result; // Return by value
}

// Function to print a vector
void printVector(const std::vector<double>& vec) {
    for (const double& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
