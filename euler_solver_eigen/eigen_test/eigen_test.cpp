#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

constexpr int X = 4096, Y = 4096, Z = 4;  // Tensor size

// Function to measure access time for Eigen tensors
template <typename TensorType>
double measure_access_time(TensorType& tensor, bool row_major) {
    auto start = high_resolution_clock::now();
    double sum = 0.0f;

    if (row_major) {
        for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                    sum += tensor(x, y, z);
    } else {
        for (int x = 0; x < X; ++x)
            for (int y = 0; y < Y; ++y)
                for (int z = 0; z < Z; ++z)
                    sum += tensor(x, y, z);
    }

    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

// Function to measure access time for std::vector-based matrices
double measure_vector_access_time(vector<double>& data, bool row_major) {
    auto start = high_resolution_clock::now();
    double sum = 0.0f;

    if (row_major) {
        for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                    sum += data[z * (X * Y) + y * X + x];
    } else {
        for (int x = 0; x < X; ++x)
            for (int y = 0; y < Y; ++y)
                for (int z = 0; z < Z; ++z)
                    sum += data[x * (Y * Z) + y * Z + z];
    }

    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

double measure_vector3d_access_time(vector<vector<vector<double>>>& data, bool row_major) {
    auto start = high_resolution_clock::now();
    double sum = 0.0f;

    if (row_major) {
        for (int z = 0; z < Z; ++z)
            for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                    sum += data[x][y][z];  // Row-major traversal
    } else {
        for (int x = 0; x < X; ++x)
            for (int y = 0; y < Y; ++y)
                for (int z = 0; z < Z; ++z)
                    sum += data[x][y][z];  // Column-major traversal
    }

    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

int main() {
    // Eigen Tensors
    Tensor<double, 3> tensor_col_major(X, Y, Z);  // Default Column-Major
    Tensor<double, 3, RowMajor> tensor_row_major(X, Y, Z);  // Explicit Row-Major

    tensor_col_major.setRandom();
    tensor_row_major.setRandom();

    // std::vector-based row-major and column-major matrices
    vector<double> vec_row_major(X * Y * Z);
    vector<double> vec_col_major(X * Y * Z);

    for (size_t i = 0; i < vec_row_major.size(); ++i) {
        vec_row_major[i] = static_cast<double>(rand()) / RAND_MAX;
        vec_col_major[i] = vec_row_major[i];  // Keep values the same for fair comparison
    }

    // nested std::vector
    vector<vector<vector<double>>> vec_3d(X, vector<vector<double>>(Y, vector<double>(Z)));

    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y)
            for (int z = 0; z < Z; ++z)
                vec_3d[x][y][z] = static_cast<double>(rand()) / RAND_MAX;

    // Measure times for Eigen Tensors
    double time_col_major = measure_access_time(tensor_col_major, false);
    double time_col_major_wrong = measure_access_time(tensor_col_major, true);

    double time_row_major = measure_access_time(tensor_row_major, true);
    double time_row_major_wrong = measure_access_time(tensor_row_major, false);

    // Measure times for std::vector
    double time_vec_row_major = measure_vector_access_time(vec_row_major, true);
    double time_vec_row_major_wrong = measure_vector_access_time(vec_row_major, false);

    double time_vec_col_major = measure_vector_access_time(vec_col_major, false);
    double time_vec_col_major_wrong = measure_vector_access_time(vec_col_major, true);

    // Measure times for nested std::vector
    double time_vec_3d_row_major = measure_vector3d_access_time(vec_3d, true);
    double time_vec_3d_col_major = measure_vector3d_access_time(vec_3d, false);

    // Output results
    cout << "Eigen Tensors:" << endl;
    cout << "  Column-Major (Correct Order)     : " << time_col_major << " ms\n";
    cout << "  Column-Major (Wrong Order)       : " << time_col_major_wrong << " ms\n";
    cout << "  Row-Major (Correct Order)        : " << time_row_major << " ms\n";
    cout << "  Row-Major (Wrong Order)          : " << time_row_major_wrong << " ms\n";

    cout << "\nstd::vector (Manual 3D Matrix):" << endl;
    cout << "  Row-Major (Correct Order)        : " << time_vec_row_major << " ms\n";
    cout << "  Row-Major (Wrong Order)          : " << time_vec_row_major_wrong << " ms\n";
    cout << "  Column-Major (Correct Order)     : " << time_vec_col_major << " ms\n";
    cout << "  Column-Major (Wrong Order)       : " << time_vec_col_major_wrong << " ms\n";

    cout << "\nstd::vector<std::vector<std::vector<double>>>:" << endl;
    cout << "  Row-Major Access Time  : " << time_vec_3d_row_major << " ms\n";
    cout << "  Column-Major Access Time: " << time_vec_3d_col_major << " ms\n";

    return 0;
}
