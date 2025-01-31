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
double measure_flux_time(TensorType& tensor_result, TensorType& tensor_data, int config_id) {
    auto start = high_resolution_clock::now();
    double sum = 0.0f;

    tensor_result.setZero();

    if (config_id == 0) {

        vector<int> face_x = {0, 1};
        vector<int> face_y = {1, 0};

        for (int x = 0; x < X-1; ++x) {
            for (int y = 0; y < Y-1; ++y) {
                for (int k = 0; k < 2; ++k) {

                    int x_p1 = face_x[k] + x;
                    int y_p1 = face_y[k] + y;

                    // Average in k-direction
                    double x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x_p1, y_p1, 0));
                    double x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x_p1, y_p1, 1));
                    double x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x_p1, y_p1, 2));
                    double x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x_p1, y_p1, 3));

                    tensor_result(x, y, 0) += x0;
                    tensor_result(x, y, 1) += x1;
                    tensor_result(x, y, 2) += x2;
                    tensor_result(x, y, 3) += x3;

                    tensor_result(x_p1, y_p1, 0) -= x0;
                    tensor_result(x_p1, y_p1, 1) -= x1;
                    tensor_result(x_p1, y_p1, 2) -= x2;
                    tensor_result(x_p1, y_p1, 3) -= x3;
                }
            }
        }
    }

    // Average in the x-direction and then in the y-direction
    else if (config_id == 1) {
        for (int x = 0; x < X-1; ++x) {
            for (int y = 0; y < Y-1; ++y) {

                int x_p1 = x + 1;
                int y_p1 = y + 1;

                // Average in x-direction
                double x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x_p1, y, 0));
                double x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x_p1, y, 1));
                double x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x_p1, y, 2));
                double x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x_p1, y, 3));

                tensor_result(x, y, 0) += x0;
                tensor_result(x, y, 1) += x1;
                tensor_result(x, y, 2) += x2;
                tensor_result(x, y, 3) += x3;

                tensor_result(x_p1, y, 0) -= x0;
                tensor_result(x_p1, y, 1) -= x1;
                tensor_result(x_p1, y, 2) -= x2;
                tensor_result(x_p1, y, 3) -= x3;

                // Average in the y-direction
                x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x, y_p1, 0));
                x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x, y_p1, 1));
                x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x, y_p1, 2));
                x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x, y_p1, 3));

                tensor_result(x, y, 0) += x0;
                tensor_result(x, y, 1) += x1;
                tensor_result(x, y, 2) += x2;
                tensor_result(x, y, 3) += x3;

                tensor_result(x, y_p1, 0) -= x0;
                tensor_result(x, y_p1, 1) -= x1;
                tensor_result(x, y_p1, 2) -= x2;
                tensor_result(x, y_p1, 3) -= x3;
            }
        }
    }

    // Average in the y-direction first and then x-direction
    if (config_id == 2) {
        for (int x = 0; x < X-1; ++x) {
            for (int y = 0; y < Y-1; ++y) {

                int x_p1 = x + 1;
                int y_p1 = y + 1;

                // Average in the y-direction
                double x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x, y_p1, 0));
                double x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x, y_p1, 1));
                double x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x, y_p1, 2));
                double x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x, y_p1, 3));

                tensor_result(x, y, 0) += x0;
                tensor_result(x, y, 1) += x1;
                tensor_result(x, y, 2) += x2;
                tensor_result(x, y, 3) += x3;

                tensor_result(x, y_p1, 0) -= x0;
                tensor_result(x, y_p1, 1) -= x1;
                tensor_result(x, y_p1, 2) -= x2;
                tensor_result(x, y_p1, 3) -= x3;

                // Average in x-direction
                x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x_p1, y, 0));
                x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x_p1, y, 1));
                x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x_p1, y, 2));
                x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x_p1, y, 3));

                tensor_result(x, y, 0) += x0;
                tensor_result(x, y, 1) += x1;
                tensor_result(x, y, 2) += x2;
                tensor_result(x, y, 3) += x3;

                tensor_result(x_p1, y, 0) -= x0;
                tensor_result(x_p1, y, 1) -= x1;
                tensor_result(x_p1, y, 2) -= x2;
                tensor_result(x_p1, y, 3) -= x3;
            }
        }
    }

    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

template <typename TensorType>
auto measure_flux_time_with_return(TensorType& tensor_data, int config_id) {
    auto start = high_resolution_clock::now();
    double sum = 0.0f;

    Tensor<double, 3, RowMajor> tensor_result(X, Y, Z);
    tensor_result.setZero();

    if (config_id == 0) {

        vector<int> face_x = {1, 0};
        vector<int> face_y = {0, 1};

        for (int x = 0; x < X-1; ++x) {
            for (int y = 0; y < Y-1; ++y) {
                for (int k = 0; k < 2; ++k) {

                    int x_p1 = face_x[k] + x;
                    int y_p1 = face_y[k] + y;

                    // Average in k-direction
                    double x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x_p1, y_p1, 0));
                    double x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x_p1, y_p1, 1));
                    double x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x_p1, y_p1, 2));
                    double x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x_p1, y_p1, 3));

                    tensor_result(x, y, 0) += x0;
                    tensor_result(x, y, 1) += x1;
                    tensor_result(x, y, 2) += x2;
                    tensor_result(x, y, 3) += x3;

                    tensor_result(x_p1, y_p1, 0) -= x0;
                    tensor_result(x_p1, y_p1, 1) -= x1;
                    tensor_result(x_p1, y_p1, 2) -= x2;
                    tensor_result(x_p1, y_p1, 3) -= x3;
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    double req_time = duration<double, std::milli>(end - start).count();

    return std::make_tuple(req_time, tensor_result);
}

template <typename TensorType>
Tensor<double, 3, RowMajor> measure_flux_time_with_return_no_timer(TensorType& tensor_data, int config_id) {

    double sum = 0.0f;

    Tensor<double, 3, RowMajor> tensor_result(X, Y, Z);
    tensor_result.setZero();

    if (config_id == 0) {

        vector<int> face_x = {1, 0};
        vector<int> face_y = {0, 1};

        for (int x = 0; x < X-1; ++x) {
            for (int y = 0; y < Y-1; ++y) {
                for (int k = 0; k < 2; ++k) {

                    int x_p1 = face_x[k] + x;
                    int y_p1 = face_y[k] + y;

                    // Average in k-direction
                    double x0 = 0.5*(tensor_data(x, y, 0) + tensor_data(x_p1, y_p1, 0));
                    double x1 = 0.5*(tensor_data(x, y, 1) + tensor_data(x_p1, y_p1, 1));
                    double x2 = 0.5*(tensor_data(x, y, 2) + tensor_data(x_p1, y_p1, 2));
                    double x3 = 0.5*(tensor_data(x, y, 3) + tensor_data(x_p1, y_p1, 3));

                    tensor_result(x, y, 0) += x0;
                    tensor_result(x, y, 1) += x1;
                    tensor_result(x, y, 2) += x2;
                    tensor_result(x, y, 3) += x3;

                    tensor_result(x_p1, y_p1, 0) -= x0;
                    tensor_result(x_p1, y_p1, 1) -= x1;
                    tensor_result(x_p1, y_p1, 2) -= x2;
                    tensor_result(x_p1, y_p1, 3) -= x3;
                }
            }
        }
    }

    return tensor_result;
}

// // Function to measure access time for std::vector-based matrices
// double measure_vector_access_time(vector<double>& data, bool row_major) {
//     auto start = high_resolution_clock::now();
//     double sum = 0.0f;
//
//     if (row_major) {
//         for (int z = 0; z < Z; ++z)
//             for (int y = 0; y < Y; ++y)
//                 for (int x = 0; x < X; ++x)
//                     sum += data[z * (X * Y) + y * X + x];
//     } else {
//         for (int x = 0; x < X; ++x)
//             for (int y = 0; y < Y; ++y)
//                 for (int z = 0; z < Z; ++z)
//                     sum += data[x * (Y * Z) + y * Z + z];
//     }
//
//     auto end = high_resolution_clock::now();
//     return duration<double, std::milli>(end - start).count();
// }

// double measure_vector3d_access_time(vector<vector<vector<double>>>& data, bool row_major) {
//     auto start = high_resolution_clock::now();
//     double sum = 0.0f;
//
//     if (row_major) {
//         for (int z = 0; z < Z; ++z)
//             for (int y = 0; y < Y; ++y)
//                 for (int x = 0; x < X; ++x)
//                     sum += data[x][y][z];  // Row-major traversal
//     } else {
//         for (int x = 0; x < X; ++x)
//             for (int y = 0; y < Y; ++y)
//                 for (int z = 0; z < Z; ++z)
//                     sum += data[x][y][z];  // Column-major traversal
//     }
//
//     auto end = high_resolution_clock::now();
//     return duration<double, std::milli>(end - start).count();
// }

int main() {
    // Eigen Tensors
    // Tensor<double, 3> tensor_col_major(X, Y, Z);  // Default Column-Major
    // Tensor<double, 3, RowMajor> tensor_row_major(X, Y, Z);  // Explicit Row-Major

    Tensor<double, 3, RowMajor> tensor_col_data(X, Y, Z);
    // Tensor<double, 3, RowMajor> tensor_row_data(X, Y, Z);

    Tensor<double, 3, RowMajor> tensor_col_result(X, Y, Z);
    // Tensor<double, 3, RowMajor> tensor_row_result(X, Y, Z);

    // tensor_col_major.setRandom();
    // tensor_row_major.setRandom();
    //

    tensor_col_data.setRandom();
    // tensor_row_data.setRandom();

    tensor_col_result.setRandom();
    // tensor_row_result.setRandom();

    // // std::vector-based row-major and column-major matrices
    // vector<double> vec_row_major(X * Y * Z);
    // vector<double> vec_col_major(X * Y * Z);

    // for (size_t i = 0; i < vec_row_major.size(); ++i) {
    //     vec_row_major[i] = static_cast<double>(rand()) / RAND_MAX;
    //     vec_col_major[i] = vec_row_major[i];  // Keep values the same for fair comparison
    // }
    //
    // // nested std::vector
    // vector<vector<vector<double>>> vec_3d(X, vector<vector<double>>(Y, vector<double>(Z)));
    //
    // for (int x = 0; x < X; ++x)
    //     for (int y = 0; y < Y; ++y)
    //         for (int z = 0; z < Z; ++z)
    //             vec_3d[x][y][z] = static_cast<double>(rand()) / RAND_MAX;


    double time_config_0 = 0;
    double time_config_1 = 0;
    double time_config_2 = 0;
    double time_config_3 = 0;
    double time_config_4 = 0;
    double time_config_5 = 0;

    int N_TRIAL = 10;

    for(int i = 0; i < N_TRIAL; i++) {
        cout << "Trial: " << i+1 << " / " << N_TRIAL << endl;

        // Measure times for Eigen Tensors
        time_config_0 += measure_flux_time(tensor_col_result, tensor_col_data, 0);
        time_config_1 += measure_flux_time(tensor_col_result, tensor_col_data, 1);
        time_config_2 += measure_flux_time(tensor_col_result, tensor_col_data, 2);

        tuple<double, Tensor<double, 3, RowMajor>> res = measure_flux_time_with_return(tensor_col_data, 0);
        time_config_3 += get<0>(res);

        Tensor<double, 3, RowMajor> empty_tensor(X, Y, Z);
        auto start = high_resolution_clock::now();
        empty_tensor = measure_flux_time_with_return_no_timer(tensor_col_data, 0);
        auto end = high_resolution_clock::now();
        time_config_4 += duration<double, std::milli>(end - start).count();

        // Tensor<double, 3> empty_tensor(X, Y, Z);
        auto start2 = high_resolution_clock::now();
        auto empty_tensor2 = measure_flux_time_with_return_no_timer(tensor_col_data, 0);
        auto end2 = high_resolution_clock::now();
        time_config_5 += duration<double, std::milli>(end2 - start2).count();
    }




    // double time_col_major_wrong = measure_access_time(tensor_col_major, true);

    // double time_row_major = measure_access_time(tensor_row_major, true);
    // double time_row_major_wrong = measure_access_time(tensor_row_major, false);

    // // Measure times for std::vector
    // double time_vec_row_major = measure_vector_access_time(vec_row_major, true);
    // double time_vec_row_major_wrong = measure_vector_access_time(vec_row_major, false);
    //
    // double time_vec_col_major = measure_vector_access_time(vec_col_major, false);
    // double time_vec_col_major_wrong = measure_vector_access_time(vec_col_major, true);
    //
    // // Measure times for nested std::vector
    // double time_vec_3d_row_major = measure_vector3d_access_time(vec_3d, true);
    // double time_vec_3d_col_major = measure_vector3d_access_time(vec_3d, false);

    // Output results
    cout << "Eigen Tensors:" << endl;
    cout << "  Config 0 : " << time_config_0/N_TRIAL << " ms\n";
    cout << "  Config 1 : " << time_config_1/N_TRIAL << " ms\n";    // 8% faster
    cout << "  Config 2 : " << time_config_2/N_TRIAL << " ms\n";    // 45% slower

    // --> The additional time comes from the initialization of the temporary tensor in the function
    cout << "  Config 3 : " << time_config_3/N_TRIAL << " ms\n";    // This takes about twice more time than config 0
    cout << "  Config 4 : " << time_config_4/N_TRIAL << " ms\n";    // Takes about the same amount of time as config 4
    cout << "  Config 5 : " << time_config_5/N_TRIAL << " ms\n";

    // cout << "  Column-Major (Wrong Order)       : " << time_col_major_wrong << " ms\n";
    // cout << "  Row-Major (Correct Order)        : " << time_row_major << " ms\n";
    // cout << "  Row-Major (Wrong Order)          : " << time_row_major_wrong << " ms\n";

    // cout << "\nstd::vector (Manual 3D Matrix):" << endl;
    // cout << "  Row-Major (Correct Order)        : " << time_vec_row_major << " ms\n";
    // cout << "  Row-Major (Wrong Order)          : " << time_vec_row_major_wrong << " ms\n";
    // cout << "  Column-Major (Correct Order)     : " << time_vec_col_major << " ms\n";
    // cout << "  Column-Major (Wrong Order)       : " << time_vec_col_major_wrong << " ms\n";
    //
    // cout << "\nstd::vector<std::vector<std::vector<double>>>:" << endl;
    // cout << "  Row-Major Access Time  : " << time_vec_3d_row_major << " ms\n";
    // cout << "  Column-Major Access Time: " << time_vec_3d_col_major << " ms\n";

    return 0;
}
