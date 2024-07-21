#include <iostream>
#include <chrono>
#include <utility>
#include <random>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <omp.h>
#include <Eigen/SVD>

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) { return a + b+1 ; }

// Solve Ax = 0
inline Eigen::VectorXf svd(const Eigen::MatrixXf &A)
{
    // Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto V = svd.matrixV();                  // Get V matrix
    Eigen::VectorXf x = V.col(V.cols() - 1); // Get last column
    x /= x(0);                               // Divide by the first element
    return x;
}

// Get denominator and numerator of the rational function
std::pair<Eigen::Matrix<float, Eigen::Dynamic, 16>,
          Eigen::Matrix<float, Eigen::Dynamic, 16>>
getNumDen(const Eigen::VectorXf &theta)
{
    const auto sin2 = (2 * theta).array().sin();
    const auto cos2 = (2 * theta).array().cos();
    const auto sin10 = (10 * theta).array().sin();
    const auto cos10 = (10 * theta).array().cos();
    const auto sin20 = (20 * theta).array().sin();

    // pn (N, 16)
    Eigen::Matrix<float, Eigen::Dynamic, 16> pn(theta.size(), 16);
    pn.col(0) = Eigen::VectorXf::Zero(theta.size());
    pn.col(1) = -8.0 * sin2 * cos2;
    pn.col(2) = -4.0 * sin2 * sin2 + 4.0 * cos2 * cos2;
    pn.col(3) = 4.0 * cos2;
    pn.col(4) = -20.0 * sin20;
    pn.col(5) = -8.0 * sin2 * cos2 * cos10 * cos10 - 20.0 * sin20 * cos2 * cos2;
    pn.col(6) = -4.0 * sin2 * sin2 * cos10 * cos10 - 20.0 * sin2 * sin20 * cos2 + 4.0 * cos2 * cos2 * cos10 * cos10;
    pn.col(7) = -20.0 * sin2 * sin20 + 4.0 * cos2 * cos10 * cos10;
    pn.col(8) = -20.0 * sin10 * sin10 + 20.0 * cos10 * cos10;
    pn.col(9) = -4.0 * sin2 * sin20 * cos2 - 20.0 * sin10 * sin10 * cos2 * cos2 + 20.0 * cos2 * cos2 * cos10 * cos10;
    pn.col(10) = -2.0 * sin2 * sin2 * sin20 - 20.0 * sin2 * sin10 * sin10 * cos2 + 20.0 * sin2 * cos2 * cos10 * cos10 + 2.0 * sin20 * cos2 * cos2;
    pn.col(11) = -20.0 * sin2 * sin10 * sin10 + 20.0 * sin2 * cos10 * cos10 + 2.0 * sin20 * cos2;
    pn.col(12) = -20.0 * cos10;
    pn.col(13) = 8.0 * sin2 * sin10 * cos2 - 20.0 * cos2 * cos2 * cos10;
    pn.col(14) = 4.0 * sin2 * sin2 * sin10 - 20.0 * sin2 * cos2 * cos10 - 4.0 * sin10 * cos2 * cos2;
    pn.col(15) = -20.0 * sin2 * cos10 - 4.0 * sin10 * cos2;

    // pd (N, 16)
    Eigen::Matrix<float, Eigen::Dynamic, 16> pd(theta.size(), 16);
    pd.col(0) = 2 * Eigen::VectorXf::Ones(theta.size());
    pd.col(1) = 2 * cos2 * cos2;
    pd.col(2) = 2 * sin2 * cos2;
    pd.col(3) = 2 * sin2;
    pd.col(4) = 2 * cos10 * cos10;
    pd.col(5) = 2 * cos2 * cos2 * cos10 * cos10;
    pd.col(6) = 2 * sin2 * cos2 * cos10 * cos10;
    pd.col(7) = 2 * sin2 * cos10 * cos10;
    pd.col(8) = sin20;
    pd.col(9) = sin20 * cos2 * cos2;
    pd.col(10) = sin2 * sin20 * cos2;
    pd.col(11) = sin2 * sin20;
    pd.col(12) = -2 * sin10;
    pd.col(13) = -2 * sin10 * cos2 * cos2;
    pd.col(14) = -2 * sin2 * sin10 * cos2;
    pd.col(15) = -2 * sin2 * sin10;

    return {pn, pd};
}

Eigen::VectorXf init(const Eigen::VectorXf &theta, const Eigen::VectorXf &time_diff)
{
    auto [pn, pd] = getNumDen(theta);

    Eigen::VectorXf x_final = Eigen::VectorXf::Zero(16);
    float error_final = std::numeric_limits<float>::max();

    Eigen::Matrix<float, Eigen::Dynamic, 16> A(theta.size(), 16);
    A = (pn.array() - time_diff.replicate(1, 16).array() * pd.array());
    x_final = svd(A);
    Eigen::VectorXf time_diff_pred_ = (pn * x_final).array() / (pd * x_final).array();
    Eigen::VectorXf r_ = time_diff - time_diff_pred_;
    error_final = r_.array().abs().mean();

    Eigen::VectorXf x = Eigen::VectorXf::Zero(16);
    // [[0, 1, 2, 3],
    //  [4, 5, 6, 7],
    //  [8, 9, 10, 11],
    //  [12, 13, 14, 15]]
    x(0) = 1.0;

    // Random value geneartor
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

#pragma omp parallel for
    for (int i = 0; i < 5000; ++i)
    {
        x(5) = std::abs(dis(gen));
        x(10) = -x(5);
        x(15) = -x(5);

        // x(1) = 0.25 * dis(gen);
        // x(2) = 0.25 * dis(gen);
        // x(3) = 0.25 * dis(gen);
        // x(4) = 0.25 * dis(gen);
        // x(6) = 0.25 * dis(gen);
        // x(7) = 0.25 * dis(gen);
        // x(8) = 0.25 * dis(gen);
        // x(9) = 0.25 * dis(gen);
        // x(11) = 0.25 * dis(gen);
        // x(12) = 0.25 * dis(gen);
        // x(13) = 0.25 * dis(gen);
        // x(14) = 0.25 * dis(gen);

        Eigen::VectorXf time_diff_pred = (pn * x).array() / (pd * x).array();
        Eigen::VectorXf r = time_diff - time_diff_pred;
        float error = r.array().abs().mean();
        // float error = r.array().square().mean();
        if (error < error_final)
        {
            x_final = x;
            error_final = error;
        }
    }

    return x_final;
}

template <bool DEBUG = false>
Eigen::VectorXf fit(const Eigen::VectorXf &theta,
                    const Eigen::VectorXf &time_diff,
                    int max_iter = 50,
                    float eps = 1e-6,
                    float tol = 1e-3)
{
    // Count number of data points
    const int N = theta.size();
    if (N <= 0)
    {
        // Return nan
        return Eigen::VectorXf::Constant(16, std::numeric_limits<float>::quiet_NaN());
    }

    auto [pn, pd] = getNumDen(theta);

    // Add eps for pd
    // pd.array() += 1e-8;

    // A (N, 16)
    Eigen::Matrix<float, Eigen::Dynamic, 16> A(theta.size(), 16);
    A = (pn.array() - time_diff.replicate(1, 16).array() * pd.array());

    // Initial guess
    Eigen::Vector<float, 16> x;
    x = svd(A);
    // x = init(theta, time_diff);

    // IRLS loop
    Eigen::Matrix<float, Eigen::Dynamic, 16> Aw = A;
    float error_prev = std::numeric_limits<float>::max();
    for (int i_loop = 0; i_loop < max_iter; ++i_loop)
    {
        // x should be between -1 and 1
        x = x.cwiseMax(-1.0).cwiseMin(1.0);

        // Check convergence
        Eigen::VectorXf time_diff_pred = (pn * x).array() / (pd * x).array();
        Eigen::VectorXf r = time_diff - time_diff_pred;
        float error = r.array().abs().mean();
        if (std::abs(error_prev - error) < tol)
        {
            break;
        }
        error_prev = error;

        // Update matrix A by vector w
        Eigen::VectorXf w = 1.0 / (r.array().abs() + eps); // l1 weight
        for (int i = 0; i < 16; ++i)
        {
            Aw.col(i) = A.col(i).array() * w.array();
        }

        // Solve Ax = 0
        x = svd(Aw);

        if constexpr (DEBUG)
        {
            auto np = nb::module_::import_("numpy");
            auto plot = nb::module_::import_("matplotlib.pyplot");
            auto theta_np = np.attr("array")(theta);
            auto time_diff_np = np.attr("array")(time_diff);
            auto time_diff_pred_np = np.attr("array")(time_diff_pred);
            plot.attr("clf")();
            plot.attr("plot")(theta_np, time_diff_np, "o", "label"_a = "data");
            plot.attr("plot")(theta_np, time_diff_pred_np, "-", "label"_a = "fit");
            plot.attr("legend")();
            plot.attr("pause")(0.1);
        }
    }

    return x;
}

Eigen::VectorXf fit_bind(const nb::DRef<Eigen::VectorXf> &theta,
                         const nb::DRef<Eigen::VectorXf> &time_diff,
                         int max_iter = 50,
                         float eps = 1e-6,
                         float tol = 1e-3,
                         bool debug = false)
{
    if (debug)
    {
        return fit<true>(theta, time_diff, max_iter, eps, tol);
    }
    else
    {
        return fit<false>(theta, time_diff, max_iter, eps, tol);
    }
}

std::vector<Eigen::Vector<float, 16>> fit_batch(const std::vector<nb::DRef<Eigen::VectorXf>> &theta,
                                                const std::vector<nb::DRef<Eigen::VectorXf>> &time_diff,
                                                int max_iter = 50,
                                                float eps = 1e-6,
                                                float tol = 1e-3)
{
    int num = theta.size();
    std::vector<Eigen::Vector<float, 16>> result;
    result.resize(num);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num; ++i)
    {
        auto x = fit(theta[i], time_diff[i], max_iter, eps, tol);
        result.at(i) = x;
    }
    return result;
}

NB_MODULE(_eventellipsometry_impl, m)
{
    m.def("add", &add);
    m.def("fit_batch", &fit_batch, nb::arg("theta").noconvert(), nb::arg("time_diff").noconvert(), nb::arg("max_iter") = 50, nb::arg("eps") = 1e-6, nb::arg("tol") = 1e-3);
    m.def("fit", &fit_bind, nb::arg("theta").noconvert(), nb::arg("time_diff").noconvert(), nb::arg("max_iter") = 50, nb::arg("eps") = 1e-6, nb::arg("tol") = 1e-3, nb::arg("debug") = false);
}