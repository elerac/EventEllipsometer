#include <iostream>
#include <chrono>
#include <utility>
#include <random>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <omp.h>
#include <Eigen/SVD>

#include "equations.h"
#include "optim.h"

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) { return a + b; }

Eigen::VectorXf init(const Eigen::VectorXf &theta, const Eigen::VectorXf &time_diff)
{
    auto [pn, pd] = calcNumenatorDenominatorCoffs(theta, 0.0, 0.0);

    Eigen::VectorXf x_final = Eigen::VectorXf::Zero(16);
    float error_final = std::numeric_limits<float>::max();

    Eigen::Matrix<float, Eigen::Dynamic, 16> A(theta.size(), 16);
    A = (pn.array() - time_diff.replicate(1, 16).array() * pd.array());
    x_final = svdSolve(A);
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
    for (int i = 0; i < 100000; ++i)
    {
        x(5) = dis(gen);
        x(10) = dis(gen);
        x(15) = dis(gen);

        x(1) = 0.25 * dis(gen);
        x(2) = 0.25 * dis(gen);
        x(3) = 0.25 * dis(gen);
        x(4) = 0.25 * dis(gen);
        x(6) = 0.25 * dis(gen);
        x(7) = 0.25 * dis(gen);
        x(8) = 0.25 * dis(gen);
        x(9) = 0.25 * dis(gen);
        x(11) = 0.25 * dis(gen);
        x(12) = 0.25 * dis(gen);
        x(13) = 0.25 * dis(gen);
        x(14) = 0.25 * dis(gen);

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
                    float phi1,
                    float phi2,
                    float delta = 1.35,
                    int max_iter = 50,
                    float tol = 1e-3)
{
    // Count number of data points
    size_t num = theta.size();

    // Return nan if no data
    if (num == 0)
    {
        return Eigen::VectorXf::Constant(16, std::numeric_limits<float>::quiet_NaN());
    }

    auto [pn, pd] = calcNumenatorDenominatorCoffs(theta, phi1, phi2);

    Eigen::Matrix<float, Eigen::Dynamic, 16> A(num, 16);
    A = (pn.array() - time_diff.replicate(1, 16).array() * pd.array());

    // Weight by time_diff
    Eigen::VectorXf inv_time_diff = 1.0f / time_diff.array().abs();
    for (int i = 0; i < 16; ++i)
    {
        A.col(i) = A.col(i).array() * inv_time_diff.array();
    }

    // Initial guess
    Eigen::Vector<float, 16> x;
    x = svdSolve(A);

    // IRLS loop
    Eigen::Matrix<float, Eigen::Dynamic, 16> Aw(num, 16);
    float error_prev = std::numeric_limits<float>::max();
    for (int i_loop = 0; i_loop < max_iter; ++i_loop)
    {
        // x should be between -1 and 1
        x = x.cwiseMax(-1.0).cwiseMin(1.0);

        // float length = x(1) * x(1) + x(2) * x(2) + x(3) * x(3) + x(4) * x(4);
        // if (length > 1.0)
        // {
        //     length = std::sqrt(length);
        //     x(1) /= length;
        //     x(2) /= length;
        //     x(3) /= length;
        // }

        // length = x(5) * x(5) + x(9) * x(9) + x(13) * x(13) + x(15) * x(15);
        // if (length > 1.0)
        // {
        //     length = std::sqrt(length);
        //     x(5) /= length;
        //     x(9) /= length;
        //     x(13) /= length;
        // }
        // x = x.cwiseMax(-1.0).cwiseMin(1.0);

        // Check convergence
        Eigen::VectorXf time_diff_pred = (pn * x).array() / (pd * x).array();
        Eigen::VectorXf r = time_diff - time_diff_pred;
        auto r_abs = r.array().abs();
        float error = r_abs.mean();
        if (std::abs(error_prev - error) < tol)
        {
            break;
        }
        error_prev = error;

        // l1 weight
        // Eigen::VectorXf w = 1.0 / (r.array().abs() + eps);

        // Huber weight
        const Eigen::VectorXf w = (r_abs < delta).select(1.0, delta / r_abs);

        for (int i = 0; i < 16; ++i)
        {
            Aw.col(i) = A.col(i).array() * w.array();
        }

        x = svdSolve(Aw);

        if constexpr (DEBUG)
        {
            std::cout << "iter: " << i_loop << "/" << max_iter << ", error: " << error << std::endl;
            auto np = nb::module_::import_("numpy");
            auto plt = nb::module_::import_("matplotlib.pyplot");
            auto theta_np = np.attr("array")(theta);
            auto time_diff_np = np.attr("array")(time_diff);
            auto time_diff_pred_np = np.attr("array")(time_diff_pred);
            auto w_np = np.attr("array")(w / w.maxCoeff());
            plt.attr("clf")();
            plt.attr("scatter")(theta_np, time_diff_np, "label"_a = "data");
            plt.attr("scatter")(theta_np, time_diff_pred_np, "label"_a = "fit", "c"_a = w_np, "cmap"_a = "viridis");
            plt.attr("legend")();
            // plt.attr("pause")(1);
            plt.attr("show")();
        }
    }

    return x;
}

// std::vector<Eigen::Vector<float, 16>> fit_batch(const std::vector<nb::DRef<Eigen::VectorXf>> &theta,
//                                                 const std::vector<nb::DRef<Eigen::VectorXf>> &time_diff,
//                                                 int max_iter = 50,
//                                                 float eps = 1e-6,
//                                                 float tol = 1e-3)
// {
//     int num = theta.size();
//     std::vector<Eigen::Vector<float, 16>> result;
//     result.resize(num);

// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < num; ++i)
//     {
//         auto x = fit(theta[i], time_diff[i], max_iter, eps, tol);
//         result.at(i) = x;
//     }
//     return result;
// }

NB_MODULE(_eventellipsometry_impl, m)
{
    m.def("add", &add);
    m.def("svdSolve", [](const nb::DRef<Eigen::MatrixXf> &A)
          { return svdSolve(A); }, nb::arg("A").noconvert(), "Solve Ax = 0\n\nParameters\n----------\nA : numpy.ndarray\n    Matrix A. (n, m)\n\nReturns\n-------\nx : numpy.ndarray\n    Solution x. (m,). The x is normalized by first element.");
    // m.def("fit_batch", &fit_batch, nb::arg("theta").noconvert(), nb::arg("time_diff").noconvert(), nb::arg("max_iter") = 50, nb::arg("eps") = 1e-6, nb::arg("tol") = 1e-3);
    m.def("fit", [](const nb::DRef<Eigen::VectorXf> &theta, const nb::DRef<Eigen::VectorXf> &time_diff, float phi1, float phi2, float delta = 1.35, int max_iter = 50, float tol = 1e-3, bool debug = false)
          {
              if (debug)
                  return fit<true>(theta, time_diff, phi1, phi2, delta, max_iter, tol);
              else
                  return fit<false>(theta, time_diff, phi1, phi2, delta, max_iter, tol);
              //
          },
          nb::arg("theta").noconvert(), nb::arg("time_diff").noconvert(), nb::arg("phi1"), nb::arg("phi2"), nb::arg("delta") = 1.35, nb::arg("max_iter") = 50, nb::arg("tol") = 1e-3, nb::arg("debug") = false, "Fit the data");
    m.def("calcNumenatorDenominatorCoffs", [](const nb::DRef<Eigen::VectorXf> &theta, float phi1, float phi2)
          { return calcNumenatorDenominatorCoffs(theta, phi1, phi2); }, nb::arg("theta").noconvert(), nb::arg("phi1"), nb::arg("phi2"), "Calculate numenator and denominator cofficients");
    m.def("diffLn", [](const nb::DRef<Eigen::Vector<float, 16>> &M, const nb::DRef<Eigen::VectorXf> &theta, float phi1, float phi2)
          { return diffLn(M, theta, phi1, phi2); }, nb::arg("M").noconvert(), nb::arg("theta").noconvert(), nb::arg("phi1"), nb::arg("phi2"));
}