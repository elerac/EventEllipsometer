#include <iostream>
#include <chrono>
#include <utility>
#include <random>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>

#include <omp.h>
#include <Eigen/SVD>

#include "equations.h"
#include "optim.h"

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) { return a + b; }

float median(const Eigen::VectorXf &v)
{
    auto v_ = v;
    std::sort(v_.data(), v_.data() + v_.size());
    return v_(v_.size() / 2);
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

    Eigen::Matrix<float, Eigen::Dynamic, 16> Ainit = A;

    // Weight by time_diff
    Eigen::VectorXf inv_time_diff = 1.0f / time_diff.array().abs();
    for (int i = 0; i < 16; ++i)
    {
        A.col(i) = A.col(i).array() * inv_time_diff.array();
        // Ainit.col(i) = Ainit.col(i).array() * inv_time_diff.array();
    }

    // Initial guess
    Eigen::Vector<float, 16> x;
    x = svdSolve(A);

    // IRLS loop
    Eigen::Matrix<float, Eigen::Dynamic, 16> Aw(num, 16);
    float error_prev = std::numeric_limits<float>::max();
    float error_best = std::numeric_limits<float>::max();
    Eigen::Vector<float, 16> x_best = x;
    for (int i_loop = 0; i_loop < max_iter; ++i_loop)
    {
        // x should be between -1 and 1
        // x = x.cwiseMax(-1.0).cwiseMin(1.0);

        // Check convergence
        Eigen::VectorXf time_diff_pred = (pn * x).array() / (pd * x).array();
        Eigen::VectorXf r = time_diff - time_diff_pred;
        auto r_abs = r.array().abs();
        // auto r_sq = r.array().square();
        float error = r_abs.mean();
        if (error < error_best)
        {
            error_best = error;
            x_best = x;
        }

        float s = median((r.array() - median(r)).abs()) / 0.6745;
        auto r_abs_scaled = (r_abs / s);

        if (std::abs(error_prev - error) < tol)
        {
            break;
        }
        error_prev = error;

        // l1 weight
        // Eigen::VectorXf w = 1.0 / (r.array().abs() + eps);

        // Huber weight
        // const Eigen::VectorXf w = (r_abs < delta).select(1.0, delta / r_abs);
        Eigen::VectorXf w = (r_abs_scaled < delta).select(1.0, delta / r_abs_scaled);

        for (int i = 0; i < 16; ++i)
        {
            Aw.col(i) = A.col(i).array() * w.array();
        }

        x = svdSolve(Aw);

        if constexpr (DEBUG)
        {
            std::cout << "iter: " << i_loop << "/" << max_iter << ", error: " << error << std::endl;
            std::cout << "s: " << s << std::endl;
            auto np = nb::module_::import_("numpy");
            auto plt = nb::module_::import_("matplotlib.pyplot");
            auto theta_np = np.attr("array")(theta);
            auto time_diff_np = np.attr("array")(time_diff);
            auto time_diff_pred_np = np.attr("array")(time_diff_pred);
            auto w_np = np.attr("array")(w / (1e-8 + w.maxCoeff()));
            plt.attr("clf")();
            plt.attr("plot")(theta_np, time_diff_pred_np);
            plt.attr("scatter")(theta_np, time_diff_np, "label"_a = "data");
            plt.attr("scatter")(theta_np, time_diff_pred_np, "label"_a = "fit", "alpha"_a = w_np);
            plt.attr("legend")();
            // plt.attr("pause")(1);

            Eigen::Vector<float, 16> x_ = {1, 0, 0, 0,
                                           0, 0.99, 0, 0,
                                           0, 0, -0.99, 0,
                                           0, 0, 0, -0.99};
            // Eigen::Vector<float, 16> x_ = {1, -0.99, 0, 0,
            //                                -0.99, 0.99, 0, 0,
            //                                0, 0, 0, 0,
            //                                0, 0, 0, 0};
            Eigen::VectorXf time_diff_pred_gt = (pn * x_).array() / (pd * x_).array();
            Eigen::VectorXf r_gt = time_diff - time_diff_pred_gt;
            float error_gt = r_gt.array().abs().mean();

            std::cout << "error_gt: " << error_gt << std::endl;
            auto time_diff_pred_gt_np = np.attr("array")(time_diff_pred_gt);
            plt.attr("plot")(theta_np, time_diff_pred_gt_np, "label"_a = "gt", "linestyle"_a = "--", "color"_a = "tab:green");

            auto cv2 = nb::module_::import_("cv2");
            auto ee = nb::module_::import_("eventellipsometry");

            // Show Mueller image
            Eigen::Matrix<float, 4, 4> M;
            M.row(0) = x.segment(0, 4).transpose();
            M.row(1) = x.segment(4, 4).transpose();
            M.row(2) = x.segment(8, 4).transpose();
            M.row(3) = x.segment(12, 4).transpose();
            auto M_np = np.attr("array")(M);
            auto img_M_np = ee.attr("mueller_image")(M_np);
            cv2.attr("imshow")("img", img_M_np);
            cv2.attr("moveWindow")("img", 0, 0);
            cv2.attr("waitKey")(1);

            plt.attr("show")();
            plt.attr("close")();
        }
    }

    return x_best;
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

auto cvtEventStrucure(
    nb::ndarray<const uint16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> x,
    nb::ndarray<const uint16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> y,
    nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> t,
    nb::ndarray<const int16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> p,
    int width,
    int height,
    std::optional<int64_t> t_min,
    std::optional<int64_t> t_max)
{
    // check size should be same
    if (x.shape(0) != y.shape(0) || x.shape(0) != t.shape(0) || x.shape(0) != p.shape(0))
    {
        throw std::invalid_argument("x, y, t, p should have same size");
    }

    size_t num = x.shape(0);

    // Initialize image structure
    std::vector<std::vector<std::vector<int64_t>>> img_t(height, std::vector<std::vector<int64_t>>(width));
    std::vector<std::vector<std::vector<int16_t>>> img_p(height, std::vector<std::vector<int16_t>>(width));
    img_t.resize(height);
    img_p.resize(height);
    for (size_t i = 0; i < height; ++i)
    {
        img_t[i].resize(width);
        img_p[i].resize(width);
    }

    bool enable_t_range = t_min.has_value() || t_max.has_value();
    if (!t_min.has_value() && t_max.has_value())
    {
        t_min = 0;
    }
    if (!t_max.has_value() && t_min.has_value())
    {
        t_max = std::numeric_limits<int64_t>::max();
    }

    // Fill
    auto x_v = x.view();
    auto y_v = y.view();
    auto t_v = t.view();
    auto p_v = p.view();
    for (size_t i = 0; i < num; ++i)
    {
        int64_t t_ = t_v(i);

        if (enable_t_range)
        {
            if (t_ < t_min || t_ > t_max)
            {
                continue;
            }
        }

        uint16_t x_ = x_v(i);
        uint16_t y_ = y_v(i);
        int16_t p_ = p_v(i);

        img_t[y_][x_].push_back(t_);
        img_p[y_][x_].push_back(p_);
    }

    return std::make_pair(img_t, img_p);
}

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
    m.def("cvtEventStrucure", &cvtEventStrucure, nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("t").noconvert(), nb::arg("p").noconvert(), nb::arg("width"), nb::arg("height"), nb::arg("t_min") = nb::none(), nb::arg("t_max") = nb::none());
}