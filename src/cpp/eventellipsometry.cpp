#include <iostream>
#include <chrono>
#include <utility>
#include <random>
#include <optional>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>

#include <omp.h>

#include "utils.h"
#include "equations.h"
#include "optim.h"
#include "eventmap.h"
#include "array.h"
#include "dataframe.h"
#include "mueller.h"
#include "loss.h"

namespace nb = nanobind;
using namespace nb::literals;

template <bool DEBUG = false>
Eigen::Vector<float, 16> fit_mueller_svd(const Eigen::VectorXf &theta,
                                         const Eigen::VectorXf &dlogI,
                                         float phi1,
                                         float phi2,
                                         float delta = 1.35,
                                         int max_iter = 10,
                                         float tol = 1e-2,
                                         const std::optional<Eigen::VectorXf> &weights = std::nullopt)
{
    // Count number of data points
    size_t num = theta.size();

    // Return nan if there are not enough data points
    if (num < 15)
    {
        return Eigen::Vector<float, 16>::Constant(16, std::numeric_limits<float>::quiet_NaN());
    }

    auto [pn, pd] = calculate_ndcoffs(theta, phi1, phi2);

    // Construct matrix A for SVD
    Eigen::Matrix<float, Eigen::Dynamic, 16> A(num, 16);
    A = (pn.array() - dlogI.replicate(1, 16).array() * pd.array());

    // Apply weights for A
    if (weights.has_value())
    {
        for (int i = 0; i < 16; ++i)
        {
            A.col(i) = A.col(i).array() * weights.value().array();
        }
    }

    // Loss function
    // HuberLoss loss_func(delta);
    L1Loss loss_func;

    // Solve initial guess
    Eigen::Vector<float, 16> x;
    x = svdSolve(A);
    x = filter_mueller(x); // x should be physically realizable Mueller matrix

    // Evaluate initial prediction
    Eigen::VectorXf dlogI_pred = (pn * x).array() / (pd * x).array();
    float error = loss_func(dlogI_pred, dlogI);

    // IRLS loop
    Eigen::Matrix<float, Eigen::Dynamic, 16> Aw(num, 16);
    for (int i_loop = 0; i_loop < max_iter; ++i_loop)
    {
        // Huber weight with previous residuals
        Eigen::VectorXf w = loss_func.weights();

        // Update A
        for (int i = 0; i < 16; ++i)
        {
            Aw.col(i) = A.col(i).array() * w.array();
        }

        // Solve
        x = svdSolve(Aw);
        x = filter_mueller(x);

        // Evaluate prediction
        dlogI_pred = (pn * x).array() / (pd * x).array();
        float error_new = loss_func(dlogI_pred, dlogI);

        // Check convergence
        if (std::abs(error - error_new) < tol)
        {
            error = error_new;
            break;
        }
        error = error_new;

        if constexpr (DEBUG)
        {
            std::cout << "iter: " << i_loop << "/" << max_iter << ", error: " << error << std::endl;
            auto np = nb::module_::import_("numpy");
            auto plt = nb::module_::import_("matplotlib.pyplot");
            auto theta_np = np.attr("array")(theta);
            auto dlogI_np = np.attr("array")(dlogI);
            auto dlogI_pred_np = np.attr("array")(dlogI_pred);
            auto w_np = np.attr("array")(w / (1e-8 + w.maxCoeff()));
            plt.attr("clf")();
            plt.attr("plot")(theta_np, dlogI_pred_np);
            plt.attr("scatter")(theta_np, dlogI_np, "label"_a = "data");
            plt.attr("scatter")(theta_np, dlogI_pred_np, "label"_a = "fit", "alpha"_a = w_np);
            plt.attr("legend")();
            // plt.attr("pause")(1);

            Eigen::Vector<float, 16> x_ = {1, 0, 0, 0,
                                           0, 0.99, 0, 0,
                                           0, 0, -0.99, 0,
                                           0, 0, 0, -0.99};
            Eigen::VectorXf dlogI_pred_gt = (pn * x_).array() / (pd * x_).array();
            Eigen::VectorXf r_gt = dlogI - dlogI_pred_gt;
            float error_gt = r_gt.array().abs().mean();

            // std::cout << "error_gt: " << error_gt << std::endl;
            auto dlogI_pred_gt_np = np.attr("array")(dlogI_pred_gt);
            plt.attr("plot")(theta_np, dlogI_pred_gt_np, "label"_a = "gt", "linestyle"_a = "--", "color"_a = "tab:green");

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

    return x;
}

auto fit_mueller(const std::vector<EventEllipsometryDataFrame> &dataframes, bool verbose = false)
{
    std::chrono::system_clock::time_point start, end;

    size_t num_frames = dataframes.size();
    size_t height = dataframes[0].shape(0);
    size_t width = dataframes[0].shape(1);
    VideoMueller video_mueller(num_frames, height, width);

    if (verbose)
    {
        std::cout << "Fitting Mueller matrix video..." << std::endl;
        std::cout << "(" << num_frames << ", " << height << ", " << width << ", 4, 4)" << std::endl;
    }

    // --------------------------------------------------------------------------------------------
    // Per-pixel Reconstruction
    // --------------------------------------------------------------------------------------------

    if (verbose)
    {
        std::cout << "Per-pixel Reconstruction..." << std::endl;
        //     std::cout << "  " << "phi_calib1: " << phi_calib1 << std::endl;
        //     std::cout << "  " << "phi_calib2: " << phi_calib2 << std::endl;
        //     std::cout << "  " << "delta: " << delta << std::endl;
        //     std::cout << "  " << "max_iter_svd: " << max_iter_svd << std::endl;
        //     std::cout << "  " << "tol: " << tol << std::endl;
    }

    start = std::chrono::system_clock::now();
    for (int iz = 0; iz < num_frames; ++iz)
    {
        auto &dataframe = dataframes[iz];
#pragma omp parallel for schedule(dynamic)
        for (int iy = 0; iy < height; ++iy)
        {
            for (int ix = 0; ix < width; ++ix)
            {
                auto [theta, dlogI, weights, phi_offset] = dataframe.get(ix, iy);
                Eigen::Vector<float, 16> m = fit_mueller_svd(theta, dlogI, 1.68, 2.66 - 5 * phi_offset, 1.35, 10, 1e-2, weights);
                video_mueller(iz, iy, ix) = m;
            }
        }
    }
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose)
    {
        std::cout << "Elapsed (per-pixel): " << elapsed * 0.001 << " s" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    // --------------------------------------------------------------------------------------------
    // Propagation and Refinement
    // --------------------------------------------------------------------------------------------
    if (verbose)
    {
        std::cout << "Propagation and Refinement..." << std::endl;
        // std::cout << "  " << "max_iter_propagate: " << max_iter_propagate << std::endl;
    }

    start = std::chrono::system_clock::now();

    // HuberLoss loss_func_huber(1.35);
    for (int iter = 0; iter < 10; ++iter)
    {

        for (int i_red_black = 0; i_red_black < 2; ++i_red_black) // Red: 0, Black: 1
        {

            for (int iz = 0; iz < num_frames; ++iz)
            {
                auto &dataframe = dataframes[iz];
#pragma omp parallel for schedule(dynamic)
                for (int iy = 0; iy < height; ++iy)
                {
                    for (int ix = 0; ix < width; ++ix)
                    {
                        if ((ix + iy) % 2 != i_red_black)
                        {
                            continue;
                        }

                        Eigen::Vector<float, 16> m_best;
                        float loss_best;

                        // Get MM of the target pixel
                        m_best = video_mueller(iz, iy, ix);

                        // Skip if NaN
                        if (m_best.array().isNaN().any())
                        {
                            continue;
                        }

                        // Define loss function for target pixel
                        auto [theta, dlogI, weight, phi_offset] = dataframe.get(ix, iy);
                        auto [pn, pd] = calculate_ndcoffs(theta, 1.68, 2.66 - 5 * phi_offset);
                        auto loss_func = [&pn, &pd, &dlogI, &weight](const Eigen::Vector<float, 16> &m)
                        {
                            Eigen::VectorXf dlogI_pred = (pn * m).array() / (pd * m).array();
                            Eigen::VectorXf r = dlogI - dlogI_pred;
                            return r.array().abs().mean();
                            // return (r.array().abs() * weight.array()).mean();
                        };

                        // Initialize the best loss (baseline)
                        loss_best = loss_func(m_best);

                        // Update the Mueller matrix via propagation
                        // std::vector<std::pair<int, int>> neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-5, 0}, {5, 0}, {0, -5}, {0, 5}};

                        std::vector<std::tuple<int, int, int>> neighbors = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {-5, 0, 0}, {5, 0, 0}, {0, -5, 0}, {0, 5, 0}, {0, 0, -1}, {0, 0, 1}};
                        for (auto [dx, dy, dz] : neighbors)
                        {
                            int iy_ = iy + dy;
                            int ix_ = ix + dx;
                            int iz_ = iz + dz;
                            if (iy_ < 0 || iy_ >= height || ix_ < 0 || ix_ >= width || iz_ < 0 || iz_ >= num_frames)
                            {
                                continue;
                            }

                            Eigen::Vector<float, 16> m = video_mueller(iz_, iy_, ix_);
                            if (m.array().isNaN().any())
                            {
                                continue;
                            }

                            float loss = loss_func(m);
                            if (loss < loss_best)
                            {
                                m_best = m;
                                loss_best = loss;
                            }
                        }

                        // Refine the Mueller matrix via random perturbation
                        Eigen::Vector<float, 16> m_perturbed = filter_mueller(perturb_mueller(m_best));
                        float loss = loss_func(m_perturbed);
                        if (loss < loss_best)
                        {
                            m_best = m_perturbed;
                        }

                        // Update the Mueller matrix
                        video_mueller(iz, iy, ix) = m_best;
                    }
                }
            }
        }
    }

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (verbose)
    {
        std::cout << "Elapsed (propagate): " << elapsed * 0.001 << " s" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    // --------------------------------------------------------------------------------------------

    // Convert to numpy array
    float *data = new float[video_mueller._vector.size()];
    std::move(video_mueller._vector.begin(), video_mueller._vector.end(), data);
    nb::capsule owner(data, [](void *p) noexcept
                      { delete[] (float *)p; });
    return nb::ndarray<nb::numpy, float, nb::shape<-1, -1, -1, 4, 4>, nb::c_contig>(data, {num_frames, height, width, 4, 4}, owner);
}

NB_MODULE(_eventellipsometry_impl, m)
{
    m.def("searchsorted", [](const nb::DRef<Eigen::VectorX<float>> &a, float v_min, float v_max)
          { return searchsorted<float>(a, v_min, v_max); }, nb::arg("a").noconvert(), nb::arg("v_min"), nb::arg("v_max"));

    m.def("svdSolve", [](const nb::DRef<Eigen::Matrix<float, Eigen::Dynamic, 16>> &A)
          { return svdSolve<float, 16>(A); }, nb::arg("A").noconvert(), "Solve Ax = 0\n\nParameters\n----------\nA : numpy.ndarray\n    Matrix A. (n, m)\n\nReturns\n-------\nx : numpy.ndarray\n    Solution x. (m,). The x is normalized by first element.");
    // m.def("fit_mueller_svd", [](const nb::DRef<Eigen::VectorXf> &theta, const nb::DRef<Eigen::VectorXf> &dlogI, float phi1, float phi2, float delta = 1.35, int max_iter = 10, float tol = 1e-2, bool debug = false)
    //       {
    //           if (debug)
    //               return fit_mueller_svd<true>(theta, dlogI, phi1, phi2, delta, max_iter, tol);
    //           else
    //               return fit_mueller_svd<false>(theta, dlogI, phi1, phi2, delta, max_iter, tol);
    //           //
    //       },
    //       nb::arg("theta").noconvert(), nb::arg("dlogI").noconvert(), nb::arg("phi1"), nb::arg("phi2"), nb::arg("delta") = 1.35, nb::arg("max_iter") = 10, nb::arg("tol") = 1e-2, nb::arg("debug") = false, "Fit the data");

    m.def("fit_mueller", &fit_mueller, nb::arg("dataframes").noconvert(), nb::arg("verbose") = false, nb::rv_policy::reference);

    m.def("calculate_ndcoffs", [](const nb::DRef<Eigen::VectorXf> &theta, float phi1, float phi2)
          { return calculate_ndcoffs(theta, phi1, phi2); }, nb::arg("theta").noconvert(), nb::arg("phi1"), nb::arg("phi2"), "Calculate numenator and denominator cofficients");
    m.def("median", [](const nb::DRef<Eigen::VectorXf> &v)
          { return median(v); }, nb::arg("v").noconvert(), "Calculate median");
    m.def("filter_mueller", [](const nb::DRef<Eigen::Vector<float, 16>> &m)
          { return filter_mueller(m); }, nb::arg("m").noconvert(), "Apply filter to acquire physically realizable Mueller matrix.\n\nThis method is based on Shane R. Cloude, \"Conditions For The Physical Realisability Of Matrix Operators In Polarimetry\", Proc. SPIE 1166, 1990.\n\nParameters\n----------\nm : numpy.ndarray\n    Mueller matrix. (16,)\n\nReturns\n-------\nm_ : numpy.ndarray\n    Filtered Mueller matrix. (16,)");

    nb::class_<EventMap>(m, "EventMap")
        .def(nb::init<nb::DRef<Eigen::VectorX<uint16_t>>, nb::DRef<Eigen::VectorX<uint16_t>>, nb::DRef<Eigen::VectorX<int64_t>>, nb::DRef<Eigen::VectorX<int16_t>>, int, int>(),
             nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("t").noconvert(), nb::arg("p").noconvert(), nb::arg("width"), nb::arg("height"))
        .def(nb::init<nb::DRef<Eigen::VectorX<uint16_t>>, nb::DRef<Eigen::VectorX<uint16_t>>, nb::DRef<Eigen::VectorX<int64_t>>, nb::DRef<Eigen::VectorX<int16_t>>, int, int, int64_t, int64_t>(),
             nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("t").noconvert(), nb::arg("p").noconvert(), nb::arg("width"), nb::arg("height"), nb::arg("t_min"), nb::arg("t_max"))
        .def("get", nb::overload_cast<size_t, size_t>(&EventMap::get, nb::const_), nb::arg("x"), nb::arg("y"))
        .def("get", nb::overload_cast<size_t, size_t, int64_t, int64_t>(&EventMap::get, nb::const_), nb::arg("x"), nb::arg("y"), nb::arg("t_min"), nb::arg("t_max"));

    nb::class_<EventEllipsometryDataFrame>(m, "EventEllipsometryDataFrame")
        .def(nb::init<int, int>(), nb::arg("width"), nb::arg("height"))
        .def("set", &EventEllipsometryDataFrame::set, nb::arg("x"), nb::arg("y"), nb::arg("theta"), nb::arg("dlogI"), nb::arg("weight"), nb::arg("phi_offset"))
        .def("get", &EventEllipsometryDataFrame::get, nb::arg("x"), nb::arg("y"))
        .def("shape", &EventEllipsometryDataFrame::shape, nb::arg("i"));

    m.def("construct_dataframes", [](const nb::DRef<Eigen::VectorX<uint16_t>> &x, const nb::DRef<Eigen::VectorX<uint16_t>> &y, const nb::DRef<Eigen::VectorX<int64_t>> &t, const nb::DRef<Eigen::VectorX<int16_t>> &p, int width, int height, const nb::DRef<Eigen::VectorX<int64_t>> &trig_t, const nb::DRef<Eigen::VectorX<int16_t>> &trig_p, const nb::DRef<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> &img_C_on, const nb::DRef<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> &img_C_off)
          { return construct_dataframes(x, y, t, p, width, height, trig_t, trig_p, img_C_on, img_C_off); }, nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("t").noconvert(), nb::arg("p").noconvert(), nb::arg("width"), nb::arg("height"), nb::arg("trig_t").noconvert(), nb::arg("trig_p").noconvert(), nb::arg("img_C_on").noconvert(), nb::arg("img_C_off").noconvert());
    m.def("construct_dataframes", [](const nb::DRef<Eigen::VectorX<uint16_t>> &x, const nb::DRef<Eigen::VectorX<uint16_t>> &y, const nb::DRef<Eigen::VectorX<int64_t>> &t, const nb::DRef<Eigen::VectorX<int16_t>> &p, int width, int height, const nb::DRef<Eigen::VectorX<int64_t>> &trig_t, const nb::DRef<Eigen::VectorX<int16_t>> &trig_p, float C_on, float C_off)
          { return construct_dataframes(x, y, t, p, width, height, trig_t, trig_p, C_on, C_off); }, nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("t").noconvert(), nb::arg("p").noconvert(), nb::arg("width"), nb::arg("height"), nb::arg("trig_t").noconvert(), nb::arg("trig_p").noconvert(), nb::arg("C_on"), nb::arg("C_off"));
}