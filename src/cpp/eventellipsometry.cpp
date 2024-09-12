#include <iostream>
#include <chrono>
#include <utility>
#include <random>
#include <optional>
#include <complex>
#include <random>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/complex.h>

#include <omp.h>

#include "equations.h"
#include "optim.h"
#include "eventmap.h"
#include "array.h"
#include "array_mueller.h"
#include "dataframe.h"
#include "mueller.h"

namespace nb = nanobind;
using namespace nb::literals;

Eigen::Vector<float, 16> perturb_mueller(const Eigen::Vector<float, 16> &m, float sigma = 0.01)
{
    // Generate perturbed Mueller matrix via random perturbation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(1.0, sigma); // Centered at 1.0 with standard deviation sigma
    Eigen::Vector<float, 16> v;
    v(0) = m(0);
    for (int i = 1; i < 16; ++i)
    {
        v(i) = m(i) * dist(gen);
    }
    return v;
}

int add(int a, int b)
{
    return a + b;
}

float median(const Eigen::VectorXf &v)
{
    auto v_ = v;
    std::sort(v_.data(), v_.data() + v_.size());
    if (v_.size() % 2)
    {
        // odd
        return v_(v_.size() / 2);
    }
    else
    {
        // even
        size_t i = v_.size() / 2;
        return (v_(i - 1) + v_(i)) * 0.5;
    }
}

template <bool DEBUG = false>
Eigen::VectorXf fit(const Eigen::VectorXf &theta,
                    const Eigen::VectorXf &dlogI,
                    float phi1,
                    float phi2,
                    float delta = 1.35,
                    int max_iter = 50,
                    float tol = 1e-3)
{
    // Count number of data points
    size_t num = theta.size();

    // Return nan if there are not enough data points
    if (num < 15)
    {
        return Eigen::VectorXf::Constant(16, std::numeric_limits<float>::quiet_NaN());
    }

    auto [pn, pd] = calcNumenatorDenominatorCoffs(theta, phi1, phi2);

    Eigen::Matrix<float, Eigen::Dynamic, 16> A(num, 16);
    A = (pn.array() - dlogI.replicate(1, 16).array() * pd.array());

    Eigen::Matrix<float, Eigen::Dynamic, 16> Ainit = A;

    // Weight by dlogI
    Eigen::VectorXf inv_dlogI = 1.0f / dlogI.array().abs();
    for (int i = 0; i < 16; ++i)
    {
        A.col(i) = A.col(i).array() * inv_dlogI.array();
        // Ainit.col(i) = Ainit.col(i).array() * inv_dlogI.array();
    }

    // Initial guess
    Eigen::Vector<float, 16> x;
    x = svdSolve(A);

    // IRLS loop
    Eigen::Matrix<float, Eigen::Dynamic, 16> Aw(num, 16);
    float error_prev = std::numeric_limits<float>::max();
    Eigen::Vector<float, 16> x_best = x;
    for (int i_loop = 0; i_loop < max_iter; ++i_loop)
    {
        // x should be physically realizable Mueller matrix
        x = filter_mueller(x);

        // Check convergence
        Eigen::VectorXf dlogI_pred = (pn * x).array() / (pd * x).array();
        Eigen::VectorXf r = dlogI - dlogI_pred;
        auto r_abs = r.array().abs();
        // auto r_sq = r.array().square();
        float error = r_abs.mean();
        if (error < error_prev)
        {
            error_prev = error;
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

            std::cout << "error_gt: " << error_gt << std::endl;
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

auto fit_frames(const std::vector<EventEllipsometryDataFrame> &dataframes)
{
    size_t num_frames = dataframes.size();
    size_t height = dataframes[0].shape(0);
    size_t width = dataframes[0].shape(1);
    VideoMueller video_mueller(num_frames, height, width);

    std::chrono::system_clock::time_point start, end;

    start = std::chrono::system_clock::now();
    for (int iz = 0; iz < num_frames; ++iz)
    {
        auto &dataframe = dataframes[iz];
#pragma omp parallel for schedule(dynamic)
        for (int iy = 0; iy < height; ++iy)
        {
            for (int ix = 0; ix < width; ++ix)
            {
                auto [theta, dlogI, weight, phi_offset] = dataframe.get(ix, iy);
                Eigen::Vector<float, 16> m = fit(theta, dlogI, 1.68, 2.66 - 5 * phi_offset, 1.35, 3, 1e-2);
                video_mueller(iz, iy, ix) = m;
            }
        }
    }
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed (initial guess): " << elapsed << std::endl;

    start = std::chrono::system_clock::now();
    // propagate
    for (int iter = 0; iter < 30; ++iter)
    {
        for (int irb = 0; irb < 2; ++irb)
        {
            for (int iz = 0; iz < num_frames; ++iz)
            {
#pragma omp parallel for schedule(dynamic)
                for (int iy = 0; iy < height; ++iy)
                {
                    int iy_even = iy % 2;
                    int ix0 = irb - iy_even;
                    for (int ix = ix0; ix < width; ++ix)
                    {
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
                        auto [theta, dlogI, weight, phi_offset] = dataframes[iz].get(ix, iy);
                        auto [pn, pd] = calcNumenatorDenominatorCoffs(theta, 1.68, 2.66 - 5 * phi_offset);
                        auto loss_func = [&pn, &pd, &dlogI](const Eigen::Vector<float, 16> &m)
                        {
                            if (m.array().isNaN().any())
                            {
                                return std::numeric_limits<float>::max();
                            }
                            else
                            {
                                Eigen::VectorXf dlogI_pred = (pn * m).array() / (pd * m).array();
                                Eigen::VectorXf r = dlogI - dlogI_pred;
                                return r.array().abs().mean();
                            }
                        };

                        // Initialize the best loss (baseline)
                        loss_best = loss_func(m_best);

                        // Update the Mueller matrix via propagation
                        std::vector<std::pair<int, int>> neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-5, 0}, {5, 0}, {0, -5}, {0, 5}};
                        for (auto [dy, dx] : neighbors)
                        {
                            int iy_ = iy + dy;
                            int ix_ = ix + dx;
                            if (iy_ < 0 || iy_ >= height || ix_ < 0 || ix_ >= width)
                            {
                                continue;
                            }

                            Eigen::Vector<float, 16> m = video_mueller(iz, iy_, ix_);
                            float loss = loss_func(m);
                            if (loss < loss_best)
                            {
                                m_best = m;
                                loss_best = loss;
                            }
                        }

                        // Refine the 3D line map via random perturbation
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
    std::cout << "Elapsed (propagate): " << elapsed << std::endl;

    return video_mueller._vector;
}

NB_MODULE(_eventellipsometry_impl, m)
{
    m.def("searchsorted", [](const nb::DRef<Eigen::VectorX<float>> &a, float v_min, float v_max)
          { return searchsorted<float>(a, v_min, v_max); }, nb::arg("a").noconvert(), nb::arg("v_min"), nb::arg("v_max"));

    m.def("test", [](nb::ndarray<int, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu> array_py)
          {
              //
              size_t height = array_py.shape(0);
              size_t width = array_py.shape(1);

              Array3d<int> array(height, width, 3);

              for (size_t i = 0; i < height; ++i)
              {
                  for (size_t j = 0; j < width; ++j)
                  {
                      for (size_t k = 0; k < 3; ++k)
                      {
                          array(i, j, k) = array_py(i, j, k);
                      }
                  }
              }

              std::chrono::system_clock::time_point start, end;

              start = std::chrono::system_clock::now();
              for (int i = 0; i < height; ++i)
              {
                  for (size_t j = 0; j < width; ++j)
                  {
                      for (size_t k = 0; k < 3; ++k)
                      {
                          array(i, j, k) *= 2;
                      }
                  }
              }
              end = std::chrono::system_clock::now();
              double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
              std::cout << "Elapsed: " << elapsed << std::endl;

              start = std::chrono::system_clock::now();

              size_t width_3 = width * 3;
              size_t height_width_3 = height * width_3;
              int *ptr0 = array._vector.data();
              for (size_t i = 0; i < height_width_3; i += width_3)
              {
                  int *ptr01 = ptr0 + i;
                  for (size_t j = 0; j < width_3; j += 3)
                  {
                      // get pointer and convert to eigen vector
                      int *ptr = ptr01 + j;
                      Eigen::Map<Eigen::Vector3i> v(ptr);
                      v *= 2;
                  }
              }

              end = std::chrono::system_clock::now();
              elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
              std::cout << "Elapsed: " << elapsed << std::endl;

              int *data = new int[array._vector.size()];
              std::move(array._vector.begin(), array._vector.end(), data);
              nb::capsule owner(data, [](void *p) noexcept
                                { delete[] (int *)p; });
              return nb::ndarray<nb::numpy, int>(data, {height, width, 3}, owner);
              //
          });
    m.def("add", &add);
    m.def("svdSolve", [](const nb::DRef<Eigen::Matrix<float, Eigen::Dynamic, 16>> &A)
          { return svdSolve(A); }, nb::arg("A").noconvert(), "Solve Ax = 0\n\nParameters\n----------\nA : numpy.ndarray\n    Matrix A. (n, m)\n\nReturns\n-------\nx : numpy.ndarray\n    Solution x. (m,). The x is normalized by first element.");
    // m.def("fit_batch", &fit_batch, nb::arg("theta").noconvert(), nb::arg("dlogI").noconvert(), nb::arg("max_iter") = 50, nb::arg("eps") = 1e-6, nb::arg("tol") = 1e-3);
    m.def("fit", [](const nb::DRef<Eigen::VectorXf> &theta, const nb::DRef<Eigen::VectorXf> &dlogI, float phi1, float phi2, float delta = 1.35, int max_iter = 50, float tol = 1e-3, bool debug = false)
          {
              if (debug)
                  return fit<true>(theta, dlogI, phi1, phi2, delta, max_iter, tol);
              else
                  return fit<false>(theta, dlogI, phi1, phi2, delta, max_iter, tol);
              //
          },
          nb::arg("theta").noconvert(), nb::arg("dlogI").noconvert(), nb::arg("phi1"), nb::arg("phi2"), nb::arg("delta") = 1.35, nb::arg("max_iter") = 50, nb::arg("tol") = 1e-3, nb::arg("debug") = false, "Fit the data");

    m.def("fit_frames", &fit_frames, nb::arg("dataframes"), "Fit the data frames");

    m.def("calcNumenatorDenominatorCoffs", [](const nb::DRef<Eigen::VectorXf> &theta, float phi1, float phi2)
          { return calcNumenatorDenominatorCoffs(theta, phi1, phi2); }, nb::arg("theta").noconvert(), nb::arg("phi1"), nb::arg("phi2"), "Calculate numenator and denominator cofficients");
    m.def("median", [](const nb::DRef<Eigen::VectorXf> &v)
          { return median(v); }, nb::arg("v").noconvert(), "Calculate median");
    m.def("filter_mueller", [](const nb::DRef<Eigen::Vector<float, 16>> &m)
          { return filter_mueller(m); }, nb::arg("m").noconvert(), "Apply filter to acquire physically realizable Mueller matrix.\n\nThis method is based on Shane R. Cloude, \"Conditions For The Physical Realisability Of Matrix Operators In Polarimetry\", Proc. SPIE 1166, 1990.\n\nParameters\n----------\nm : numpy.ndarray\n    Mueller matrix. (16,)\n\nReturns\n-------\nm_ : numpy.ndarray\n    Filtered Mueller matrix. (16,)");
    // m.def("propagate", &propagate, nb::arg("video_mueller").noconvert(), "Propagate the Mueller matrix");

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