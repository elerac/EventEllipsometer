#pragma once
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std::complex_literals;

Eigen::Vector<float, 16> filter_mueller(const Eigen::Vector<float, 16> &m)
{
    Eigen::Matrix4cf H;
    H(0, 0) = std::complex<float>(0.25 * (m(0) + m(10) + m(15) + m(5)), 0);
    H(1, 0) = std::complex<float>(0.25 * (m(1) + m(4)), 0.25 * (m(11) - m(14)));
    H(2, 0) = std::complex<float>(0.25 * (m(2) + m(8)), -0.25 * (-m(13) + m(7)));
    H(3, 0) = std::complex<float>(0.25 * (m(12) + m(3)), 0.25 * (m(6) - m(9)));
    H(0, 1) = std::complex<float>(0.25 * (m(1) + m(4)), -0.25 * (m(11) - m(14)));
    H(1, 1) = std::complex<float>(0.25 * (m(0) - m(10) - m(15) + m(5)), 0);
    H(2, 1) = std::complex<float>(0.25 * (m(6) + m(9)), -0.25 * (-m(12) + m(3)));
    H(3, 1) = std::complex<float>(0.25 * (m(13) + m(7)), -0.25 * (m(2) - m(8)));
    H(0, 2) = std::complex<float>(0.25 * (m(2) + m(8)), 0.25 * (-m(13) + m(7)));
    H(1, 2) = std::complex<float>(0.25 * (m(6) + m(9)), 0.25 * (-m(12) + m(3)));
    H(2, 2) = std::complex<float>(0.25 * (m(0) + m(10) - m(15) - m(5)), 0);
    H(3, 2) = std::complex<float>(0.25 * (m(11) + m(14)), -0.25 * (m(1) - m(4)));
    H(0, 3) = std::complex<float>(0.25 * (m(12) + m(3)), -0.25 * (m(6) - m(9)));
    H(1, 3) = std::complex<float>(0.25 * (m(13) + m(7)), 0.25 * (m(2) - m(8)));
    H(2, 3) = std::complex<float>(0.25 * (m(11) + m(14)), 0.25 * (m(1) - m(4)));
    H(3, 3) = std::complex<float>(0.25 * (m(0) - m(10) + m(15) - m(5)), 0);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cf> es;
    es.compute(H);
    Eigen::Matrix4cf D = es.eigenvalues().cwiseMax(0).asDiagonal();
    Eigen::Matrix4cf H_ = es.eigenvectors() * D * es.eigenvectors().adjoint();

    Eigen::Vector<std::complex<float>, 16> h_ = H_.reshaped<Eigen::RowMajor>();

    Eigen::Vector<float, 16> m_;
    std::complex<float> I(0, 1);
    m_(0) = (h_(0) + h_(10) + h_(15) + h_(5)).real();
    m_(1) = (h_(1) - I * h_(11) + I * h_(14) + h_(4)).real();
    m_(2) = (I * h_(13) + h_(2) - I * h_(7) + h_(8)).real();
    m_(3) = (h_(12) + h_(3) - I * h_(6) + I * h_(9)).real();
    m_(4) = (h_(1) + I * h_(11) - I * h_(14) + h_(4)).real();
    m_(5) = (h_(0) - h_(10) - h_(15) + h_(5)).real();
    m_(6) = (-I * h_(12) + I * h_(3) + h_(6) + h_(9)).real();
    m_(7) = (h_(13) - I * h_(2) + h_(7) + I * h_(8)).real();
    m_(8) = (-I * h_(13) + h_(2) + I * h_(7) + h_(8)).real();
    m_(9) = (I * h_(12) - I * h_(3) + h_(6) + h_(9)).real();
    m_(10) = (h_(0) + h_(10) - h_(15) - h_(5)).real();
    m_(11) = (I * h_(1) + h_(11) + h_(14) - I * h_(4)).real();
    m_(12) = (h_(12) + h_(3) + I * h_(6) - I * h_(9)).real();
    m_(13) = (h_(13) + I * h_(2) + h_(7) - I * h_(8)).real();
    m_(14) = (-I * h_(1) + h_(11) + h_(14) + I * h_(4)).real();
    m_(15) = (h_(0) - h_(10) + h_(15) - h_(5)).real();

    return m_;
}

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