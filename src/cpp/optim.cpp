#include <iostream>
#include <chrono>
#include <utility>
#include <random>

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "equations.h"

// Solve Ax = 0
Eigen::Vector<float, 16> svdSolve(const Eigen::Matrix<float, Eigen::Dynamic, 16> &A)
{
    // https://eigen.tuxfamily.org/dox/group__SVD__Module.html
    Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 16>> svd(A, Eigen::ComputeThinV);
    // Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto V = svd.matrixV();                           // Get V matrix
    Eigen::Vector<float, 16> x = V.col(V.cols() - 1); // Get last column
    float div_x0 = 1.0 / x(0);                        // Precompute 1/x0
    x *= div_x0;                                      // Normalize the vector
    return x;
}

Eigen::VectorXf diffLn(const Eigen::Vector<float, 16> &m, const Eigen::VectorXf &thetaVector, float phi1, float phi2)
{
    auto [numenator_coffs, denominator_coffs] = calcNumenatorDenominatorCoffs(thetaVector, phi1, phi2);
    Eigen::VectorXf time_diff = (numenator_coffs * m).array() / (denominator_coffs * m).array();
    return time_diff;
}
