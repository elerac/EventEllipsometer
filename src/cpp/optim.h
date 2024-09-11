#pragma once
#include <iostream>
#include <chrono>
#include <utility>
#include <random>

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "equations.h"

Eigen::Vector<float, 16> svdSolve(const Eigen::Matrix<float, Eigen::Dynamic, 16> &A);

Eigen::VectorXf diffLn(const Eigen::Vector<float, 16> &m, const Eigen::VectorXf &thetaVector, float phi1, float phi2);

Eigen::Vector<float, 16> filterMueller(const Eigen::Vector<float, 16> &m);