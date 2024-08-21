#pragma once
#include <iostream>
#include <chrono>
#include <utility>
#include <random>

#include <omp.h>
#include <Eigen/SVD>

#include "equations.h"

Eigen::VectorXf svdSolve(const Eigen::MatrixXf &A);

Eigen::VectorXf diffLn(const Eigen::Vector<float, 16> &M, const Eigen::VectorXf &thetaVector, float phi1, float phi2);