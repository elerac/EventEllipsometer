#pragma once
#include <iostream>
#include <chrono>
#include <utility>
#include <random>

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "equations.h"

Eigen::Vector<float, 16> svdSolve(const Eigen::Matrix<float, Eigen::Dynamic, 16> &A);