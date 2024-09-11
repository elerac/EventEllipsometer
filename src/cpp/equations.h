#pragma once
#include <utility>
#include <Eigen/Core>

std::pair<Eigen::Matrix<float, Eigen::Dynamic, 16>, Eigen::Matrix<float, Eigen::Dynamic, 16>>
calcNumenatorDenominatorCoffs(const Eigen::VectorXf &thetaVector, float phi1, float phi2);
