#pragma once
#include <Eigen/Core>
#include "utils.h"

// https://www.mathworks.com/help/stats/robustfit.html

class HuberLoss
{
public:
    float delta;

private:
    Eigen::VectorXf r_abs;
    Eigen::ArrayX<bool> r_abs_le_1;

public:
    HuberLoss(float delta = 1.35) : delta(delta) {}

    void compute(const Eigen::VectorXf &input, const Eigen::VectorXf &target)
    {
        Eigen::VectorXf resid = input - target;
        float sigma = mad(resid) / 0.6745f;
        r_abs = resid.array().abs() / (delta * sigma);
        r_abs_le_1 = r_abs.array() <= 1;
    }

    float loss() const
    {
        return (r_abs_le_1).select(0.5 * r_abs.array().square(), r_abs.array() - 0.5).mean();
    }

    Eigen::VectorXf weight() const
    {
        return (r_abs_le_1).select(1.0f, 1.0f / r_abs.array());
    }

    float operator()(const Eigen::VectorXf &input, const Eigen::VectorXf &target)
    {
        compute(input, target);
        return loss();
    }
};