#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <Eigen/Core>
#include "array.h"
#include "eventmap.h"

class EventEllipsometryDataFrame
{
    // Event data format for EventEllipsometry setup
public:
    int width;
    int height;
    Array2d<Eigen::VectorXf> map_theta;  // theta
    Array2d<Eigen::VectorXf> map_dlogI;  // Derivative of log intensity with respect to theta
    Array2d<Eigen::VectorXf> map_weight; // Weight vector
    float phi_offset;                    // Offset

    EventEllipsometryDataFrame(int width, int height) : width(width), height(height), map_theta(height, width), map_dlogI(height, width), map_weight(height, width)
    {
    }

    auto get(size_t x, size_t y) const
    {
        if (x >= width || y >= height) [[unlikely]]
        {
            std::string msg = "x, y" + std::to_string(x) + " " + std::to_string(y) + " should be in the range [0, " + std::to_string(width) + "), [0, " + std::to_string(height) + ")";
            throw std::invalid_argument(msg);
        }
        // return std::make_pair(std::ref(map_theta(y, x)), std::ref(map_dlogI(y, x)));
        return std::make_tuple(std::ref(map_theta(y, x)), std::ref(map_dlogI(y, x)), std::ref(map_weight(y, x)), phi_offset);
    }

    // set the value of theta and dlogI
    void set(size_t x, size_t y, const Eigen::VectorXf &theta, const Eigen::VectorXf &dlogI, const Eigen::VectorXf &weight, float phi_offset)
    {
        if (x >= width || y >= height) [[unlikely]]
        {
            std::string msg = "x, y" + std::to_string(x) + " " + std::to_string(y) + " should be in the range [0, " + std::to_string(width) + "), [0, " + std::to_string(height) + ")";
            throw std::invalid_argument(msg);
        }
        map_theta(y, x) = theta;
        map_dlogI(y, x) = dlogI;
        map_weight(y, x) = weight;
        this->phi_offset = phi_offset;
    }

    size_t shape(int i) const
    {
        if (i == 0)
        {
            return height;
        }
        else if (i == 1)
        {
            return width;
        }
        else
        {
            throw std::invalid_argument("i should be 0 or 1");
        }
    }
};

std::vector<EventEllipsometryDataFrame> construct_dataframes(const Eigen::VectorX<uint16_t> &x,
                                                             const Eigen::VectorX<uint16_t> &y,
                                                             const Eigen::VectorX<int64_t> &t,
                                                             const Eigen::VectorX<int16_t> &p,
                                                             int width,
                                                             int height,
                                                             const Eigen::VectorX<int64_t> &trig_t,
                                                             const Eigen::VectorX<int16_t> &trig_p,
                                                             const Eigen::MatrixX<float> &img_C_on,
                                                             const Eigen::MatrixX<float> &img_C_off)
{

    // Triggers
    std::vector<int64_t> trig_t_x1;
    std::vector<int64_t> trig_t_x5;
    for (size_t i = 0; i < trig_t.size(); ++i)
    {
        bool is_on = trig_p(i) > 0;
        if (is_on)
        {
            trig_t_x5.push_back(trig_t(i));
        }
        else
        {
            trig_t_x1.push_back(trig_t(i));
        }
    }

    // If the first trigger is a 5th, delete it.
    if ((trig_t_x5[0] - trig_t_x1[0]) < 0)
    {
        trig_t_x5.erase(trig_t_x5.begin());
    }

    // If the length of the triggers is not the same, delete the last one from the longer one.
    if (trig_t_x5.size() > trig_t_x1.size())
    {
        trig_t_x5.pop_back();
    }
    if (trig_t_x5.size() < trig_t_x1.size())
    {
        trig_t_x1.pop_back();
    }

    // Calculate the phi_offsets
    std::vector<float> phi_offsets(trig_t_x1.size() - 1);
    for (size_t i = 0; i < trig_t_x1.size() - 1; ++i)
    {
        float trig_t_diff = trig_t_x5[i] - trig_t_x1[i];
        float period = trig_t_x1[i + 1] - trig_t_x1[i];
        phi_offsets[i] = trig_t_diff / period * M_PI;
    }

    size_t num = trig_t_x1.size() - 1;
    std::vector<EventEllipsometryDataFrame> ellipsometry_eventmaps;
    ellipsometry_eventmaps.reserve(num);

    for (size_t it = 0; it < num; ++it)
    {
        int64_t t_min = trig_t_x1[it];
        int64_t t_max = trig_t_x1[it + 1];
        auto [index_min, index_max] = searchsorted<int64_t>(t, t_min, t_max);

        Eigen::VectorX<uint16_t> x_sub = x.segment(index_min, index_max - index_min);
        Eigen::VectorX<uint16_t> y_sub = y.segment(index_min, index_max - index_min);
        Eigen::VectorX<int64_t> t_sub = t.segment(index_min, index_max - index_min);
        Eigen::VectorX<int16_t> p_sub = p.segment(index_min, index_max - index_min);
        EventMap eventmap(x_sub, y_sub, t_sub, p_sub, width, height);

        // Convert (t, p) to (theta, dlogI)
        EventEllipsometryDataFrame ellipsometry_eventmap(width, height);
        for (size_t iy = 0; iy < height; ++iy)
        {
            for (size_t ix = 0; ix < width; ++ix)
            {
                auto [_t, _p] = eventmap.get(ix, iy);
                if (_t.size() <= 1)
                {
                    continue;
                }

                // time to theta (0 to pi)
                Eigen::VectorXf _theta = (_t.cast<float>() - Eigen::VectorXf::Constant(_t.size(), t_min).cast<float>()) / static_cast<float>(t_max - t_min) * M_PI;

                // theta_diff = np.convolve(theta, np.array([1, -1]), mode="valid")
                // theta = np.convolve(theta, np.array([0.5, 0.5]), mode="valid")
                std::vector<float> vec_theta;
                std::vector<float> vec_dlogI;
                std::vector<float> vec_weight;
                vec_theta.reserve(_theta.size() - 1);
                vec_dlogI.reserve(_theta.size() - 1);
                vec_weight.reserve(_theta.size() - 1);

                for (size_t i = 0; i < _theta.size() - 1; ++i)
                {
                    // dlogI
                    float theta_diff = _theta(i + 1) - _theta(i);
                    if (theta_diff == 0.0f)
                    {
                        continue;
                    }
                    bool is_on = _p(i + 1) > 0;
                    float C = is_on ? img_C_on(iy, ix) : img_C_off(iy, ix);
                    vec_dlogI.push_back(1.0f / theta_diff * C);

                    // theta
                    vec_theta.push_back((_theta(i) + _theta(i + 1)) * 0.5f);

                    // weight
                    vec_weight.push_back(1.0f / theta_diff);
                }

                Eigen::VectorXf theta = Eigen::Map<Eigen::VectorXf>(vec_theta.data(), vec_theta.size());
                Eigen::VectorXf dlogI = Eigen::Map<Eigen::VectorXf>(vec_dlogI.data(), vec_dlogI.size());
                Eigen::VectorXf weight = Eigen::Map<Eigen::VectorXf>(vec_weight.data(), vec_weight.size());
                ellipsometry_eventmap.set(ix, iy, theta, dlogI, weight, phi_offsets[it]);
            }
        }

        ellipsometry_eventmaps.push_back(ellipsometry_eventmap);
    }

    return ellipsometry_eventmaps;
}

std::vector<EventEllipsometryDataFrame> construct_dataframes(const Eigen::VectorX<uint16_t> &x,
                                                             const Eigen::VectorX<uint16_t> &y,
                                                             const Eigen::VectorX<int64_t> &t,
                                                             const Eigen::VectorX<int16_t> &p,
                                                             int width,
                                                             int height,
                                                             const Eigen::VectorX<int64_t> &trig_t,
                                                             const Eigen::VectorX<int16_t> &trig_p,
                                                             float C_on,
                                                             float C_off)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_C_on = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Constant(height, width, C_on);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_C_off = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Constant(height, width, C_off);
    return construct_dataframes(x, y, t, p, width, height, trig_t, trig_p, img_C_on, img_C_off);
}
