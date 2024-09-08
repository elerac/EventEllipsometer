#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <Eigen/Core>

class SpatialEventMap
{
    // Event data format that allows accessing the event data `(t, p)` by specifying the spatial coordinates `(x, y)`.

private:
    int width;
    int height;

public:
    std::vector<std::vector<Eigen::Vector<int64_t, Eigen::Dynamic>>> map_t;
    std::vector<std::vector<Eigen::Vector<int16_t, Eigen::Dynamic>>> map_p;

    SpatialEventMap(const Eigen::Vector<uint16_t, Eigen::Dynamic> &x,
                    const Eigen::Vector<uint16_t, Eigen::Dynamic> &y,
                    const Eigen::Vector<int64_t, Eigen::Dynamic> &t,
                    const Eigen::Vector<int16_t, Eigen::Dynamic> &p,
                    int width,
                    int height) : width(width), height(height)
    {
        // Check size should be same
        if (x.size() != y.size() || x.size() != t.size() || x.size() != p.size())
        {
            throw std::invalid_argument("x, y, t, p should have same size");
        }

        size_t num = x.size();

        // Initialize the width and height
        std::vector<std::vector<std::vector<int64_t>>> _map_t(height, std::vector<std::vector<int64_t>>(width));
        std::vector<std::vector<std::vector<int16_t>>> _map_p(height, std::vector<std::vector<int16_t>>(width));
        _map_t.resize(height);
        _map_p.resize(height);
        for (size_t i = 0; i < height; ++i)
        {
            _map_t[i].resize(width);
            _map_p[i].resize(width);
        }

        // Map the event stream to the spatial map
        for (size_t i = 0; i < num; ++i)
        {
            uint16_t _x = x(i);
            uint16_t _y = y(i);
            _map_t[_y][_x].push_back(t(i));
            _map_p[_y][_x].push_back(p(i));
        }

        // Convert std::vector to Eigen::Vector
        map_t.resize(height);
        map_p.resize(height);
        for (size_t iy = 0; iy < height; ++iy)
        {
            map_t[iy].resize(width);
            map_p[iy].resize(width);

            for (size_t ix = 0; ix < width; ++ix)
            {
                map_t[iy][ix] = Eigen::Map<Eigen::Vector<int64_t, Eigen::Dynamic>>(_map_t[iy][ix].data(), _map_t[iy][ix].size());
                _map_t[iy][ix].clear();
                map_p[iy][ix] = Eigen::Map<Eigen::Vector<int16_t, Eigen::Dynamic>>(_map_p[iy][ix].data(), _map_p[iy][ix].size());
                _map_p[iy][ix].clear();
            }
        }
    }

    auto get(int x, int y) const
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            std::string msg = "x, y" + std::to_string(x) + " " + std::to_string(y) + " should be in the range [0, " + std::to_string(width) + "), [0, " + std::to_string(height) + ")";
            throw std::invalid_argument(msg);
        }
        return std::make_pair(std::ref(map_t[y][x]), std::ref(map_p[y][x]));
    }

    auto get(int x, int y, int t_min, int t_max) const
    {
        auto [t, p] = get(x, y);

        int left = std::lower_bound(t.data(), t.data() + t.size(), t_min) - t.data();
        int right = std::upper_bound(t.data() + left, t.data() + t.size(), t_max) - t.data();

        int n = right - left;
        auto t_sub = Eigen::Vector<int64_t, Eigen::Dynamic>(t.segment(left, n));
        auto p_sub = Eigen::Vector<int16_t, Eigen::Dynamic>(p.segment(left, n));

        return std::make_pair(t_sub, p_sub);
    }

    auto operator()(int x, int y) const
    {
        return get(x, y);
    }

    auto operator()(int x, int y, int t_min, int t_max) const
    {
        return get(x, y, t_min, t_max);
    }
};

class SpatialEventMapForEventEllipsometry : public SpatialEventMap
{
    // Event data format that allows accessing the event data `(theta, time_diff)` by specifying the spatial coordinates `(x, y)`.
    // This class is derived from `SpatialEventMap` and provides additional functionalities for Event Ellipsometry.

public:
    // First, initialize the base class `SpatialEventMap` with the given event data.
    // and calciulate the `theta` and `time_diff` from the event data.

    std::vector<std::vector<Eigen::VectorXf>> map_theta;
    std::vector<std::vector<Eigen::VectorXf>> map_time_diff;

    SpatialEventMapForEventEllipsometry(const Eigen::Vector<uint16_t, Eigen::Dynamic> &x,
                                        const Eigen::Vector<uint16_t, Eigen::Dynamic> &y,
                                        const Eigen::Vector<int64_t, Eigen::Dynamic> &t,
                                        const Eigen::Vector<int16_t, Eigen::Dynamic> &p,
                                        int width,
                                        int height) : SpatialEventMap(x, y, t, p, width, height)
    {
        // Initialize the width and height
        map_theta.resize(height);
        map_time_diff.resize(height);
        for (size_t iy = 0; iy < height; ++iy)
        {
            map_theta[iy].resize(width);
            map_time_diff[iy].resize(width);
        }

        // Calculate theta and time_diff from the event data
        for (size_t iy = 0; iy < height; ++iy)
        {
            for (size_t ix = 0; ix < width; ++ix)
            {
                auto [t, p] = get(ix, iy);
            }
        }
    }
};