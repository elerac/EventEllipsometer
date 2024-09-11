#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <Eigen/Core>
#include "array.h"

#define _USE_MATH_DEFINES
#include <math.h>

template <typename T>
auto searchsorted(const Eigen::VectorX<T> &a, T v_min, T v_max)
{
    // Find the upper limit indicies of `v_min` and `v_max` in the sorted array `a`.
    // The returned indices `i_min` and `i_max` satisfy `a[i_min-1] < v_min <= a[i_min]` and `a[i_max-1] < v_max <= a[i_max]`.
    //
    // By using the upper limit, the range `[i_min, i_max)` can be obtained by `a.segment(i_min, i_max - i_min)`.
    //
    // This function is equivalent to `np.searchsorted(a, [v_min, v_max], side="right")` in numpy.

    if (v_min > v_max) [[unlikely]]
    {
        throw std::invalid_argument("v_min should be less than v_max. v_min=" + std::to_string(v_min) + ", v_max=" + std::to_string(v_max));
    }
    size_t i_min = std::upper_bound(a.data(), a.data() + a.size(), v_min) - a.data();
    size_t i_max = std::upper_bound(a.data() + i_min, a.data() + a.size(), v_max) - a.data();
    return std::make_pair(i_min, i_max);
}

class EventMap
{
    // Event data format that allows accessing the event data `(t, p)` by specifying the spatial coordinates `(x, y)`.
private:
    int width;
    int height;

public:
    Array2d<Eigen::VectorX<int64_t>> map_t;
    Array2d<Eigen::VectorX<int16_t>> map_p;

    EventMap(const Eigen::VectorX<uint16_t> &x,
             const Eigen::VectorX<uint16_t> &y,
             const Eigen::VectorX<int64_t> &t,
             const Eigen::VectorX<int16_t> &p,
             int width,
             int height) : width(width), height(height), map_t(height, width), map_p(height, width)
    {
        // Check the size of the input data
        if (x.size() != y.size() || x.size() != t.size() || x.size() != p.size())
        {
            throw std::invalid_argument("x, y, t, p should have same size");
        }

        // Check t is sorted
        if (!std::is_sorted(t.data(), t.data() + t.size()))
        {
            throw std::invalid_argument("events should be sorted by time t");
        }

        size_t num = x.size();

        // Initialize the width and height
        Array2d<std::vector<int64_t>> _map_t(height, width);
        Array2d<std::vector<int16_t>> _map_p(height, width);

        // Append the (t, p) to the corresponding spatial coordinates
        for (size_t i = 0; i < num; ++i)
        {
            uint16_t _x = x(i);
            uint16_t _y = y(i);
            _map_t(_y, _x).push_back(t(i));
            _map_p(_y, _x).push_back(p(i));
        }

        // Convert std::vector to Eigen::Vector
        for (size_t iy = 0; iy < height; ++iy)
        {
            for (size_t ix = 0; ix < width; ++ix)
            {
                map_t(iy, ix) = Eigen::Map<Eigen::VectorX<int64_t>>(_map_t(iy, ix).data(), _map_t(iy, ix).size());
                _map_t(iy, ix).clear();
                map_p(iy, ix) = Eigen::Map<Eigen::VectorX<int16_t>>(_map_p(iy, ix).data(), _map_p(iy, ix).size());
                _map_p(iy, ix).clear();
            }
        }
    }

    EventMap(const Eigen::VectorX<uint16_t> &x,
             const Eigen::VectorX<uint16_t> &y,
             const Eigen::VectorX<int64_t> &t,
             const Eigen::VectorX<int16_t> &p,
             int width,
             int height,
             int64_t t_min,
             int64_t t_max) : EventMap(segment_by_time<uint16_t>(x, t, t_min, t_max),
                                       segment_by_time<uint16_t>(y, t, t_min, t_max),
                                       segment_by_time<int64_t>(t, t, t_min, t_max),
                                       segment_by_time<int16_t>(p, t, t_min, t_max),
                                       width,
                                       height) {};

    auto get(size_t x, size_t y) const
    {
        if (x >= width || y >= height) [[unlikely]]
        {
            std::string msg = "x, y" + std::to_string(x) + " " + std::to_string(y) + " should be in the range [0, " + std::to_string(width) + "), [0, " + std::to_string(height) + ")";
            throw std::invalid_argument(msg);
        }
        return std::make_pair(std::ref(map_t(y, x)), std::ref(map_p(y, x)));
    }

    auto get(size_t x, size_t y, int64_t t_min, int64_t t_max) const
    {
        auto [t, p] = this->get(x, y);
        auto [index_min, index_max] = searchsorted<int64_t>(t, t_min, t_max);
        auto t_sub = Eigen::VectorX<int64_t>(t.segment(index_min, index_max - index_min));
        auto p_sub = Eigen::VectorX<int16_t>(p.segment(index_min, index_max - index_min));
        return std::make_pair(t_sub, p_sub);
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

private:
    template <typename T>
    static auto segment_by_time(const Eigen::VectorX<T> &v, const Eigen::VectorX<int64_t> &t, int64_t t_min, int64_t t_max)
    {
        if (!std::is_sorted(t.data(), t.data() + t.size()))
        {
            throw std::invalid_argument("events should be sorted by time t");
        }
        auto [index_min, index_max] = searchsorted<int64_t>(t, t_min, t_max);
        return v.segment(index_min, index_max - index_min);
    }
};
