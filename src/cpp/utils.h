#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <Eigen/Core>

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

float median(const Eigen::VectorXf &v)
{
    // https://stackoverflow.com/a/34077478
    auto v_ = v;
    auto n = v_.size() / 2;
    std::nth_element(v_.begin(), v_.begin() + n, v_.end());
    auto med = v_[n];
    if (!(v_.size() & 1))
    { // If the set size is even
        auto max_it = std::max_element(v_.begin(), v_.begin() + n);
        med = (*max_it + med) / 2.0;
    }
    return med;
}

float mad(const Eigen::VectorXf &v)
{
    // Median Absolute Deviation (MAD)
    return median((v.array() - median(v)).abs());
}
