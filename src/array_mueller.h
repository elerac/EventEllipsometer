#pragma once
#include <Eigen/Core>
#include "array.h"

class VideoMueller : public Array4d<float>
{
public:
    VideoMueller(size_t num, size_t height, size_t width) : Array4d<float>(num, height, width, 16) {}

    Eigen::Vector<float, 16> operator()(size_t i, size_t j, size_t k) const
    {
        return Eigen::Map<const Eigen::Vector<float, 16>>(this->data() + (i * this->shape(1) * this->shape(2) + j * this->shape(2) + k) * 16);
    }

    Eigen::Vector<float, 16> &operator()(size_t i, size_t j, size_t k)
    {
        return *reinterpret_cast<Eigen::Vector<float, 16> *>(this->data() + (i * this->shape(1) * this->shape(2) + j * this->shape(2) + k) * 16);
    };
};