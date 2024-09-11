#pragma once
#include <vector>

template <typename T>
class Array2d
{
private:
    size_t _shape[2];

public:
    std::vector<T> _vector;

    Array2d(size_t shape0, size_t shape1) : _shape{shape0, shape1}
    {
        _vector.resize(shape0 * shape1);
    }

    inline T &operator()(size_t i, size_t j)
    {
        return _vector[i * _shape[1] + j];
    }

    inline const T &operator()(size_t i, size_t j) const
    {
        return _vector[i * _shape[1] + j];
    }

    size_t size() const
    {
        return _vector.size();
    }

    size_t shape(int i) const
    {
        return _shape[i];
    }

    T *data()
    {
        return _vector.data();
    }

    const T *data() const
    {
        return _vector.data();
    }
};

template <typename T>
class Array3d
{
private:
    size_t _shape12;
    size_t _shape[3];

public:
    std::vector<T> _vector;

    Array3d(size_t shape0, size_t shape1, size_t shape2) : _shape{shape0, shape1, shape2}, _shape12(shape1 * shape2)
    {
        _vector.resize(shape0 * shape1 * shape2);
    }

    inline T &operator()(size_t i, size_t j, size_t k)
    {
        return _vector[i * _shape12 + j * _shape[2] + k];
    }

    inline const T &operator()(size_t i, size_t j, size_t k) const
    {
        return _vector[i * _shape12 + j * _shape[2] + k];
    }

    size_t size() const
    {
        return _vector.size();
    }

    size_t shape(int i) const
    {
        return _shape[i];
    }

    T *data()
    {
        return _vector.data();
    }

    const T *data() const
    {
        return _vector.data();
    }
};

template <typename T>
class Array4d
{
private:
    size_t _shape123;
    size_t _shape12;
    size_t _shape[4];

public:
    std::vector<T> _vector;

    Array4d(size_t shape0, size_t shape1, size_t shape2, size_t shape3) : _shape{shape0, shape1, shape2, shape3}, _shape12(shape1 * shape2), _shape123(shape0 * shape1 * shape2)
    {
        _vector.resize(shape0 * shape1 * shape2 * shape3);
    }

    T &operator()(size_t i, size_t j, size_t k, size_t l)
    {
        return _vector[i * _shape123 + j * _shape12 + k * _shape[3] + l];
    }

    const T &operator()(size_t i, size_t j, size_t k, size_t l) const
    {
        return _vector[i * _shape123 + j * _shape12 + k * _shape[3] + l];
    }

    size_t size() const
    {
        return _vector.size();
    }

    size_t shape(int i) const
    {
        return _shape[i];
    }

    T *data()
    {
        return _vector.data();
    }

    const T *data() const
    {
        return _vector.data();
    }
};