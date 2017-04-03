//
// Created by Jason on 02/04/2017.
//

#ifndef TENSORLAB_SUPPORT_ALGORITHM_H
#define TENSORLAB_SUPPORT_ALGORITHM_H

template <typename T>
inline T clip(const T& v, const T& min, const T& max)
{
    return v < min ? min : v > max ? max : v;
}

template <typename T>
inline T around(const T& v)
{
    return (T)((double)v + 0.5);
}

#endif //PROJECT_ALGORITHM_H
