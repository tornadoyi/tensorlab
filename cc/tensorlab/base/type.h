//
// Created by Jason on 16/03/2017.
//

#ifndef TENSORLAB_TYPE_H
#define TENSORLAB_TYPE_H


#define DEF_ALL_TYPE_SUFFIX(Func) \
    Func(int,                  i) \
    Func(long,                 l) \
    Func(float,                f) \
    Func(double,               d) \
    Func(std::complex<float>,  cf) \
    Func(std::complex<double>, cd)

#endif //PROJECT_TYPE_H
