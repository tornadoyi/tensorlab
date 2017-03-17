//
// Created by Jason on 17/03/2017.
//

#ifndef TENSORLAB_BIND_PYTHON_UTILS_H
#define TENSORLAB_BIND_PYTHON_UTILS_H

#include <boost/python.hpp>

using namespace boost::python;

#define DECL_WRAP_CORE(cls, fname, wname) \
    auto wrap_##wname(cls& o) -> decltype(o.fname()){return o.fname();}

#define DECL_WRAP_CORE_1(cls, fname, wname, T1) \
    auto wrap_##wname(cls& o, T1 a) -> decltype(o.fname(a)){return o.fname(a);}

#define DECL_WRAP_CORE_2(cls, fname, wname, T1, T2) \
    auto wrap_##wname(cls& o, T1 a, T2 b) -> decltype(o.fname(a, b)){return o.fname(a, b);}



#define DECL_WRAP(cls, name) DECL_WRAP_CORE(cls, name, name)

#define DECL_WRAP_1(cls, name, T1) DECL_WRAP_CORE_1(cls, name, name, T1)

#define DECL_WRAP_2(cls, name, T1, T2) DECL_WRAP_CORE_2(cls, name, name, T1, T2)


#define REF_WRAP(name) &wrap_##name



#define DECL_GETER_SETER(cls, name) \
    auto wrap_get_##name(const cls& o) -> decltype(o.name()){ return o.name(); } \
    void wrap_set_##name(cls& o, decltype(o.name()) v){o.name() = v;}

#define REF_GETER_SETER(name) REF_WRAP(get_##name), REF_WRAP(set_##name)



#define DECL_WRAP_STR(cls) \
    string wrap_str(const cls& o) \
    { \
        std::ostringstream sout; \
        sout << o; \
        return sout.str(); \
    }

#define REF_WRAP_STR wrap_str

#endif //PROJECT_BIND_PYTHON_UTILS_H
