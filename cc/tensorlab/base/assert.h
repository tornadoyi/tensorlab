//
// Created by Jason on 16/03/2017.
//

#ifndef TENSORLAB_ASSERT_H
#define TENSORLAB_ASSERT_H


#include <sstream>
#include <iosfwd>
#include "error.h"



#define TLM_CASSERT(_exp,_message)                                              \
    {if ( !(_exp) )                                                             \
    {                                                                       \
        dlib_assert_breakpoint();                                           \
        std::ostringstream tl_o_out;                                       \
        tl_o_out << "\n\nError detected at line " << __LINE__ << ".\n";    \
        tl_o_out << "Error detected in file " << __FILE__ << ".\n";      \
        tl_o_out << "Error detected in function " << TL_FUNCTION_NAME << ".\n\n";      \
        tl_o_out << "Failing expression was " << #_exp << ".\n";           \
        tl_o_out << std::boolalpha << _message << "\n";                    \
        throw fatal_error(dlib::EBROKEN_ASSERT,tl_o_out.str());      \
    }}



// This macro is not needed if you have a real C++ compiler.  It's here to work around bugs in Visual Studio's preprocessor.
#define TL_WORKAROUND_VISUAL_STUDIO_BUGS(x) x
// Make it so the 2nd argument of TL_CASSERT is optional.  That is, you can call it like
// TL_CASSERT(exp) or TL_CASSERT(exp,message).
#define TLM_CASSERT_1_ARGS(exp)              TLM_CASSERT(exp,"")
#define TLM_CASSERT_2_ARGS(exp,message)      TLM_CASSERT(exp,message)
#define TLM_GET_3TH_ARG(arg1, arg2, arg3, ...) arg3
#define TLM_CASSERT_CHOOSER(...) TL_WORKAROUND_VISUAL_STUDIO_BUGS(TLM_GET_3TH_ARG(__VA_ARGS__,  TLM_CASSERT_2_ARGS, TLM_CASSERT_1_ARGS))
#define TL_CASSERT(...) TL_WORKAROUND_VISUAL_STUDIO_BUGS(TLM_CASSERT_CHOOSER(__VA_ARGS__)(__VA_ARGS__))



#ifdef ENABLE_ASSERTS
    #define TL_ASSERT(...) TL_CASSERT(__VA_ARGS__)
    #define TL_IF_ASSERT(exp) exp
#else
    #define TL_ASSERT(...) {}
    #define TL_IF_ASSERT(exp)
#endif


extern "C"
{
inline void dlib_assert_breakpoint(
) {}
/*!
    ensures
        - this function does nothing
          It exists just so you can put breakpoints on it in a debugging tool.
          It is called only when an TL_ASSERT or TL_CASSERT fails and is about to
          throw an exception.
!*/
}

#endif //PROJECT_ASSERT_H
