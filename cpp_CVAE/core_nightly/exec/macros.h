#pragma once
#include <iostream>
#include <cassert>
//Safety compile-time Asserts that stay enabled
#define STATIC_ASSERT(cond, msg) static_assert(cond, msg)

// Safety run-time Asserts that stay enabled
#define ASSERT(x) \
    do { \
        if (!(x)) { \
            std::cerr << "[ASSERT FAIL] " << #x << " @ " << __FILE__ << " : " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

// Debug-only asserts (enabled with -DEBUG_MODE)
#ifdef DEBUG_MODE
    #define DASSERT(x) ASSERT(x)
    #define DSTATIC_ASSERT(cond, msg) STATIC_ASSERT(cond, msg)
#else
    #define DASSERT(x) ((void)0)
    #define DSTATIC_ASSERT(cond, msg) ((VOID)0)
#endif

// Verbose print logging
#if !defined(VERBOSE_MODE)
    #define VERBOSE_MODE -1
#endif

#if VERBOSE_MODE >= 0
    #define VERBOSEL0(msg) \
        do { std::cout << "[VERBOSE-0] " << msg << std::endl; } while (0)
#else
    #define VERBOSEL0(msg) ((void)0)
#endif

#if VERBOSE_MODE >= 1
    #define VERBOSEL1(msg) \
        do { std::cout << "[VERBOSE-1] " << msg << std::endl; } while (0)
#else
    #define VERBOSEL1(msg) ((void)0)
#endif

#if VERBOSE_MODE >= 2
    #define VERBOSEL2(msg) \
        do { std::cout << "[VERBOSE-2] " << msg << std::endl; } while (0)
#else
    #define VERBOSEL2(msg) ((void)0)
#endif

// // Test mode (used in test binaries with -TEST_MODE)
// #ifdef TEST_MODE
//     #define REGISTER_TEST(fn) fn();
// #else
//     #define REGISTER_TEST(fn) ((void)0)
// #endif
