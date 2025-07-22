#pragma once

#include <iostream>
#include <cstdlib>

// Safety Asserts that stay enabled
#define ASSERT(x) \
    do { \
        if (!(x)) { \
            std::cerr << "[ASSERT FAIL] " << #x << " @ " << __FILE__ << " : " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

// Debug-only asserts (enabled with -DDEBUG_MODE)
#ifdef DEBUG_MODE
    #define DASSERT(x) ASSERT(x)
#else
    #define DASSERT(x) ((void)0)
#endif

// Verbose print logging (enabled with -DVERBOSE_MODE)
#ifdef VERBOSE_MODE
    #define VLOG(x) \
        do { \
            std::cout << "[VERBOSE] " << x << std::endl; \
        } while (0)
#else
    #define VLOG(x) ((void)0)
#endif

// Test mode (used in test binaries with -DTEST_MODE)
#ifdef TEST_MODE
    #define REGISTER_TEST(fn) fn();
#else
    #define REGISTER_TEST(fn) ((void)0)
#endif
