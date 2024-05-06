#pragma once
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <corecrt_math_defines.h>
#include <thrust/complex.h>

typedef thrust::complex<float> tcf;

//"mu" is the mean of the distribution, and "sigma" is the standard deviation.
tcf generateGaussianNoise(float mu0, float mu1, float sigma0, float sigma1)
{
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    constexpr float two_pi = M_PI * 2.0;

    //initialize the random uniform number generator (runif) in a range 0 to 1
    static std::mt19937 rng(std::random_device{}()); // Standard mersenne_twister_engine seeded with rd()
    static std::uniform_real_distribution<> runif(0.0, 1.0);

    //create two random numbers, make sure u1 is greater than epsilon
    float u1, u2;
    do
    {
        u1 = runif(rng);
    }
    while (u1 <= epsilon);
    u2 = runif(rng);

    //compute z0 and z1
    float mag0 = sigma0 * sqrt(-2.0 * log(u1));
    float mag1 = sigma1 * sqrt(-2.0 * log(u2));
    float z0  = mag0 * cos(two_pi * u2) + mu0;
    float z1  = mag1 * sin(two_pi * u2) + mu1;

    return tcf(z0, z1);
}