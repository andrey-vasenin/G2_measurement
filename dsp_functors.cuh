#ifndef DSP_FUNCTORS_CUH
#define DSP_FUNCTORS_CUH
#include "dsp.cuh"
#include <cuda_fp16.h>

struct calibration_functor : thrust::unary_function<half2, half2>
{
    half a_qi, a_qq;
    half c_i, c_q;

    calibration_functor(float _a_qi, float _a_qq,
        float _c_i, float _c_q) : a_qi{ __float2half(_a_qi) }, a_qq{ __float2half(_a_qq) },
        c_i{ __float2half(_c_i) }, c_q{ __float2half(_c_q) }
    {
    }

    __device__
        half2 operator()(const half2& x)
    {
        return make_half2(__low2half(x) + c_i,
            a_qi * __low2half(x) + a_qq * __high2half(x) + c_q);
    }
};

struct millivolts_functor : thrust::binary_function<char, char, half2>
{
    float scale;

    millivolts_functor(float s) : scale(s) {}

    __device__ half2 operator()(const char& i, const char& q)
    {
        return make_half2(__float2half(static_cast<float>(i) * scale),
            __float2half(static_cast<float>(q) * scale));
    }
};



struct add_half2_to_tcf_functor : thrust::binary_function<half2, tcf, tcf>
{
    __device__ tcf operator()(const half2& x, const tcf& y)
    {
        float2 x2 = __half22float2(x);
        return tcf(y.real() + x2.x, y.imag() + x2.y);
    }
};

struct multiply_half2_functor : thrust::binary_function<half2, half2, half2>
{
    __device__ half2 operator()(const half2& x, const half2& y)
    {
        half a = __low2half(x);
        half b = __high2half(x);
        half c = __low2half(y);
        half d = __high2half(y);
        return make_half2(a * c - b * d, a * d + b * c);
    }
};

struct subtract_half2_functor : thrust::binary_function<half2, half2, half2>
{
    __device__ half2 operator()(const half2& x, const half2& y)
    {
        half a = __low2half(x);
        half b = __high2half(x);
        half c = __low2half(y);
        half d = __high2half(y);
        return make_half2(a - c, b - d);
    }
};

struct half_to_float_functor : thrust::unary_function<half, float>
{
    __host__ __device__ float operator()(const half& x)
    {
        return __half2float(x);
    }
};

struct float_to_half_functor : thrust::unary_function<float, half>
{
    __host__ __device__ half operator()(const float& x)
    {
        return __float2half(x);
    }
};

struct tcf_to_half2_functor : thrust::unary_function<tcf, half2>
{
    __host__ __device__ half2 operator()(const tcf& x)
    {
        return make_half2(__float2half(x.real()), __float2half(x.imag()));
    }
};

struct half2_to_tcf_functor : thrust::unary_function<half2, tcf>
{
    __device__ tcf operator()(const half2& x)
    {
        half a = __low2half(x);
        half b = __high2half(x);
        return tcf(__half2float(a), __half2float(b));
    }
};

struct field_functor
{
    __device__
        void operator()(const half2& x, const half2& y, tcf& z)
    {
        half a = __low2half(x);
        half b = __high2half(x);
        half c = __low2half(y);
        half d = __high2half(y);
        z += tcf(__half2float(a + c),
            __half2float(b + d));
    }
};

struct power_functor
{
    __device__
        void operator()(const half2& x, const half2& y, float& z)
    {
        float a = __half2float(__low2half(x));
        float b = __half2float(__high2half(x));
        float c = __half2float(__low2half(y));
        float d = __half2float(__high2half(y));
        z += a * a + b * b - c * c - d * d;
    }
};



struct add_all_functor
{
    __device__
        void operator()(const half2& x1, const half2& x2, tcf& z1, tcf& z2)
    {
        float2 y1 = __half22float2(x1);
        float2 y2 = __half22float2(x2);
        z1 = tcf(z1.real() + y1.x, z1.imag() + y1.y);
        z2 = tcf(z2.real() + y2.x, z2.imag() + y2.y);
    }
};

struct abs_value_squared : public thrust::binary_function<tcf, float, float>
{
    __host__ __device__
        float operator()(const tcf& x, const float& z)
    {
        return z + thrust::norm(x);
    }
};

struct triplet_functor
{
    __host__ __device__
        void operator()(const tcf& x, const tcf& y, float& z)
    {
        z += thrust::norm(x) - 2*thrust::norm(y);
    }
};



struct taper_functor : public thrust::binary_function<half2, half, half2>
{
    __device__
        half2 operator()(const half2& data, const half& taper)
    {
        return __hmul2(data, __half2half2(taper));
    }
};


#endif // !DSP_FUNCTORS_CUH