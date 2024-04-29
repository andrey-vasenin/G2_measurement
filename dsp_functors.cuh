#ifndef DSP_FUNCTORS_CUH
#define DSP_FUNCTORS_CUH

#include "dsp.cuh"

struct complex_conjugate {
    __device__
    thrust::complex<float> operator()(thrust::complex<float>& x) const {
        return thrust::conj(x);
    }
};

struct calibration_functor : thrust::unary_function<tcf &, void>
{
    const float a_qi, a_qq;
    const float c_i, c_q;

    calibration_functor(float _a_qi, float _a_qq,
                        float _c_i, float _c_q) : a_qi{_a_qi}, a_qq{_a_qq},
                                                  c_i{_c_i}, c_q{_c_q}
    {
    }

    __device__ inline void operator()(tcf &x)
    {
        float xr = x.real();
        float xi = x.imag();
        x.real(xr + c_i);
        x.imag(a_qi * xr + a_qq * xi + c_q);
    }
};


struct millivolts_functor
{
    const float scale;

    millivolts_functor(float s) : scale(s) {}

    __device__ inline void operator()(const char4 &b, tcf &d1, tcf &d2)
    {
        d1.real(static_cast<float>(b.x) * scale);
        d1.imag(static_cast<float>(b.y) * scale);
        d2.real(static_cast<float>(b.z) * scale);
        d2.imag(static_cast<float>(b.w) * scale);
    }
};

struct field_functor
{
    __device__ inline void operator()(const tcf &x, const tcf &y, tcf &z)
    {
        z += x - y;
    }
};

struct power_functor
{
    __device__ inline void operator()(const tcf &x, const tcf &y, float &z)
    {
        z += thrust::norm(x) - thrust::norm(y);
    }
};

struct corr_functor
{
    __device__ inline void operator()(const tcf &x, const tcf &y, tcf &z)
    {
        z += x * y;
    }
};

struct cross_power_functor : thrust::binary_function<const tcf &, const tcf &, tcf>
{   

    __device__ inline tcf operator()(const tcf &x, const tcf &y)
    {
        return thrust::conj(x) * y;
    }
};

struct downconv_functor : public thrust::binary_function<const tcf &, const tcf &, tcf>
{
    __device__ inline
        tcf
        operator()(const tcf &x, const tcf &y)
    {
        return x * y;
    }
};

struct taper_functor : public thrust::binary_function<const tcf &, const float &, tcf>
{
    __device__ inline
        tcf
        operator()(const tcf &data, const float &taper)
    {
        return data * taper;
    }
};

struct downsample2_functor
{
    __device__ inline tcf operator()(const tcf &s1, const tcf &s2)
    {
        return (s1 + s2) / 2.f;
    }
};

struct downsample4_functor
{
    __device__ inline tcf operator()(const tcf &s1, const tcf &s2, const tcf &s3, const tcf &s4)
    {
        return (s1 + s2 + s3 + s4) / 4.f;
    }
};

#endif // !DSP_FUNCTORS_CUH