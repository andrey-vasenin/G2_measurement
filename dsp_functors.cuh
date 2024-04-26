#ifndef DSP_FUNCTORS_CUH
#define DSP_FUNCTORS_CUH

#include "dsp.cuh"
#include <corecrt_math_defines.h>

struct hann_window_functor
{
    const int N;
    hann_window_functor(int _N) : N(_N) {}

    __host__ __device__
    float operator()(const int& n) const
    {
        return 0.5 * (1 - cos(2 * M_PI * n / (N - 1)));
    }
};

struct gaussian_window_functor
{
    const int N;
    const float sigma;

    gaussian_window_functor(int _N, float _sigma) : N(_N), sigma(_sigma) {}

    __host__ __device__
    float operator()(const int& n) const
    {
        float numerator = static_cast<float>(n) - (N - 1) / 2.0f;
        float exponent = -0.5f * (numerator / sigma) * (numerator / sigma);
        return exp(exponent);
    }
};

struct replication_and_windowing_functor {
    int L; // Длина сегмента
    int M; // Перекрытие
    int original_length; // Длина исходного сигнала
    const tcf* original_signal; // Указатель на исходный сигнал
    const float* window_data; // Сырой указатель на данные окна

    replication_and_windowing_functor(int _L, int _M, int _original_length, const tcf* _original_signal, const float* _window_data)
        : L(_L), M(_M), original_length(_original_length), original_signal(_original_signal), window_data(_window_data) {}

    __device__ tcf operator()(const size_t& i) const {
        int segment_index = i % L; // Индекс внутри сегмента
        int segment_number = i / L; // Номер сегмента
        int original_index = segment_number * (L - M) + segment_index;

        if (original_index < original_length) {
            return original_signal[original_index] * tcf(window_data[segment_index]);
        } else {
            return tcf(0.f); // Дополнение нулями, если выходим за пределы исходного сигнала
        }
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

    __device__ inline __forceinline__ void operator()(tcf &x)
    {
        float xr = x.real();
        float xi = x.imag();
        x.real(xr + c_i);
        x.imag(a_qi * xr + a_qq * xi + c_q);
    }
};

struct scaler_functor
{
    float scalar;

    scaler_functor(float scalar) : scalar(scalar) {}

    __device__ inline void operator()(tcf& x) const {
        x *= scalar;
    }
};

struct millivolts_functor
{
    const float scale;

    millivolts_functor(float s) : scale(s) {}

    __device__ inline tcf operator()(const char2 &b)
    {
        return tcf(static_cast<float>(b.x), static_cast<float>(b.y)) * scale;
    }
};

struct field_functor
{
    __device__ inline __forceinline__ void operator()(const tcf &x, const tcf &y, tcf &z)
    {
        z += x - y;
    }
};

struct power_functor
{
    __device__ inline __forceinline__ void operator()(const tcf &x, const tcf &y, float &z)
    {
        z += thrust::norm(x) - thrust::norm(y);
    }
};

struct cross_power_functor
{   

    __device__ inline __forceinline__ tcf operator()(const tcf &x, const tcf &y)
    {
        return thrust::conj(x) * y;
    }
};

struct downconv_functor
{
    __device__ inline __forceinline__
        tcf
        operator()(const tcf &x, const tcf &y)
    {
        return x * y;
    }
};

struct taper_functor
{
    __device__ inline __forceinline__
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