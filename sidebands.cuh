#pragma once
#ifndef SIDEBANDS_H
#define SIDEBANDS_H
#include <cufft.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include "dsp_types.cuh"
#include "dsp_helpers.cuh"
#include "dsp_functors.cuh"
#include "dsp.cuh"
#include "dsp_micromodules.cuh"

class sidebands_module : public micromodule
{
private:
    int trace_length;
    int total_length;
    int batch_size;
    float samplerate;
    cufftHandle plan;
    cudaStream_t stream;
    gpuvec_c data_sideband1;
    gpuvec_c data_sideband2;
    gpuvec_c data_central_peak;
    gpuvec_c noise_sideband1;
    gpuvec_c noise_sideband2;
    gpuvec_c noise_central_peak;
    gpuvec_c data_tmp;
    gpuvec_c noise_tmp;
    gpuvec_c data_product;
    gpuvec_c noise_product;
    gpuvec_c corrfirwin1;
    gpuvec_c corrfirwin2;
    gpuvec_c corrfirwin_central;
    gpuvec PSD_sideband1;
    gpuvec PSD_sideband2;
    gpuvec PSD_rayleigh;
    gpuvec PSD_total;
    gpuvec PSD_sidebands_product;

public:
    sidebands_module(int len, int batch_size, float sample_rate, cufftHandle cufft_plan,
                     cudaStream_t cuda_stream) : trace_length{len}, batch_size{batch_size},
                                                 total_length{len * batch_size}, samplerate{sample_rate},
                                                 plan{cufft_plan}, stream{cuda_stream}
    {
        data_sideband1.resize(total_length, tcf(0));
        data_sideband2.resize(total_length, tcf(0));
        data_central_peak.resize(total_length, tcf(0));
        noise_sideband1.resize(total_length, tcf(0));
        noise_sideband2.resize(total_length, tcf(0));
        noise_central_peak.resize(total_length, tcf(0));
        data_tmp.resize(total_length, tcf(0));
        noise_tmp.resize(total_length, tcf(0));
        data_product.resize(total_length, tcf(0));
        noise_product.resize(total_length, tcf(0));
        PSD_sideband1.resize(total_length, 0);
        PSD_sideband2.resize(total_length, 0);
        PSD_rayleigh.resize(total_length, 0);
        PSD_total.resize(total_length, 0);
        PSD_sidebands_product.resize(total_length, 0);
    };

    void reset()
    {
        auto fill_with_zero = [](auto &dv)
        {
            using ContainerType = typename std::remove_reference<decltype(dv)>::type;
            thrust::fill(dv.begin(), dv.end(), typename ContainerType::value_type(0));
        };
        fill_with_zero(data_sideband1);
        fill_with_zero(data_sideband2);
        fill_with_zero(data_central_peak);
        fill_with_zero(noise_sideband1);
        fill_with_zero(noise_sideband2);
        fill_with_zero(noise_central_peak);
        fill_with_zero(data_tmp);
        fill_with_zero(noise_tmp);

        fill_with_zero(PSD_sideband1);
        fill_with_zero(PSD_sideband2);
        fill_with_zero(PSD_rayleigh);
        fill_with_zero(PSD_total);
        fill_with_zero(PSD_sidebands_product);

        fill_with_zero(data_product);
        fill_with_zero(noise_product);
    };

    void set_filter_cutoffs(std::vector<std::pair<float, float>> &cutoffs) override
    {
        if (cutoffs.size() != 3)
            throw std::runtime_error("3 pairs of cutoff frequencies are required by the sidebands module");
        dsp::makeFilterWindow(std::get<0>(cutoffs[0]), std::get<1>(cutoffs[0]), corrfirwin1, trace_length, total_length, samplerate);
        dsp::makeFilterWindow(std::get<0>(cutoffs[1]), std::get<1>(cutoffs[1]), corrfirwin2, trace_length, total_length, samplerate);
        dsp::makeFilterWindow(std::get<0>(cutoffs[2]), std::get<1>(cutoffs[2]), corrfirwin_central, trace_length, total_length, samplerate);
    }

    void compute(gpuvec_c &data, gpuvec_c &noise) override
    {
        // PSD for the whole trace
        calculatePSD(data, noise, PSD_total);

        // Left sideband
        extractSideband(data, data_sideband1, corrfirwin1);
        extractSideband(noise, noise_sideband1, corrfirwin1);
        calculatePSD(data_sideband1, noise_sideband1, PSD_sideband1);

        // Right sideband
        extractSideband(data, data_sideband2, corrfirwin2);
        extractSideband(noise, noise_sideband2, corrfirwin2);
        calculatePSD(data_sideband2, noise_sideband2, PSD_sideband2);

        // Central rayleigh peak
        extractSideband(data, data_central_peak, corrfirwin_central);
        extractSideband(noise, noise_central_peak, corrfirwin_central);
        calculatePSD(data_central_peak, noise_central_peak, PSD_rayleigh);

        // Photons from two sidebands
        calculateSidebandsProductPSD(data_sideband1,
                                     data_sideband2,
                                     noise_sideband1,
                                     noise_sideband2,
                                     PSD_sidebands_product);
    };

    void extractSideband(gpuvec_c &src, gpuvec_c &dst, gpuvec_c &filterwin)
    {
        thrust::copy(thrust::cuda::par_nosync.on(stream), src.begin(), src.end(), dst.begin());
        dsp::applyFilter(dst, filterwin, trace_length, stream, plan);
    };

    void calculatePSD(gpuvec_c &data, gpuvec_c &noise, gpuvec &output)
    {
        cufftComplex *cufft_data_src = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(data.data()));
        cufftComplex *cufft_data_dst = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(data_tmp.data()));
        auto cufftstat_d = cufftExecC2C(plan, cufft_data_src, cufft_data_dst, CUFFT_FORWARD);
        check_cufft_error(cufftstat_d, "Error executing cufft");

        cufftComplex *cufft_noise_src = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(noise.data()));
        cufftComplex *cufft_noise_dst = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(noise_tmp.data()));
        auto cufftstat_n = cufftExecC2C(plan, cufft_noise_src, cufft_noise_dst, CUFFT_FORWARD);
        check_cufft_error(cufftstat_n, "Error executing cufft");

        thrust::for_each(thrust::cuda::par_nosync.on(stream),
                         thrust::make_zip_iterator(data_tmp.begin(), noise_tmp.begin(), output.begin()),
                         thrust::make_zip_iterator(data_tmp.end(), noise_tmp.end(), output.end()),
                         thrust::make_zip_function(power_functor()));
    };

    void calculateSignalsProduct(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &output)
    {
        thrust::transform(thrust::cuda::par_nosync.on(stream), data1.cbegin(), data1.cend(), data2.cbegin(), output.begin(), thrust::multiplies<tcf>());
    };

    void calculateSidebandsProductPSD(gpuvec_c &sideband1, gpuvec_c &sideband2,
                                      gpuvec_c &noise1, gpuvec_c &noise2, gpuvec &psd)
    {
        calculateSignalsProduct(sideband1, sideband2, data_product);
        calculateSignalsProduct(noise1, noise2, noise_product);
        calculatePSD(data_product, noise_product, psd);
    };

    std::tuple<hostvec, hostvec, hostvec, hostvec, hostvec> getPSDResults()
    {
        return {dsp::sumOverBatch(PSD_sideband1, batch_size),
                dsp::sumOverBatch(PSD_sideband2, batch_size),
                dsp::sumOverBatch(PSD_rayleigh, batch_size),
                dsp::sumOverBatch(PSD_total, batch_size),
                dsp::sumOverBatch(PSD_sidebands_product, batch_size)};
    };

    std::vector<hostvec> getRealResults() override
    {
        std::vector<hostvec> results(0);
        results.push_back(dsp::sumOverBatch(PSD_sideband1, batch_size));
        results.push_back(dsp::sumOverBatch(PSD_sideband2, batch_size));
        results.push_back(dsp::sumOverBatch(PSD_rayleigh, batch_size));
        results.push_back(dsp::sumOverBatch(PSD_total, batch_size));
        results.push_back(dsp::sumOverBatch(PSD_sidebands_product, batch_size));

        return results;
    };

    std::vector<hostvec_c> getComplexResults() override
    {
        std::vector<hostvec_c> results(0);
        return results;
    };
};
#endif // SIDEBANDS_H