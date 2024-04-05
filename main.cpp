#include "digitizer.h"
#include "dsp.cuh"
#include "measurement.cuh"
#include <algorithm>
#include <chrono>
#include <iostream>

int main()
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::microseconds;

    double part = 1;
    int second_oversampling = 2;
    try {
        auto dig = new Digitizer("/dev/spcm1");
        if (dig) { // Check if dig is not null
            int channels[] = {0, 1, 2, 3};
            int amps[] = {1000, 1000, 1000, 1000};

            dig->setupChannels(channels, amps, 4);

            dig->setSamplingRate(1250000000 / 4);
            dig->setExt0TriggerOnPositiveEdge(1000);
            // dig->setupMultRecFifoMode(800, 32, 0);
            // dig->autoTrigger();
            dig->setupSingleRecFifoMode(32);
            dig->setSegmentSize(800);
        }
        auto avg = 1 << 22;
        auto batch_size = 1 << 11;
        auto num_iter = int(avg / batch_size);
        auto mes = std::make_unique<Measurement>(dig, avg, batch_size, part, second_oversampling, "yok1");
        mes->setFirwin(1, 99);
        mes->setIntermediateFrequency(0.05f);
        mes->setCalibration(0, 1, 0, 0, 0);
        mes->setCalibration(1, 1, 0, 0, 0);
        mes->setAmplitude(100);
        mes->setCurrents(0, 0);
        std::pair<float, float> firwin_l (1, 99);
        std::pair<float, float> firwin_r (1, 99);
        mes->setCorrelationFirwin(firwin_l, firwin_r);
        auto t1 = high_resolution_clock::now();
        mes->measureWithCoil();
        auto t2 = high_resolution_clock::now();
        auto dur = duration_cast<microseconds>(t2 - t1);
        std::cout << "Measurement duration: " << dur.count() << " mcs\n";
        auto one_iter_dur = dur / num_iter;
        std::cout << "One iteration duration: " << one_iter_dur.count() << " mcs\n";
        // auto sd = mes->getSubtractionData();
        // mes->setSubtractionTrace(sd);
        // auto st = mes->getSubtractionTrace();
        // tcf a = st[0][0];
        // tcf b = sd[0][0];
        // std::cout << a << ' ' << b << std::endl;
        auto g2 = mes->getG2Correlator();
        auto g2_cross = mes->getG2CrossSegmentCorrelator();
        auto g2_filt = mes->getG2FilteredCorrelator();
        auto g2_filt_cross = mes->getG2FilteredCrossSegmentCorrelator();

        std::cout << g2[0][0] << std::endl;
        std::cout << g2_cross[0][0] << std::endl;
        std::cout << g2_filt[0][0] << std::endl;
        std::cout << g2_filt_cross[0][0] << std::endl;

    }
    catch (const std::runtime_error& e) {
        // Handle the exception
        std::cout << "Caught a runtime_error exception: " << e.what() << '\n';
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        // Catch any other types of exceptions
        std::cout << "Caught an unknown exception\n";
        return 1;
    }
    std::cout << "all right" << std::endl;
 
    // try {
    //     auto mes = std::make_unique<Measurement>(1 << 12, 1 << 6, 800, part, second_oversampling);
    //     mes->setFirwin(1, 99);
    //     // mes->setIntermediateFrequency(0.05f);
    //     // mes->setCalibration(1, 0, 0, 0);
    //     mes->setAmplitude(1);
    //     // mes->setCurrents(-0.5435e-3f, -2.5e-3f);
    //     // float firwin_l[2] = {1, 99};
    //     // float firwin_r[2] = {1, 99};
    //     // mes->setCorrelationFirwin(firwin_l, firwin_l);
    //     std::vector<int8_t> input(4000, 127);
    //     mes->setTestInput(input);
    //     mes->measureTest();
    //     auto g2 = mes->getG2Correlator("g2_full");
    // }
    // catch (const std::runtime_error& e) {
    //     // Handle the exception
    //     std::cout << "Caught a runtime_error exception: " << e.what() << '\n';
    //     return 1;
    // }
    // catch (const std::exception& e) {
    //     std::cerr << "Exception: " << e.what() << std::endl;
    //     return 1;
    // }
    // catch (...) {
    //     // Catch any other types of exceptions
    //     std::cout << "Caught an unknown exception\n";
    //     return 1;
    // }
    // std::cout << "All right!" << std::endl;
    
    // std::fill(mes.test_input.begin(), mes.test_input.end(), 10);
    // mes.measureTest();
    // gpubuf vec(mes.test_input.begin(), mes.test_input.end());
    // gpuvec_c data(mes.test_input.size());
    // auto func = millivolts_functor(1.5);

    // cudaStream_t s1;`
    // cudaStreamCreate(&s1);
    // strided_range<gpubuf::iterator> channelI(vec.begin(), vec.end(), 2);
    // strided_range<gpubuf::iterator> channelQ(vec.begin() + 1, vec.end(), 2);
    // thrust::transform(thrust::device.on(s1), channelI.begin(), channelI.end(), channelQ.begin(), data.begin(), func);
    // std::cout << data[0] << std::endl;


    return 0;
}
