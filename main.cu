#include "digitizer.h"
#include "measurement.cuh"
#include <algorithm>
#include <chrono>

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::seconds;

    auto dig = Digitizer("/dev/spcm1");
    int channels[] = { 0,1 };
    int amps[] = { 1000, 1000 };

    dig.setupChannels(channels, amps, 2);
    dig.setSamplingRate(1250000000 / 4);
    dig.setExt0TriggerOnPositiveEdge(1000);
    dig.setupMultRecFifoMode(5024, 32, 0);
    double part = 0.40764331210191085;
    int K = 26;
    auto mes = Measurement(&dig, 1 << 20, 1 << 11, part, K, "yok1");

    // Generate fake tapers
    int N = mes.getTraceLength();
    std::vector<stdvec> tapers(K);
    for (int i = 0; i < K; i++)
    {
        tapers[i].resize(N);
        std::fill(tapers[i].begin(), tapers[i].end(), static_cast<float>(i));
    }

    mes.setTapers(tapers);

    mes.setFirwin(1, 99);
    mes.setIntermediateFrequency(0.05);
    mes.setCalibration(1, 0, 0, 0);
    mes.setAmplitude(100);
    mes.setCurrents(-0.5435e-3, -2.5e-3);

    auto t1 = high_resolution_clock::now();
    mes.measure();
    auto t2 = high_resolution_clock::now();
    auto dur = duration_cast<seconds>(t2 - t1);

    std::cout << "Measurement duration: " << dur.count() << "s\n";

    //std::fill(mes.test_input.begin(), mes.test_input.end(), 10);
   /* mes.measureTest();
    gpubuf vec(mes.test_input.begin(), mes.test_input.end());
    gpuvec_c data(mes.test_input.size());
    auto func = millivolts_functor(1.5);

    cudaStream_t s1;
    cudaStreamCreate(&s1);
    strided_range<gpubuf::iterator> channelI(vec.begin(), vec.end(), 2);
    strided_range<gpubuf::iterator> channelQ(vec.begin() + 1, vec.end(), 2);
    thrust::transform(thrust::device.on(s1), channelI.begin(), channelI.end(), channelQ.begin(), data.begin(), func);
    std::cout << data[0] << std::endl;*/

    auto field = mes.getPSD();
    std::cout << field[0] << std::endl;

    return 0;
}