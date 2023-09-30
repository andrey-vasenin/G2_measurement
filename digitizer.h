//
// Created by andrei on 3/26/21.
//

#ifndef CPPMEASUREMENT_DIGITIZER_H
#define CPPMEASUREMENT_DIGITIZER_H

#include <vector>
#include <functional>
#include <ostream>
#include "dlltyp.h"
#include "regs.h"
#include "dsp.cuh"

typedef std::function<void(int8_t *)> proc_t;

class Digitizer
{
    drv_handle handle;
    int32 slot;
    int8_t *buffer;
    size_t buffersize;
    char errortext[ERRORTEXTLEN];
    bool created_here = false;

    void loadProperties();

public:
    /* Constructors */
    Digitizer(void *h);

    Digitizer(const char *addr);

    ~Digitizer();

    /* Getters */
    int32 getSlotNumber();

    size_t getBufferSize();

    size_t getMemsize();

    int8_t *getBuffer();

    size_t getSegmentSize();

    int getSegmentsNumber();

    void *getHandle();

    /* Setters */
    void setBuffer(int8_t *buf, size_t buffersize);

    /* Setup */
    void setupChannels(const int *channels, const int *amplitudes, int size);

    void antialiasing(bool flag);

    void setDelay(int delay);

    void setSamplingRate(int samplerate);

    int getSamplingRate();

    void setTimeout(int milliseconds);

    void handleError();

    void setExt0TriggerOnPositiveEdge(int32 voltageThreshold);

    /* Mode setters */
    void setupMultRecFifoMode(int32 segmentsize, int32 pretrigger, int segments);

    /* Measurers */
    void prepareFifo(uint32 notifysize);

    void stopFifo();

    void launchFifo(uint32 notifysize, int n, proc_t processor, bool computing);

    /* Control */
    void stopCard();

    /* Operators */
    friend std::ostream &operator<<(std::ostream &out, const Digitizer &dig)
    {
        return out << "digitizer in PXIe slot #" << dig.slot;
    };

    int64_t getTriggerCounter();
};

#endif // CPPMEASUREMENT_DIGITIZER_H
