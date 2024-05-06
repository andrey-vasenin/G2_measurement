#pragma once
#ifndef YOKOGAWA_GS210_H
#define YOKOGAWA_GS210_H
#include "visa.h"
#include <string>

class yokogawa_gs210
{
	ViRsrc VISA_ADDRESS;
	ViSession resourceManager{0};
	ViSession session{0};
	ViStatus status{0};

public:
	yokogawa_gs210(const char *addr);
	~yokogawa_gs210();
	std::string get_device_information();
	void set_current(double current);
	double get_current();

private:
	void check_status();
	double read_double(const char *);
	std::string read_string(const char *);
};
#endif // YOKOGAWA_GS210_H