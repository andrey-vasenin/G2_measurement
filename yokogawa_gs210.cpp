#include "yokogawa_gs210.h"
#include <stdexcept>
#include <format>
#include <cstring>

yokogawa_gs210::yokogawa_gs210(const char* addr)
{
	// Assign the address of your instrument to the variable
	VISA_ADDRESS = new char[100];
	strcpy_s(VISA_ADDRESS, 100 * sizeof(char), addr);
	// Create a connection to the instrument
	try
	{
		status = viOpenDefaultRM(&resourceManager);
		this->check_status();
		status = viOpen(resourceManager, VISA_ADDRESS, VI_NO_LOCK, 0, &session);
		this->check_status();
	}
	catch (const std::exception exc)
	{
		throw exc;
	}
}

yokogawa_gs210::~yokogawa_gs210()
{
	// remove pointers to the dynamic memory
	delete [] VISA_ADDRESS;
	// Close the connection to the instrument
	viClose(session);
	viClose(resourceManager);
}

std::string yokogawa_gs210::get_device_information()
{
	return std::format("*IDN? returned: {}\n", this->read_string("*IDN?\n"));
}

void yokogawa_gs210::set_current(double current)
{
	viPrintf(session, "SOURce:LEVEL %le\n", current);
}

double yokogawa_gs210::get_current()
{
	return this->read_double("SOURce:LEVEL?\n");
}

void yokogawa_gs210::check_status()
{
	using namespace std::string_literals;
	if (status < VI_SUCCESS)
	{
		ViChar errorMessage[256];
		int error = viStatusDesc(session, status, errorMessage);
		throw std::runtime_error(std::format("{}\nError code: {}\n"s, errorMessage, status));
	}
}

double yokogawa_gs210::read_double(const char* command)
{
	ViReal64 value;
	viPrintf(session, command);
	viScanf(session, "%le", &value);
	return value;
}

std::string yokogawa_gs210::read_string(const char* command)
{
	viPrintf(session, "*IDN?\n");
	ViChar idnResponse[100];
	viScanf(session, "%t", idnResponse);
	return std::string{ idnResponse };
}
