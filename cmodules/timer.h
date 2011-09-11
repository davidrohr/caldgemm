#ifndef TIMER_H
#define TIMER_H

class HighResTimer {

public:
	HighResTimer();
	~HighResTimer();
	void Start();
	void Stop();
	void Reset();
	double GetElapsedTime();

private:
	static double Frequency;
	static double GetFrequency();

	double ElapsedTime;
	double StartTime;
}; 

#endif
