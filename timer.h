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

	double Frequency;
	double ElapsedTime;
	double StartTime;
}; 

#endif
