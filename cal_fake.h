#include "cmodules/timer.h"
#include <pthread.h>
#include <cal.h>

#define NUM_FAKE_EVENTS 1000000

class cal_fake_event
{
public:
    HighResTimer timer;
    int initialized;
    int queried;
    int reused;
    double delay;
    
    cal_fake_event() {initialized = queried = reused = 0;}
};

class cal_fake
{
public:
    cal_fake_event event[NUM_FAKE_EVENTS];
    pthread_mutex_t mutex;
    int curevent;
    
    cal_fake()
    {
	pthread_mutex_init(&mutex, NULL);
	curevent = 0;
    }
    
    ~cal_fake()
    {
	pthread_mutex_destroy(&mutex);
	for (int i = 0;i < curevent;i++)
	{
	    if (event[i].queried == 0) printf("Warning, event %d not queried\n", i);
	}
    }
    
    CALresult AddEvent(CALevent* pevent)
    {
	//printf("CREATE %d\n", curevent);
	*pevent = curevent;
	pthread_mutex_lock(&mutex);
	if (event[curevent].initialized) event[curevent].reused = 1;
	event[curevent].initialized = 1;
	event[curevent].queried = 0;
	event[curevent].timer.Reset();
	event[curevent].timer.Start();
	event[curevent].delay = (rand() % 1000) / 100000.;
	curevent = (curevent + 1) % NUM_FAKE_EVENTS;
	pthread_mutex_unlock(&mutex);
	return(CAL_RESULT_OK);
    }

    CALresult QueryEvent(CALevent num)
    {
	//printf("QUERY %d\n", num);
	CALresult retVal;
	pthread_mutex_lock(&mutex);
	if (num >= NUM_FAKE_EVENTS)
	{
	    printf("------------------------- Requested fake event with handle %d >= %d\n", num, NUM_FAKE_EVENTS);
	    retVal = CAL_RESULT_BAD_HANDLE;
	}
	else if (event[num].initialized == 0)
	{
	    printf("------------------------- Fake event with handle %d not initialized\n", num);
	    retVal = CAL_RESULT_BAD_HANDLE;
	}
	else if (event[num].queried)
	{
	    printf("------------------------- Fake event with handle %d already queried\n", num);
	    retVal = CAL_RESULT_BAD_HANDLE;
	}
	else
	{
	    event[num].timer.Stop();
	    if (event[num].timer.GetElapsedTime() > event[num].delay)
	    {
		event[num].queried = 1;
		retVal = CAL_RESULT_OK;
	    }
	    else
	    {
		event[num].timer.Start();
		retVal = CAL_RESULT_PENDING;
	    }
	}
	pthread_mutex_unlock(&mutex);
	if(retVal == CAL_RESULT_BAD_HANDLE) while(true);
	return(retVal);
    }
};

cal_fake fake;

#define calCtxRunProgram(event, ctx, func, rect) fake.AddEvent(event)
#define calMemCopy(event, ctx, src, dest, flags) fake.AddEvent(event)
#define calCtxIsEventDone(ctx, event) fake.QueryEvent(event)
