#include "cmodules/timer.h"
#ifdef _WIN32
#include "cmodules/pthread_mutex_win32_wrapper.h"
#else
#include <pthread.h>
#endif
#include <cal.h>

#define NUM_FAKE_EVENTS 1000000
#define NUM_FAKE_MEM 10000
#define NUM_FAKE_MODULE 100
#define NUM_FAKE_NAME 1000
#define NUM_MODULE_NAMES 12

class cal_fake_event
{
public:
	HighResTimer timer;
	int initialized;
	int queried;
	int reused;
	double delay;
	int mems[NUM_MODULE_NAMES];
	int nmems;

	cal_fake_event() {initialized = queried = reused = 0;}
};

class cal_fake_mem
{
public:
	int released;
	int active;
};

class cal_fake_module
{
public:
	int released;
	int nnames;
	int names[NUM_MODULE_NAMES];
};

class cal_fake_name
{
public:
	int mem;
};

class cal_fake
{
public:
	cal_fake_event event[NUM_FAKE_EVENTS];
	pthread_mutex_t mutex;
	int curevent;

	cal_fake_mem mem[NUM_FAKE_MEM];
	int curmem;

	cal_fake_module module[NUM_FAKE_MODULE];
	int curmodule;

	cal_fake_name name[NUM_FAKE_NAME];
	int curname;

	cal_fake()
	{
		pthread_mutex_init(&mutex, NULL);
		curevent = 0;
		curmem = 0;
		curmodule = 0;
		curname = 0;
	}

	~cal_fake()
	{
		pthread_mutex_destroy(&mutex);
		for (int i = 0;i < curevent;i++)
		{
			if (event[i].queried == 0) printf("Warning, event %d not queried\n", i);
		}
	}

	CALresult AddEvent(CALevent* pevent, bool lock = true)
	{
		//printf("CREATE %d\n", curevent);
		*pevent = curevent;
		if (lock) pthread_mutex_lock(&mutex);
		if (event[curevent].initialized) event[curevent].reused = 1;
		event[curevent].initialized = 1;
		event[curevent].queried = 0;
		event[curevent].timer.Reset();
		event[curevent].timer.Start();
		event[curevent].delay = (rand() % 1000) / 100000.;
		event[curevent].nmems = 0;
		curevent = (curevent + 1) % NUM_FAKE_EVENTS;
		if (lock) pthread_mutex_unlock(&mutex);
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
				for (int i = 0;i < event[num].nmems;i++) mem[event[num].mems[i]].active = 0;
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

	CALresult AddMemHandle(CALmem* m)
	{
		pthread_mutex_lock(&mutex);
		if (curmem == NUM_FAKE_MEM)
		{
			fprintf(stderr, "NUM_FAKE_MEM overflow\n");
			exit(1);
		}
		*m = curmem;
		mem[curmem].released = 0;
		mem[curmem].active = 0;
		curmem++;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult AddModule(CALmodule* mod)
	{
		pthread_mutex_lock(&mutex);
		if (curmodule == NUM_FAKE_MODULE)
		{
			fprintf(stderr, "NUM_FAKE_MODULE overflow\n");
			exit(1);
		}
		*mod = curmodule;
		module[curmodule].released = 0;
		module[curmodule].nnames = 0;
		curmodule++;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult AddName(CALname* nam, CALmodule mod)
	{
		pthread_mutex_lock(&mutex);
		if (curname == NUM_FAKE_NAME)
		{
			fprintf(stderr, "NUM_FAKE_NAME overflow\n");
			exit(1);
		}
		if (mod > curmodule)
		{
			fprintf(stderr, "Invalid Module\n");
			exit(1);
		}
		if (module[mod].nnames == NUM_MODULE_NAMES)
		{
			fprintf(stderr, "NUM_MODULE_NAMES overflow\n");
			exit(1);
		}
		*nam = curname;
		module[mod].names[module[mod].nnames] = curname;
		module[mod].nnames++;
		name[curname].mem = 0;
		curname++;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult FakeMemcpy(CALmem mem1, CALmem mem2, CALevent* ev)
	{
		pthread_mutex_lock(&mutex);
		if (mem[mem1].active || mem[mem2].active)
		{
			fprintf(stderr, "Memory active when starting memcpy\n");
			exit(1);
		}
		AddEvent(ev, false);
		event[*ev].nmems = 2;
		event[*ev].mems[0] = mem1;
		event[*ev].mems[1] = mem2;
		mem[mem1].active = 1;
		mem[mem2].active = 1;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult FakeKernel(CALfunc func, CALevent* ev)
	{
		pthread_mutex_lock(&mutex);
		if (func > curmodule)
		{
			fprintf(stderr, "Invalid func/module");
			exit(1);
		}
		for (int i = 0;i < module[func].nnames;i++)
		{
			if (mem[name[module[func].names[i]].mem].active)
			{
				fprintf(stderr, "Memory active when starting kernel\n");
				exit(1);
			}
			mem[name[module[func].names[i]].mem].active = 1;
		}
		AddEvent(ev, false);
		event[*ev].nmems = module[func].nnames;
		for (int i = 0;i < module[func].nnames;i++) event[*ev].mems[i] = name[module[func].names[i]].mem;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult SetMem(CALname nam, CALmem m)
	{
		if (nam > curname || m > curmem)
		{
			fprintf(stderr, "Invalid name/mem\n");
			exit(1);
		}
		name[nam].mem = m;
		return(CAL_RESULT_OK);
	}

	CALresult GetFunc(CALfunc* fun, CALmodule mod)
	{
		*fun = mod;
		return(CAL_RESULT_OK);
	}

	CALresult ReleaseMem(int m)
	{
		mem[m].released = 1;
		return(CAL_RESULT_OK);
	}

	CALresult UnloadModule(int mod)
	{
		module[mod].released = 1;
		return(CAL_RESULT_OK);
	}
};

cal_fake fake;

#define calCtxRunProgram(event, ctx, func, rect) fake.AddEvent(event)
#define calMemCopy(event, ctx, src, dest, flags) fake.FakeMemcpy(src, dest, event)
#define calCtxIsEventDone(ctx, event) fake.QueryEvent(event)
#define calCtxGetMem(mem, ctx, res) fake.AddMemHandle(mem)
#define calCtxSetMem(ctx, name, mem) fake.SetMem(name, mem)
#define calCtxReleaseMem(ctx, mem) fake.ReleaseMem(mem)
#define calModuleLoad(module, ctx, image) fake.AddModule(module)
#define calModuleUnload(ctx, module) fake.UnloadModule(module)
#define calModuleGetName(name, ctx, module, string) fake.AddName(name, module)
#define calModuleGetEntry(func, ctx, module, string) fake.GetFunc(func, module)