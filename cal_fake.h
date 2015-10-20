/**
 * This file is part of the CALDGEMM library.
 *
 * Copyright 2015:
 *  - David Rohr (drohr@jwdt.org)
 *  - Matthias Bach (bach@compeng.uni-frankfurt.de)
 *  - Matthias Kretz (kretz@compeng.uni-frankfurt.de)
 *
 * This file is part of CALDGEMM.
 *
 * CALDGEMM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CALDGEMM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CALDGEMM.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CAL_FAKE_H
#define CAL_FAKE_H

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
#define NUM_MODULE_NAMES 13

#define CAL_FAKE_PASSTHROUGH
#define CAL_FAKE_CHECKMEM
//#define CAL_FAKE_VERBOSE

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
	CALevent through;

	cal_fake_event() {initialized = queried = reused = 0;}
};

class cal_fake_mem
{
public:
	int released;
	int active;
	
	CALmem through;
};

class cal_fake_module
{
public:
	int released;
	int nnames;
	int names[NUM_MODULE_NAMES];
	
	CALmodule through;
	CALfunc throughFunc;
};

class cal_fake_name
{
public:
	int mem;
	
	CALname through;
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
#ifdef CAL_FAKE_VERBOSE
		fprintf(STD_OUT, "CREATE EVENT %d\n", curevent);
#endif
		*pevent = curevent;
		if (lock) pthread_mutex_lock(&mutex);
		if (event[curevent].initialized && !event[curevent].queried)
		{
			printf("------------------------ Event reused before queried\n");
			while (true);
		}
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
#ifdef CAL_FAKE_VERBOSE
		fprintf(STD_OUT, "QUERY EVENT %d\n", num);
#endif
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
#ifndef CAL_FAKE_PASSTHROUGH
			if (event[num].timer.GetElapsedTime() <= event[num].delay)
			{
				event[num].timer.Start();
				retVal = CAL_RESULT_PENDING;
			}
			else
#endif
			{
				event[num].queried = 1;
				for (int i = 0;i < event[num].nmems;i++) mem[event[num].mems[i]].active--;
				retVal = CAL_RESULT_OK;
			}
		}
		pthread_mutex_unlock(&mutex);
		if(retVal == CAL_RESULT_BAD_HANDLE) while(true);
		return(retVal);
	}
	
	void ListMemCollisions(int mem)
	{
	    for (int i = 0;i < NUM_FAKE_EVENTS;i++)
	    {
		if (event[i].initialized && !event[i].queried)
		{
		    for (int j = 0;j < event[i].nmems;j++)
		    {
			if (event[i].mems[j] == mem)
			{
			    printf("Collision with event %d\n", i);
			}
		    }
		}
	    }
	}

	CALresult AddMemHandle(CALmem* m)
	{
		pthread_mutex_lock(&mutex);
		if (curmem == NUM_FAKE_MEM)
		{
			fprintf(stderr, "NUM_FAKE_MEM overflow\n");
			while(true);
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
			while(true);
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
		//printf("Giving name %d (mod %d)\n", curname, mod);
		pthread_mutex_lock(&mutex);
		if (curname == NUM_FAKE_NAME)
		{
			fprintf(stderr, "NUM_FAKE_NAME overflow\n");
			while(true);
		}
		if (mod > (unsigned) curmodule)
		{
			fprintf(stderr, "Invalid Module\n");
			while(true);
		}
		if (module[mod].nnames == NUM_MODULE_NAMES)
		{
			fprintf(stderr, "NUM_MODULE_NAMES overflow\n");
			while(true);
		}
		*nam = curname;
		module[mod].names[module[mod].nnames] = curname;
		module[mod].nnames++;
		name[curname].mem = 0;
		curname++;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult FakeMemcpy(CALmem mem1, CALmem mem2, CALevent* ev, int allowOverlap = 0)
	{
		pthread_mutex_lock(&mutex);
#ifdef CAL_FAKE_CHECKMEM
		if (allowOverlap == 0 && (mem[mem1].active || mem[mem2].active))
		{
			fprintf(stderr, "Memory active when starting memcpy (src: %d, dst: %d)\n", mem[mem1].active, mem[mem2].active);
			while(true);
		}
#endif
		AddEvent(ev, false);
		event[*ev].nmems = 2;
		event[*ev].mems[0] = mem1;
		event[*ev].mems[1] = mem2;
		mem[mem1].active++;
		mem[mem2].active++;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult FakeKernel(CALfunc func, CALevent* ev, int allowOverlap)
	{
		pthread_mutex_lock(&mutex);
		if (func > (unsigned) curmodule)
		{
			fprintf(stderr, "Invalid func/module");
			while(true);
		}
#ifdef CAL_FAKE_CHECKMEM
		for (int i = 0;i < module[func].nnames;i++)
		{
			if (i >= allowOverlap && mem[name[module[func].names[i]].mem].active)
			{
				fprintf(stderr, "Memory %d (of %d) active when starting kernel (allowed overlap %d)\n", i, module[func].nnames, allowOverlap);
				ListMemCollisions(name[module[func].names[i]].mem);
				while(true);
			}
			mem[name[module[func].names[i]].mem].active++;
		}
#endif
		AddEvent(ev, false);
		event[*ev].nmems = module[func].nnames;
		for (int i = 0;i < module[func].nnames;i++) event[*ev].mems[i] = name[module[func].names[i]].mem;
		pthread_mutex_unlock(&mutex);
		return(CAL_RESULT_OK);
	}

	CALresult SetMem(CALname nam, CALmem m)
	{
		if (nam > (unsigned) curname || m > (unsigned) curmem)
		{
			fprintf(stderr, "Invalid name/mem\n");
			while(true);
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

#ifndef CAL_FAKE_PASSTHROUGH
#define calCtxRunProgram(event, ctx, func, rect) fake.FakeKernel(func, event)
#define calMemCopy(event, ctx, src, dest, flags) fake.FakeMemcpy(src, dest, event)
#define calCtxIsEventDone(ctx, event) fake.QueryEvent(event)
#define calCtxGetMem(mem, ctx, res) fake.AddMemHandle(mem)
#define calCtxSetMem(ctx, name, mem) fake.SetMem(name, mem)
#define calCtxReleaseMem(ctx, mem) fake.ReleaseMem(mem)
#define calModuleLoad(module, ctx, image) fake.AddModule(module)
#define calModuleUnload(ctx, module) fake.UnloadModule(module)
#define calModuleGetName(name, ctx, module, string) fake.AddName(name, module)
#define calModuleGetEntry(func, ctx, module, string) fake.GetFunc(func, module)
#else

static inline CALresult calCtxRunProgram_a(CALevent* event, CALcontext ctx, CALfunc func, CALdomain* rect)
{
	fake.FakeKernel(func, event, 0);
	return(calCtxRunProgram(&fake.event[*event].through, ctx, fake.module[func].throughFunc, rect));
}

static inline CALresult calMemCopy_a(CALevent* event, CALcontext ctx, CALmem src, CALmem dest, CALuint flags)
{
	fake.FakeMemcpy(src, dest, event, 0);
	return(calMemCopy(&fake.event[*event].through, ctx, fake.mem[src].through, fake.mem[dest].through, flags));
}

static inline CALresult calCtxRunProgram_b(CALevent* event, CALcontext ctx, CALfunc func, CALdomain* rect, int allowOverlap = 0)
{
	fake.FakeKernel(func, event, allowOverlap);
	return(calCtxRunProgram(&fake.event[*event].through, ctx, fake.module[func].throughFunc, rect));
}

static inline CALresult calMemCopy_b(CALevent* event, CALcontext ctx, CALmem src, CALmem dest, CALuint flags, int allowOverlap = 0)
{
	fake.FakeMemcpy(src, dest, event, allowOverlap);
	return(calMemCopy(&fake.event[*event].through, ctx, fake.mem[src].through, fake.mem[dest].through, flags));
}

static inline CALresult calCtxIsEventDone_a(CALcontext ctx, CALevent event)
{
	CALresult retVal = calCtxIsEventDone(ctx, fake.event[event].through);
	if (retVal == CAL_RESULT_OK) fake.QueryEvent(event);
	return(retVal);
}

static inline CALresult calCtxGetMem_a(CALmem* mem, CALcontext ctx, CALresource res)
{
	fake.AddMemHandle(mem);
	return(calCtxGetMem(&fake.mem[*mem].through, ctx, res));
}

static inline CALresult calCtxSetMem_a(CALcontext ctx, CALname name, CALmem mem)
{
	fake.SetMem(name, mem);
	return(calCtxSetMem(ctx, fake.name[name].through, fake.mem[mem].through));
}

static inline CALresult calCtxReleaseMem_a(CALcontext ctx, CALmem mem)
{
	fake.ReleaseMem(mem);
	return(calCtxReleaseMem(ctx, fake.mem[mem].through));
}

static inline CALresult calModuleLoad_a(CALmodule* module, CALcontext ctx, CALimage image)
{
	fake.AddModule(module);
	return(calModuleLoad(&fake.module[*module].through, ctx, image));
}

static inline CALresult calModuleUnload_a(CALcontext ctx, CALmodule module)
{
	fake.UnloadModule(module);
	return(calModuleUnload(ctx, fake.module[module].through));
}

static inline CALresult calModuleGetName_a(CALname* name, CALcontext ctx, CALmodule module, const CALchar* symbolname)
{
	fake.AddName(name, module);
	return(calModuleGetName(&fake.name[*name].through, ctx, fake.module[module].through, symbolname));
}

static inline CALresult calModuleGetEntry_a(CALfunc* func, CALcontext ctx, CALmodule module, const CALchar* symbolname)
{
	fake.GetFunc(func, module);
	return(calModuleGetEntry(&fake.module[module].throughFunc, ctx, fake.module[module].through, symbolname));
}

#define calCtxRunProgram calCtxRunProgram_a
#define calMemCopy calMemCopy_a
#define calCtxIsEventDone calCtxIsEventDone_a
#define calCtxGetMem calCtxGetMem_a
#define calCtxSetMem calCtxSetMem_a
#define calCtxReleaseMem calCtxReleaseMem_a
#define calModuleLoad calModuleLoad_a
#define calModuleUnload calModuleUnload_a
#define calModuleGetName calModuleGetName_a
#define calModuleGetEntry calModuleGetEntry_a

#endif

#endif