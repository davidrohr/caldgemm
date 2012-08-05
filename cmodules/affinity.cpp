#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <WinBase.h>
#else
#include <dirent.h>
#include <syscall.h>
#include <sys/types.h>
#include <sys/syscall.h>
#endif

#include "os_low_level_helper.h"
#include "affinity.h"

#ifndef STD_OUT
#define STD_OUT stdout
#endif

struct threadNameStruct
{
	pid_t thread_id;
	std::string name;
};

static std::vector<threadNameStruct> threadNames;

void setThreadName(char* name)
{
	threadNameStruct tmp;
	tmp.thread_id = gettid();
	tmp.name = name;
	threadNames.push_back(tmp);
}

#ifdef _WIN32

pid_t gettid()
{
	return(0);
}

#else

pid_t gettid()
{
	return((pid_t) syscall(SYS_gettid));
}

void printThreadPinning()
{
	pid_t pid = getpid();
	char dirname[1024];
	sprintf(dirname, "/proc/%d/task", (int) pid);
	DIR* dp = opendir(dirname);
	if (dp)
	{
		dirent* ent;
		fprintf(STD_OUT, "%12s", "");
		for (int i = 0;i < get_number_of_cpu_cores();i++)
		{
			fprintf(STD_OUT, " %2d", i);
		}
		fprintf(STD_OUT, "\n");
		
		while ((ent = readdir(dp)) != NULL)
		{
			pid_t tid = atoi(ent->d_name);
			if (tid != 0)
			{
				fprintf(STD_OUT, "Thread %5d", tid);
				cpu_set_t threadmask;
				sched_getaffinity(tid, sizeof(threadmask), &threadmask);
				for (int i = 0;i < get_number_of_cpu_cores();i++)
				{
					if (CPU_ISSET(i, &threadmask))
					{
						fprintf(STD_OUT, "  X");
					}
					else
					{
						fprintf(STD_OUT, "  .");
					}
				}
				fprintf(STD_OUT, " - ");
				bool found = false;
				for (size_t i = 0;i < threadNames.size();i++)
				{
					if (threadNames[i].thread_id == tid)
					{
						fprintf(STD_OUT, "%s", threadNames[i].name.c_str());
						found = true;
						break;
					}
				}
				if (found == false) fprintf(STD_OUT, "Unknown Thread");
				fprintf(STD_OUT, "\n");
			}
		}
		closedir(dp);
	}
}
#endif
