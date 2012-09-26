#include "qsem.h"
#include <errno.h>
#include <stdio.h>

qSem::qSem(int num)
{
	max = num;
	if (sem_init(&sem, 0, num)) fprintf(stderr, "Error initializing semaphore");
}

qSem::~qSem()
{
	if (sem_destroy(&sem)) fprintf(stderr, "Error destroying semaphore");
}

int qSem::Lock()
{
	int retVal;
	if ((retVal = sem_wait(&sem))) fprintf(stderr, "Error locking semaphore");
	return(retVal);
}

int qSem::Unlock()
{
	int retVal;
	if ((retVal = sem_post(&sem))) fprintf(stderr, "Error unlocking semaphire");
	return(retVal);
}

int qSem::Trylock()
{
	int retVal = sem_trywait(&sem);
	if (retVal)
	{
		if (errno == EAGAIN) return(EBUSY);
		return(-1);
	}
	return(0);
}
