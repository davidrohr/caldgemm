#include "qsem.h"
#include <errno.h>

qSem::qSem(int num)
{
	max = num;
	if (sem_init(&sem, 0, num)) fprintf(stderr, "Error initializing semaphore");
}

qSem::~qSem()
{
	if (sem_destroy(&sem)) fprintf(stderr, "Error destroying semaphore");
}

void qSem::Lock()
{
	if (sem_wait(&sem)) fprintf(stderr, "Error locking semaphore");
}

void qSem::Unlock()
{
	if (sem_post(&sem)) fprintf(stderr, "Error unlocking semaphire");
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
