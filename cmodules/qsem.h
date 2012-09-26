#include <semaphore.h>

class qSem
{
public:
	qSem(int num = 1);
	~qSem();

	void Lock();
	void Unlock();
	int Trylock();

private:
	int max;
	sem_t sem;
};
