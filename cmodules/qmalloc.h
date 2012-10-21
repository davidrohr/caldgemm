#include <stdlib.h>

#ifndef QMALLOC_H
#define QMALLOC_H

class qmalloc
{
public:
	static void* qMalloc(size_t size, bool huge, bool executable, bool locked, void* alloc_addr = NULL, int interleave = false);
	static int qFree(void* ptr);

private:	
	static int qMallocCount;
	static int qMallocUsed;
	struct qMallocData
	{
		void* addr;
		size_t size;
	};
	static qMallocData* qMallocs;
};

#endif
