#ifdef __cplusplus
extern "C"
{
#endif

void setThreadName(char* name);
void printThreadPinning();
pid_t gettid();

#ifdef __cplusplus
}
#endif
