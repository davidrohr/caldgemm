extern void **ddi_interface;

void fixCal()
{
    unsigned char *func = ddi_interface[0xa8/8];
    func += 0x7fffe591b631 - 0x7fffe591b560;
    if (func[0] == 0x74) {
        func[0] = 0xeb;
        fprintf(stderr, "Replaced je with jmpq\n");
    } else {
        fprintf(stderr, "Did not find je at the expected position\n");
    }
}
void fixCal()
{
    fprintf(stderr, "x\n");
    unsigned char *foo = (unsigned char *)(&calCtxRunProgram);
    unsigned char **bar = *(unsigned char ***)((size_t)(*(unsigned int *)(foo + 2)) + foo + 6);
    fprintf(stderr, "bar = %p, ddi_interface[?] = %p\n", bar,
            bar + (0x10f588 - 0x4220)/sizeof(void*));
    unsigned char *func = *(bar + (0x10f588 - 0x4220)/sizeof(void*));
    func += 0x7fffe591b631 - 0x7fffe591b560;
    fprintf(stderr, "Read jump\n");
    if (func[0] == 0x74) {
        fprintf(stderr, "Replace je with jmpq\n");
        func[0] = 0xeb;
        fprintf(stderr, "Replaced je with jmpq\n");
    } else {
        fprintf(stderr, "Did not find je at the expected position\n");
    }
}
