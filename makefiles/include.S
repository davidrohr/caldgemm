    .global FILENAMEMOD
    .global FILENAMEMOD_size
    .section .rodata
FILENAMEMOD:
    .incbin "FILENAMENORMAL"
1:
FILENAMEMOD_size:
    .int 1b - FILENAMEMOD
