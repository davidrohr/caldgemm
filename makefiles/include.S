    .global FILENAMEMOD
    .global FILENAMEMOD_size
    .section .rodata
FILENAMEMOD:
    .incbin "FILENAMENORMAL"
FILETERM:
	
FILENAMEMOD_size:
    .int FILETERM - FILENAMEMOD
