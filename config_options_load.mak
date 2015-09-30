include config_options.mak

ifeq ($(AMDAPPSDKROOT), )
ifeq ($(INCLUDE_CAL), 1)
$(warning CAL not found, disabling INCLUDE_CAL)
endif
INCLUDE_CAL				= 0
endif

ifeq ("$(CUDA_PATH)", "")
ifeq ($(INCLUDE_CUDA), 1)
$(warning CUDA not found, disabling INCLUDE_CUDA)
endif
INCLUDE_CUDA				= 0
INCLUDE_CUBLAS				= 0
endif
