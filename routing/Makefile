# Universal Makefile for UCX+CUDA

# Compilers
CC := gcc
NVCC := nvcc

# Auto-detect UCX paths from ucx_info location
UCX_PREFIX := $(shell dirname $(shell dirname $(shell which ucx_info)))
UCX_INCLUDE := $(UCX_PREFIX)/include
UCX_LIB := $(UCX_PREFIX)/lib

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
    $(warning Could not detect GPU architecture, defaulting to sm_70)
    GPU_ARCH := 70
endif

# Flags
CFLAGS := -Wall --std=gnu99 -I$(UCX_INCLUDE)
NVCCFLAGS := -arch=sm_$(GPU_ARCH) -Xcompiler -fPIC -I$(UCX_INCLUDE)
LDFLAGS := -L$(UCX_LIB) -lucp -lucm -lucs
CUDA_LDFLAGS := -lcudart

PORT := port.mk
TARGETS := server client

# Debug vs Release
debug: CFLAGS += -ggdb3 -DDEBUG
debug: NVCCFLAGS += -g -G
debug: all

release: CFLAGS += -O2
release: NVCCFLAGS += -O3
release: all

all: $(PORT) $(TARGETS)

# GPU targets
server: server.o util.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

client: client.o util.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

# Compilation rules
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(PORT):
	@echo "Generating default port number"
	@awk 'BEGIN{srand();printf("CFLAGS += -DDEFAULT_PORT=%d", 55536*rand()+10000)}' > $@

.PHONY: all clean debug release
clean:
	rm -f *.o $(TARGETS) $(PORT)

include $(PORT)