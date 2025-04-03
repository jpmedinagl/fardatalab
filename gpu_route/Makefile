# UCX RDMA GPU Peer-to-Peer Makefile

# Compilers
NVCC := nvcc
CC := gcc

# UCX paths (auto-detect)
UCX_HOME ?= $(shell dirname $(shell dirname $(shell which ucx_info 2>/dev/null || echo "/usr")))
CUDA_HOME ?= /usr/local/cuda

# Flags
CFLAGS := -Wall -Wextra -I$(UCX_HOME)/include -I$(CUDA_HOME)/include
NVCCFLAGS := -arch=sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') \
             -Xcompiler -fPIC $(CFLAGS)
LDFLAGS := -L$(UCX_HOME)/lib -L$(CUDA_HOME)/lib64 -lucp -lucs -lcudart

# Targets
TARGETS := peer1 peer2

# Debug vs Release
debug: NVCCFLAGS += -g -G
debug: CFLAGS += -g -DDEBUG
debug: all

release: NVCCFLAGS += -O3
release: CFLAGS += -O2
release: all

all: $(TARGETS)

peer1: peer1.cu peer.cu utils.h
	$(NVCC) $(NVCCFLAGS) -o $@ peer1.cu peer.cu $(LDFLAGS)

peer2: peer2.cu peer.cu utils.h
	$(NVCC) $(NVCCFLAGS) -o $@ peer2.cu peer.cu $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean debug release