UCX_PATH ?= /home/jpmedina/opt/ucx

# Use Xlinker to pass linker flags through nvcc
UCX_LDFLAGS = -L$(UCX_PATH)/lib -Xlinker -rpath -Xlinker $(UCX_PATH)/lib
UCX_CPPFLAGS = -I$(UCX_PATH)/include

NVCC = nvcc
LIBS = -lucp -lucs -lcudart

all: peer1 peer2

peer1: peer1.cu utils.h sender.cu sender.cuh ring.cuh ring.cu
	$(NVCC) -o $@ $< sender.cu ring.cu $(UCX_CPPFLAGS) $(UCX_LDFLAGS) $(LIBS)

peer2: peer2.cu utils.h receiver.cu receiver.cuh ring.cuh ring.cu
	$(NVCC) -o $@ $< receiver.cu ring.cu $(UCX_CPPFLAGS) $(UCX_LDFLAGS) $(LIBS)


clean:
	rm -f peer1 peer2

check-ucx:
	@echo "=== Checking UCX Library Path ==="
	@ldd sender | grep libuc || { echo "ERROR: UCX not linked!"; exit 1; }
	@echo "=== Current Linked UCX ==="
	@ldd sender | grep libuc
	@echo "=== Expected UCX Path: $(UCX_PATH)/lib ==="

.PHONY: all clean check-ucx