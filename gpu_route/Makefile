UCX_PATH ?= /home/jpmedina/opt/ucx

# Use Xlinker to pass linker flags through nvcc
UCX_LDFLAGS = -L$(UCX_PATH)/lib -Xlinker -rpath -Xlinker $(UCX_PATH)/lib
UCX_CPPFLAGS = -I$(UCX_PATH)/include

all: sender receiver

sender: sender.cu utils.h
	nvcc -o sender sender.cu $(UCX_CPPFLAGS) $(UCX_LDFLAGS) -lucp -lucs -lcudart

receiver: receiver.cu utils.h
	nvcc -o receiver receiver.cu $(UCX_CPPFLAGS) $(UCX_LDFLAGS) -lucp -lucs -lcudart

clean:
	rm -f sender receiver

check-ucx:
	@echo "=== Checking UCX Library Path ==="
	@ldd sender | grep libuc || { echo "ERROR: UCX not linked!"; exit 1; }
	@echo "=== Current Linked UCX ==="
	@ldd sender | grep libuc
	@echo "=== Expected UCX Path: $(UCX_PATH)/lib ==="