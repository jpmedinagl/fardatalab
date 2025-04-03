all: sender receiver

LDFLAGS = -L/usr/local/cuda/lib64 -lucp -lucs -lcudart -lcuda

sender: sender.cu utils.h
	nvcc -o sender sender.cu $(LDFLAGS)

receiver: receiver.cu utils.h
	nvcc -o receiver receiver.cu $(LDFLAGS)

clean:
	rm -f sender receiver