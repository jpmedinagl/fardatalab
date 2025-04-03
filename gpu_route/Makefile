all: sender receiver

sender: sender.cu utils.h
	nvcc -o sender sender.cu -lucp -lucs -lcudart

receiver: receiver.cu utils.h
	nvcc -o receiver receiver.cu -lucp -lucs -lcudart

clean:
	rm -f sender receiver