all: sender receiver

sender: sender.cu peer.cu utils.h
	nvcc -o sender sender.cu peer.cu -lucp -lucs -lcudart

receiver: receiver.cu peer.cu utils.h
	nvcc -o receiver receiver.cu peer.cu -lucp -lucs -lcudart

clean:
	rm -f sender receiver