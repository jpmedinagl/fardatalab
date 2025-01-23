FLAGS := -Wall --std=gnu99 -I/home/juanpmg/ucx-install/include
PORT := port.mk
TARGETS := ucx_server ucx_client

debug: FLAGS += -ggdb3 -DDEBUG
debug: all

release: FLAGS += -O2
release: all

all: $(PORT) $(TARGETS)

ucx_server: ucx_server.o
	gcc $(FLAGS) -o $@ $^

ucx_client: ucx_client.o
	gcc $(FLAGS) -o $@ $^

%.o: %.c %.h 
	gcc $(FLAGS) -c $< -o $@

$(PORT):
	@echo "Generating a new default port number in $@"
	@awk 'BEGIN{srand();printf("FLAGS += -DDEFAULT_PORT=%d", 55536*rand()+10000)}' > $(PORT)

.PHONY: all clean debug release
clean:
	rm -f *.o ucx_server ucx_client $(PORT)

include $(PORT)
