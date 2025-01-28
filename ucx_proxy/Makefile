FLAGS := -Wall --std=gnu99 -I/home/neptune/csc494/ucx-install/include
PORT := port.mk
TARGETS := ucx_server ucx_client
LIBS := -L/home/neptune/csc494/ucx-install/lib -lucp -lucm -lucs

debug: FLAGS += -ggdb3 -DDEBUG
debug: all

release: FLAGS += -O2
release: all

all: $(PORT) $(TARGETS)

ucx_server: ucx_server.o ucx_util.o
	gcc $(FLAGS) -o $@ $^ $(LIBS)

ucx_client: ucx_client.o ucx_util.o
	gcc $(FLAGS) -o $@ $^ $(LIBS)

%.o: %.c %.h 
	gcc $(FLAGS) -c $< -o $@

$(PORT):
	@echo "Generating a new default port number in $@"
	@awk 'BEGIN{srand();printf("FLAGS += -DDEFAULT_PORT=%d", 55536*rand()+10000)}' > $(PORT)

.PHONY: all clean debug release
clean:
	rm -f *.o ucx_server ucx_client $(PORT)

include $(PORT)
