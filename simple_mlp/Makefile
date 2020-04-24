CC=riscv64-unknown-linux-gnu-gcc

CCFLAGS=-I. -Wno-int-conversion -lm -ggdb -DDRG_DEBUG=1 -static

DEPS = activations_fp.h

SRCS := neural_network_fp.c activations_fp.c test_main_fp.c

OBJS := $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

test_main_fp: $(OBJS)
	$(CC) -o $@ $^ $(CCFLAGS)

.PHONY: clean

clean:
	-$(RM) $(OBJS)
	-$(RM) test_main_fp
