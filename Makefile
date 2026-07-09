CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iinclude -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lrt -lpthread -lm

SRC = src/nand.c src/ftl.c src/ipc_bridge.c src/main.c
OBJ = $(SRC:.c=.o)
BIN = ftl_bench

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)
	rm -rf results/*.csv

run-baselines: $(BIN)
	mkdir -p results
	./$(BIN)

run-rl: $(BIN)
	mkdir -p results
	./$(BIN) --rl

.PHONY: all clean run-baselines run-rl
