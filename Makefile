CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iinclude -D_POSIX_C_SOURCE=200809L

ifeq ($(OS),Windows_NT)
LDFLAGS = -lws2_32 -lpthread -lm
BIN = ftl_bench.exe
else
LDFLAGS = -lrt -lpthread -lm
BIN = ftl_bench
endif

SRC = src/nand.c src/ftl.c src/ipc_bridge.c src/main.c
OBJ = $(SRC:.c=.o)

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
