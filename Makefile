.PHONY: all test run run-test clean

all:
	nvcc -I./include -o bin/main src/main.cu src/spmv.cu

test:
	nvcc -I./include -o bin/test test/test.cu src/spmv.cu

run:
	@./bin/main

run-test:
	@./bin/test

clean:
	$(RM) bin/*
