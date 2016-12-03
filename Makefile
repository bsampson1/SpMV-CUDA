.PHONY: all test run run-test clean

all:
	nvcc -I./include -o main src/main.cu src/spmv.cu

test:
	nvcc -I./include -o test src/test.cu src/spmv.cu

run:
	@./main

run-test:
	@./test

clean:
	$(RM) main test
