.PHONY: all all-tests test-spmatrix-struct run run-all-tests run-test-spmatrix-struct clean

all:
	nvcc -I./include -o main src/main.cu src/spmv.cu

all-tests: test-spmatrix-struct

test-spmatrix-struct:
	nvcc -I./include -o test-spmatrix-struct src/test-spmatrix-struct.cu src/spmv.cu

run:
	@./main

run-all-tests: run-test-spmatrix-struct 

run-test-spmatrix-struct:
	@./test-spmatrix-struct

clean:
	$(RM) main test-*
