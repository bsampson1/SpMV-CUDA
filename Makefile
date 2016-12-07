.PHONY: all all-tests test-spmatrix-struct test-spmatrix-generator run run-all-tests run-test-spmatrix-struct run-test-spmatrix-generator clean

all:
	nvcc -I./include -o main src/main.cu src/spmv.cu

all-tests: test-spmatrix-struct test-spmatrix-generator

test-spmatrix-struct:
	nvcc -I./include -o test-spmatrix-struct src/test-spmatrix-struct.cu src/spmv.cu

test-spmatrix-generator:
	nvcc -I./include -o test-spmatrix-generator src/test-spmatrix-generator.cu src/spmv.cu

run:
	@srun -N1 --gres=gpu:1 ./main

run-all-tests: run-test-spmatrix-struct test-spmatrix-generator

run-test-spmatrix-struct:
	@./test-spmatrix-struct

run-test-spmatrix-generator:
	@./test-spmatrix-generator

clean:
	$(RM) main test-*
