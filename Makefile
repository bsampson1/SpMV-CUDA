.PHONY: all all-tests test-spmatrix-generator test-spmv-timing run run-all-tests run-test-spmatrix-generator run-test-spmv-timing clean

all:
	nvcc -I./include -o main src/main.cu src/spmv.cu

all-tests:  test-spmatrix-generator test-spmv-timing

test-spmatrix-generator:
	nvcc -I./include -o test-spmatrix-generator src/test-spmatrix-generator.cu src/spmv.cu

test-spmv-timing:
	nvcc -I./include -o test-spmv-timing src/test-spmv-timing.cu src/spmv.cu

run:
	@srun -N1 --gres=gpu:1 ./main

run-all-tests: test-spmatrix-generator test-spmv-timing

run-test-spmatrix-generator:
	@srun -N1 ./test-spmatrix-generator

run-test-spmv-timing:
	@srun -N1 --gres=gpu:1 ./test-spmv-timing

clean:
	$(RM) main test-*
