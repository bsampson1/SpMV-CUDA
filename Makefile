.PHONY: all all-tests test-spmatrix-generator test-spmv-timing test-spmv-correctness run run-all-tests run-test-spmatrix-generator run-test-spmv-timing run-test-spmv-correctness clean

all:
	nvcc -I./include -o main src/main.cu src/spmv.cu

all-tests:  test-spmatrix-generator test-spmv-timing test-spmv-correctness

test-spmatrix-generator:
	nvcc -I./include -o test-spmatrix-generator src/test-spmatrix-generator.cu src/spmv.cu

test-spmv-timing:
	nvcc -I./include -o test-spmv-timing src/test-spmv-timing.cu src/spmv.cu

test-spmv-correctness:
	nvcc -I./include -o test-spmv-correctness src/test-spmv-correctness.cu src/spmv.cu

run:
	@srun -N1 --gres=gpu:1 ./main

run-all-tests: run-test-spmatrix-generator run-test-spmv-timing run-test-spmv-correctness

run-test-spmatrix-generator:
	@srun -N1 ./test-spmatrix-generator

run-test-spmv-timing:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 --gres=gpu:1 ./test-spmv-timing

run-test-spmv-correctness:
	@srun -N1 --gres=gpu:1 ./test-spmv-correctness

clean:
	$(RM) main test-*
