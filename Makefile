.PHONY: all all-tests test-spmatrix-generator test-RMSE test-spmv-timing test-spmv-correctness run run-all-tests run-test-spmatrix-generator run-test-RMSE run-test-spmv-timing run-test-spmv-correctness test-spmvStrawberry run-test-spmvStrawberry clean

all:
	nvcc -I./include -o main src/main.cu src/spmv.cu

all-tests:  test-spmatrix-generator test-RMSE test-spmv-correctness test-spmv-timing

test-spmatrix-generator:
	nvcc -I./include -o test-spmatrix-generator src/test-spmatrix-generator.cu src/spmv.cu

test-RMSE:
	nvcc -I./include -o test-RMSE src/test-RMSE.cu src/spmv.cu

test-spmv-correctness:
	nvcc -I./include -o test-spmv-correctness src/test-spmv-correctness.cu src/spmv.cu

test-spmv-timing:
	nvcc -I./include -o test-spmv-timing src/test-spmv-timing.cu src/spmv.cu

test-spmvStrawberry:
	nvcc -I./include -o test-spmvStrawberry src/test-spmvStrawberry.cu src/spmv.cu

run:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 --gres=gpu:1 ./main

run-all-tests: run-test-spmatrix-generator run-test-RMSE run-test-spmv-correctness run-test-spmv-timing

run-test-spmatrix-generator:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 ./test-spmatrix-generator

run-test-RMSE:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 ./test-RMSE

run-test-spmv-correctness:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 --gres=gpu:1 ./test-spmv-correctness

run-test-spmv-timing:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 --gres=gpu:1 ./test-spmv-timing

run-test-spmvStrawberry:
	@srun -p cpeg655 -t 00:20:00 -J spmv -N 1 --gres=gpu:1 ./test-spmvStrawberry

clean:
	$(RM) main test-*
