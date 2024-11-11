run_fp32:
	nvcc -w -O3 -arch=native -use_fast_math -o bin/run llama.cu
	./bin/run models/llama2_7b.bin -i "Hello" -s 42 -n 256

run_fp16:
	nvcc -w -arch=native -O3 -use_fast_math -o bin/run_fp16 llama_fp16.cu
	./bin/run_fp16 models/llama2_7b_fp16.bin -i "Hello" -s 42 -n 256

matmul:
	nvcc -O3 -arch=native -use_fast_math -lcublas -o bin/matmul kernels/matmul.cu
	./bin/matmul

mha:
	nvcc -O3 -arch=native -use_fast_math -o bin/mha kernels/mha.cu
	./bin/mha

rmsnorm:
	nvcc -O3 -arch=native -use_fast_math -o bin/rmsnorm kernels/rmsnorm.cu
	./bin/rmsnorm
