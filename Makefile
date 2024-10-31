run:
	nvcc -O3 -arch=native -use_fast_math  -o bin/run llama.cu