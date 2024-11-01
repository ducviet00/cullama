# WORKLOG

## 2024-11-01

Hello November 👋

I created a script to convert weights from the Huggingface format to binary based on Karpathy's [llama2.c](https://github.com/karpathy/llama2.c). This is needed to create FP16 weights and quantized weights for further use.

## 2024-10-31

Achieved `meta-llama/Llama-2-7b-hf fp32`: `44.642857` tok/s (on A100). 2060 cannot fit a 7b fp32 :)

Learn alot from [ankan-ban's implementation](https://github.com/ankan-ban/llama2.cu)
