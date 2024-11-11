import argparse
from collections import defaultdict
import torch
import struct
from dataclasses import dataclass


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    vocab_size: int = 32000
    hidden_dim: int = 11008
    norm_eps: float = 1e-5
    max_seq_len: int = 2048


def serialize(file, tensor, dtype):
    """writes one tensor to file that is open in wb mode"""
    torch_cast_dtype = {
        torch.float32: torch.float32,
        torch.float16: torch.int16,
        torch.int8: torch.int8,
    }
    d = tensor.detach().cpu().view(-1).view(torch_cast_dtype[dtype]).numpy()

    struct_format_char = {
        torch.float32: "f",
        torch.float16: "h",
        torch.int8: "b",
    }
    b = struct.pack(f"{len(d)}{struct_format_char[dtype]}", *d)
    file.write(b)


def export_hf_model_to_binary(model_path, export_path, dtype=torch.float32):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    out_file = open(export_path, "wb")
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    hf_dict = hf_model.state_dict()

    # Convert LlamaConfig to ModelArgs
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_attention_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings

    out_file = open(export_path, "wb")
    header = struct.pack(
        "iiiiiii",
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.max_seq_len,
    )
    out_file.write(header)

    # Huggingface permutes WQ and WK, this function reverses it
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2).contiguous()

    model_weights = defaultdict(list)
    model_weights["embed_tokens_weight"] = [hf_dict["model.embed_tokens.weight"]]
    for i in range(config.n_layers):
        model_weights["input_layernorm"].append(hf_dict[f"model.layers.{i}.input_layernorm.weight"])
        model_weights["self_attn_q"].append(permute_reverse(hf_dict[f"model.layers.{i}.self_attn.q_proj.weight"]))
        model_weights["self_attn_k"].append(permute_reverse(hf_dict[f"model.layers.{i}.self_attn.k_proj.weight"]))
        model_weights["self_attn_v"].append(hf_dict[f"model.layers.{i}.self_attn.v_proj.weight"])
        model_weights["self_attn_o"].append(hf_dict[f"model.layers.{i}.self_attn.o_proj.weight"])
        model_weights["post_attn_layernorm"].append(hf_dict[f"model.layers.{i}.post_attention_layernorm.weight"])
        model_weights["mlp_gate"].append(hf_dict[f"model.layers.{i}.mlp.gate_proj.weight"])
        model_weights["mlp_down"].append(hf_dict[f"model.layers.{i}.mlp.down_proj.weight"])
        model_weights["mlp_up"].append(hf_dict[f"model.layers.{i}.mlp.up_proj.weight"])

    model_weights["final_norm_weight"] = [hf_dict["model.norm.weight"]]
    model_weights["lm_head_weight"] = [hf_dict["lm_head.weight"]]

    for layer_name, layer in model_weights.items():
        print(f"Writing {layer_name} weights, dtype {layer[0].dtype}")
        for w in layer:
            serialize(out_file, w, dtype)

    # write to binary file
    out_file.close()
    print(f"Wrote {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", type=str, help="huggingface model path")
    parser.add_argument("--out", type=str, help="the output filepath")
    parser.add_argument("--dtype", type=str, default="fp32", help="the model datatype")
    args = parser.parse_args()
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "int8": torch.int8,
    }[args.dtype]

    export_hf_model_to_binary(args.hf, args.out, dtype)
