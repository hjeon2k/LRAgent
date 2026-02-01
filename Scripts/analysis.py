import os
import csv
import time
import math
import json
import jsonlines
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

import pandas as pd
import matplotlib.pyplot as plt

def analyze_accuracy_results(file_path):
    all_correct = 0
    all_wrong = 0
    right_reflect_wrong = 0
    wrong_reflect_wrong = 0
    reward = 0
    all_row = 0
    with open(file_path, "r") as f:
        for item in jsonlines.Reader(f):
            all_row += 1
            reward += item["reward"]
            if item["correct"]:
                all_correct += 1
                if "Reflect[wrong]" in item["prompt"]:
                    right_reflect_wrong += 1
            else:
                all_wrong += 1
                if "Reflect[wrong]" in item["prompt"]:
                    wrong_reflect_wrong += 1
    print(f'1. Accuracy: {all_correct/all_row}')
    print(f'2. Reflect wrong: {(right_reflect_wrong+wrong_reflect_wrong)/all_row}  Reflect right: {1-(right_reflect_wrong+wrong_reflect_wrong)/all_row}')
    print(f'3. Correct answers - Reflect right: {1-(right_reflect_wrong/all_correct)}  Wrong: {right_reflect_wrong/all_correct}')
    print(f'4. Wrong answers - Reflect right: {1-(wrong_reflect_wrong/all_wrong)}  Wrong: {wrong_reflect_wrong/all_wrong}')
    print(f'5. Reward: {reward}')


def analyze_latency_results(file_path: str):
    df = pd.read_csv(
        file_path,
        header=None,
        names=["type", "time", "prefill_len", "total_len", "decode_len"],
    )

    # ------------------------------------------------------------------
    # Precompute distance-from-last-e2e per row (for step coloring)
    # ------------------------------------------------------------------
    distances_from_last_e2e = []
    last_e2e_idx = None
    for idx in df.index:
        if df.loc[idx, "type"] == "e2e":
            last_e2e_idx = idx
            distances_from_last_e2e.append(0)
        else:
            if last_e2e_idx is None:
                # No e2e seen yet; treat as distance 0
                distances_from_last_e2e.append(0)
            else:
                distances_from_last_e2e.append(idx - last_e2e_idx)

    df["dist_from_last_e2e"] = distances_from_last_e2e

    # ------------------------------------------------------------------
    # (1) Plot non-e2e rows: time vs total_len,
    #     colored by dist_from_last_e2e, marker by type
    # ------------------------------------------------------------------
    non_e2e = df[df["type"] != "e2e"]

    plt.figure()

    # Marker mapping per type
    marker_map = {
        "plan": "o",     # dot
        "action": "^",   # triangle
        "reflect": "s",  # square
    }

    sc = None
    for t, marker in marker_map.items():
        sub = non_e2e[non_e2e["type"] == t]
        if sub.empty:
            continue
        sc = plt.scatter(
            sub["total_len"],
            sub["time"],
            c=sub["dist_from_last_e2e"],
            cmap="magma_r",
            vmin=0,
            vmax=20,   # max distance ~20, min 0
            marker=marker,
            label=t,
            alpha=0.7,
            edgecolors="face",   # outline uses the same heatmap color
        )

    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Model Latency (s)")
    plt.title("Model Latency to Sequence Length")

    plt.grid(True, alpha=0.3)

    # Colorbar with integer ticks at step length 0,5,10,15,20
    if sc is not None:
        cbar = plt.colorbar(sc, label="step index")
        cbar.set_ticks(np.arange(0, 21, 5))

    # Optional: legend for marker types, with black-filled legend markers
    legend = plt.legend(title="type")
    for lh in legend.legend_handles:
        lh.set_facecolor('black')
        lh.set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), "step.png"), dpi=400)
    plt.close()

    # Plot reflect
    sub = non_e2e[non_e2e["type"] == "reflect"]
    sc = plt.scatter(
        sub["total_len"],
        sub["time"],
        c=sub["dist_from_last_e2e"],
        cmap="magma_r",
        vmin=0,
        vmax=20,   # max distance ~20, min 0
        marker=marker,
        label=t,
        alpha=0.8,
        edgecolors="face",   # outline uses the same heatmap color
    )

    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Model Latency (s)")
    plt.title("Model Latency to Sequence Length")

    plt.grid(True, alpha=0.3)

    # Colorbar with integer ticks at step length 0,5,10,15,20
    if sc is not None:
        cbar = plt.colorbar(sc, label="step index")
        cbar.set_ticks(np.arange(0, 21, 5))

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(file_path), "reflect.png"), dpi=400)
    plt.close()

    # ------------------------------------------------------------------
    # (2) For each e2e row, compute ratio:
    #     seg_sum_time / e2e_time
    #     and last_seqlen_before_e2e = sum of last two total_len values before e2e
    # ------------------------------------------------------------------
    e2e_indices = df.index[df["type"] == "e2e"].tolist()

    e2e_times = []
    segment_sums = []
    e2e_ratios = []
    last_seqlen_before_e2e = []
    block_lengths = []  # number of rows in this block (since last e2e)

    prev_e2e_idx = -1
    for idx in e2e_indices:
        e2e_time = df.loc[idx, "time"]

        seg_start = prev_e2e_idx + 1
        seg_end = idx - 1

        if seg_start <= seg_end:
            segment = df.iloc[seg_start:idx]
            seg_sum_time = segment["time"].sum()

            # last seq len = sum of last two total_len before e2e (or just last if only one)
            last_seq = segment["total_len"].sum()

            block_len = seg_end - seg_start + 1  # number of rows in this block
        else:
            # No preceding rows (edge case)
            seg_sum_time = 0.0
            last_seq = 0.0
            block_len = 0

        e2e_times.append(e2e_time)
        segment_sums.append(seg_sum_time)
        if seg_sum_time > 0:
            # KEEPING YOUR ORIGINAL RATIO DEFINITION
            e2e_ratios.append(seg_sum_time / e2e_time)
        else:
            e2e_ratios.append(float("nan"))

        last_seqlen_before_e2e.append(last_seq)
        block_lengths.append(block_len)

        prev_e2e_idx = idx

    # ------------------------------------------------------------------
    # (3) Plot e2e_time vs seqlen, colored by number of rows in each block
    #     seqlen = sum of last two total_len rows before e2e
    # ------------------------------------------------------------------
    if e2e_times:
        x_slen = last_seqlen_before_e2e
        y_time = e2e_times
        colors = block_lengths
        vmax_blocks = max(block_lengths) if block_lengths else 1

        plt.figure()
        sc2 = plt.scatter(
            x_slen,
            y_time,
            c=colors,
            cmap="magma_r",
            vmin=0,
            vmax=vmax_blocks,
            alpha=0.8,
            edgecolors="face",   # outline uses the same heatmap color
        )

        plt.xlabel("Total sequence length (tokens)")
        plt.ylabel("E2E Latency (s)")
        plt.title("E2E Latency to Total Sequence Length")
        plt.grid(True, alpha=0.3)

        cbar2 = plt.colorbar(sc2, label="#steps")

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(file_path), "e2e.png"), dpi=400)
        plt.close()
    else:
        print("No e2e rows found to plot.")

    # ------------------------------------------------------------------
    # (4) Averages: e2e time, seg_sum_time, ratio, last_seqlen_before_e2e
    # ------------------------------------------------------------------
    valid_ratios = [
        r for r in e2e_ratios
        if not (isinstance(r, float) and math.isnan(r))
    ]

    avg_e2e_time = sum(e2e_times) / len(e2e_times) if e2e_times else 0.0
    avg_seg_sum_time = sum(segment_sums) / len(segment_sums) if segment_sums else 0.0
    avg_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float("nan")
    ratio_of_avgs = (
        avg_seg_sum_time / avg_e2e_time if avg_e2e_time > 0 else float("inf")
    )
    avg_last_seqlen = (
        sum(last_seqlen_before_e2e) / len(last_seqlen_before_e2e)
        if last_seqlen_before_e2e
        else 0.0
    )

    avg_steps_per_block = (sum(block_lengths) / len(block_lengths)) if block_lengths else float('nan')

    print("\n=== e2e summary (averages) ===")
    print(f"Number of datapoints: {len(e2e_times)}")
    print(f"Average seg_sum time:                {avg_seg_sum_time:.6f}")
    print(f"Average e2e time:                    {avg_e2e_time:.6f}")
    print(f"Average ratio of (seg_sum / e2e):    {avg_ratio:.6f}")
    print(f"Average total sequence length:       {avg_last_seqlen:.6f}")
    print(f"Average #steps per block:            {avg_steps_per_block:.2f}")


def emulate_latency_results(base_model_name, file_path, e2e):
    def setup_multi_lora_model():
        print(f"Loading Base Model: {base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto", 
            attn_implementation="flash_attention_2",
        )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(base_model, peft_config, adapter_name="plan")
        model.add_adapter("action", peft_config)
        model.add_adapter("reflect", peft_config)

        for name, module in model.named_modules():
            if "v_proj" in name.split(".")[-1]:
                module.lrcache = True
                for adapter in module.lora_A.keys():
                    module.lora_A[adapter] = module.lora_A[adapter].to(torch.float32)
                    module.lora_B[adapter] = module.lora_B[adapter].to(torch.float32)

        return model.cuda(), tokenizer

    def emulate_step(model, tokenizer, input_len, gen_len, past_key_values, past_lr_caches):
        """
        One *row* in a block:
          - Append `input_len` new tokens to the existing context (represented by past_key_values).
          - Then decode `gen_len` new tokens autoregressively.
        `past_key_values` is shared across rows within the same block.
        """
        input_ids = torch.randint(
            0,
            tokenizer.vocab_size,
            (1, input_len),
            device=model.device,
            dtype=torch.long,
        )

        torch.cuda.synchronize()
        fwd_s = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                past_lr_caches=past_lr_caches,
            )
            past_key_values = outputs.past_key_values
            past_lr_caches = outputs.past_lr_caches
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            if not e2e:
                torch.cuda.synchronize()
                fwd_e = time.time()
                torch.cuda.synchronize()
            for _ in range(gen_len):
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    past_lr_caches=past_lr_caches,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                past_lr_caches = outputs.past_lr_caches
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        if e2e:
            torch.cuda.synchronize()
            fwd_e = time.time()
            torch.cuda.synchronize()

        return fwd_e - fwd_s, past_key_values, past_lr_caches

    model, tokenizer = setup_multi_lora_model()
    model.eval()

    print("\n--- Starting Latency Emulation ---\n")
    print(f"Loading data from: {file_path}")
    print("-" * 75)

    try:
        with open(file_path, "r", newline="") as f:
            csv_rows = list(csv.reader(f))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please ensure the CSV file exists.")
        return

    new_rows = []
    past_key_values = None
    past_lr_caches = None
    cycle_latency = 0.0
    for i in tqdm(range(1), desc="Emulating latency"):
        for row in tqdm(csv_rows, desc=f"Emulating latency {i}", total=len(csv_rows), leave=False):
            if not row:
                new_rows.append(row)
                continue

            agent_type = row[0].strip()

            if agent_type == "e2e":
                past_key_values = None
                past_lr_caches = None
                e2e_row = row.copy()
                e2e_row[1] = f"{cycle_latency:.6f}"
                new_rows.append(e2e_row)
                cycle_latency = 0.0
                continue

            input_len = int(row[2])     # new tokens added in this step
            gen_len = int(row[4])

            model.set_adapter(agent_type)
            emulated_time, past_key_values, past_lr_caches = emulate_step(
                model, tokenizer, input_len, gen_len, past_key_values, past_lr_caches
            )

            new_row = row.copy()
            new_row[1] = f"{emulated_time:.6f}"
            cycle_latency += emulated_time
            new_rows.append(new_row)

    base, ext = os.path.splitext(file_path)
    out_path = base + f"_{base_model_name.split('/')[-1]}_{'e2e' if e2e else 'prefill'}.csv"
    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerows(new_rows)

    print(f"\nEmulated CSV written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze results from JSONL file.")
    parser.add_argument("--profile_type", "-t", default='l', choices=['a', 'l', 'e'], help="Profile type: accuracy or latency or emulate.")
    parser.add_argument("--file_path", "-f", default='./', help="Path to the directory containing JSONL, CSV, or directory.")
    parser.add_argument("--base_model_name", "-m", default='meta-llama/Llama-3.1-8B-Instruct', help="Base model name.")
    parser.add_argument("--e2e", "-e", action="store_true", help="Whether to emulate e2e latency.")
    args = parser.parse_args()

    if args.profile_type == 'a':
        analyze_accuracy_results(args.file_path)
    elif args.profile_type == 'l':
        analyze_latency_results(args.file_path)
    elif args.profile_type == 'e':
        emulate_latency_results(args.base_model_name, args.file_path, args.e2e)


if __name__ == "__main__":
    main()

