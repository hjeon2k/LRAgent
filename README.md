# LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents

This is the official implementation of LRAgent which contains code to train LoRA adapters for multi-agent roles and serve multiple LoRA-augmented models with FastChat to reproduce the results in our paper.


## Overview

LRAgent is an efficient KV-cache sharing framework for multi-LoRA LLM agents that shares highly similar base caches induced by the pretrained weights, while keeping lightweight low-rank caches induced by the LoRA adapters.


## 0) Google Custom Search setup

If you want to enable web search during agent execution, create a **Google Programmable Search Engine** and obtain a search engine ID (`cx`) and API key(`key`).

1. Create a project: [https://developers.google.com/custom-search/v1/overview?hl=ko](https://developers.google.com/custom-search/v1/overview?hl=ko)
2. In Google Cloud Console, enable **Custom Search JSON API** for the project.
3. Create an API key: **Cloud Console → APIs & Services → Credentials**.
4. Create a search engine and note its ID (`cx`) and retrieve the API key (`key`).
5. Paste the key and ID into:

   * `Self_Plan/Group_Planning/benchmark_run/utils.py` (line 194 and 195)


## 1) Train LoRA adapters

1. Install the LoRA training environment:

   ```bash
   conda create -n lora python==3.11 && pip install -r requirements_lora.yaml
   ```

   Pinned versions include torch 2.9.0 + CUDA 12.8 wheels, transformers 4.57.1, peft 0.17.1.

2. Launch training:

   ```bash
   bash Scripts/fastchat_lora.sh
   ```

   Set `gpu`, `agents`, `qa`, `model`, and LoRA hyperparameters inside the script.

Outputs are written under:

* `Self_Plan/Train/output/...`


## 2) Runtime setup 

1. Install the runtime environment:

   ```bash
   conda create -n lragent python==3.11 && pip install -r requirements.txt
   ```

2. Install modified source packages in `src/`:

   ```bash
   pip install -e src/fastchat src/transformers src/peft && cd src/flash-attention && python setup.py install
   ```

   You can symlink directly into your site-packages for langchain, e.g.:

   ```bash
   ln -s $(pwd)/src/langchain $CONDA_PREFIX/lib/python3.11/site-packages/langchain
   ```

3. Main modified files and folder for LRAgent:

* `src/transformers/cache_utils.py`
* `src/transformers/models/llama/modeling_llama.py`
* `src/transformers/models/mistral/modeling_mistral.py`
* `src/peft/tuners/lora/layer.py`
* `fastchat/serve/inference.py`
* `csrc/flash_lora_attn`


## 3) Start FastChat 

1. Start the controller:

   ```bash
   python3 -m fastchat.serve.controller
   ```

2. Start the OpenAI-compatible API server:

   ```bash
   python3 -m fastchat.serve.openai_api_server --host localhost --port 8888
   ```

3. Start a multi-model worker (example: 3 LoRA agents):

   ```bash
   CUDA_VISIBLE_DEVICES=0 PEFT_SHARE_BASE_WEIGHTS=true python3 -m fastchat.serve.multi_model_worker \
       --port 31022 --worker http://localhost:31022 \
       --host localhost \
       --model-path Self_Plan/Train/output/llama-3_1-8b/hotpotqa/v2/5e-5/10ep/plan/ \
       --model-names "plan" \
       --model-path Self_Plan/Train/output/llama-3_1-8b/hotpotqa/v3/6e-5/10ep/action/ \
       --model-names "action" \
       --model-path Self_Plan/Train/output/llama-3_1-8b/hotpotqa/v3/6e-5/10ep/reflect/ \
       --model-names "reflect" \
       --max-gpu-memory 40Gib \
       --dtype float16 \
       --num-gpus 1
   ```


## 4) Run end-to-end inference 

Run the HotpotQA benchmark:

```bash
python Self_Plan/Group_Planning/run_eval.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --plan_agent plan \
    --action_agent action \
    --reflect_agent reflect \
    --max_context_len 32768 \
    --task Hotpotqa \
    --task_path Self_Plan/Group_Planning/benchmark_run/data/hotpotqa \
    --save_path Self_Plan/Group_Planning/output/llama-3.1-8b/hotpotqa/output
```

Notes:

* To run on ScienceQA, replace `--task` and `--task_path` accordingly.
* `--profile` writes per-step end-to-end latency for each agent stage into the output directory.


## 5) Run trace-based efficiency benchmark

To reproduce the trace-based throughput/TTFT evaluation from the paper:

```bash
bash Scripts/emulation/run.sh
```

This script iterates over traces with different total sequence lengths and profiles end-to-end latency and TTFT.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lragent2026,
  title={LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents},
  author={Hyesung Jeon and Hyeongju Ha and Jae-Joon Kim},
  journal={arXiv preprint},
  year={2026}
}
```