python Self_Plan/Group_Planning/run_eval.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --plan_agent plan \
    --tool_agent action \
    --reflect_agent reflect \
    --max_context_len 32768 \
    --task Hotpotqa \
    --task_path Self_Plan/Group_Planning/benchmark_run/data/hotpotqa \
    --save_path Self_Plan/Group_Planning/output/llama-3_1-8b/hotpotqa/original



python Self_Plan/Group_Planning/run_eval.py \
    --agent_name ZeroshotThink_ScienceQA_run_Agent \
    --plan_agent plan \
    --action_agent action \
    --reflect_agent reflect \
    --max_context_len 32768 \
    --task Scienceqa \
    --task_path Self_Plan/Group_Planning/benchmark_run/data/scienceqa \
    --save_path Self_Plan/Group_Planning/output/llama-3_1-8b/scienceqa/original
