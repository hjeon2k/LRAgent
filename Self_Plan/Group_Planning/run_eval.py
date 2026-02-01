import os
import argparse
import numpy as np
import pandas as pd
import concurrent
import joblib
from benchmark_run.utils import summarize_trial_detailed, log_trial
import benchmark_run.utils as utils
from benchmark_run.Meta_agent_arch import get_agent
from benchmark_run.llms import get_llm_backend
from benchmark_run.config import available_agent_names
import json
import torch
import time

# SEED=42
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="ZeroshotThink_HotPotQA_run_Agent")
parser.add_argument("--plan_agent", type=str, help="Name of the plan", default="plan_peft")
parser.add_argument("--action_agent", type=str, help="Name of the action", default="action_peft")
parser.add_argument("--reflect_agent", type=str, help="Name of the reflect_agent", default="reflect_peft")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=4096)
parser.add_argument("--task",type=str ,help="task name",default="Hotpotqa")
parser.add_argument("--task_path",type=str,help="task path")
parser.add_argument("--save_path",type=str,help="save path")
parser.add_argument("--profile",action="store_true",help="profiler on off", default=False)
args = parser.parse_args()

agent_name = args.agent_name

plan_agent = args.plan_agent
action_agent = args.action_agent
reflect_agent = args.reflect_agent
max_context_len = args.max_context_len
save_path = args.save_path
task_path = args.task_path
assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def process_agent_clear_kvcache(llms):
    for llm in llms:
        llm.clear_kvcache()

def run_one_complex_level_hotpotqa(level="easy", test_num=20, profile=False):
    hotpot = joblib.load(f'{task_path}/{level}.joblib').reset_index(drop = True)
    for i in range(test_num):
        print(f"{i} / {test_num} test")
        agent_save_file = f"{save_path}/{level}/{i}.jsonl"
        agent_profile_file = f"{save_path}/profile/{level}/profile.csv"
        task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
        if os.path.exists(agent_save_file):
            sessions = utils.get_all_agent_sessions(agent_save_file)
            completed_tasks = utils.get_non_error_tasks(sessions)
            print(f"{level}:{len(completed_tasks)}")
            task_instructions = [task for task in task_instructions if task not in completed_tasks]
            utils.delete_error(agent_save_file)

        llm_plan = get_llm_backend(plan_agent)
        llm_action = get_llm_backend(action_agent)
        llm_reflect = get_llm_backend(reflect_agent)

        agent_cls = get_agent(agent_name)
        agents = [agent_cls(ques, ans, llm_plan.run, llm_action.run, llm_reflect.run, max_context_len) for ques, ans in task_instructions]
        for agent in agents:
            e2e_s = time.perf_counter()
            process_agent_run_step(agent)
            e2e_e = time.perf_counter()
            process_agent_clear_kvcache([llm_plan, llm_action, llm_reflect])
            utils.log_agent(agent, agent_save_file)
            if profile:
                utils.log_profile(agent, e2e_e - e2e_s, agent_profile_file)
        print(f'Finished Trial {level}. Total: {len(agents)}')

def run_one_complex_level_scienceqa(level="1-4", test_num=20, profile=False):
    scienceqa = json.load(open(f'{task_path}/format_scienceqa_grade{level}.json'))
    for i in range(test_num):
        print(f"{i} / {test_num} test")
        agent_save_file = f"{save_path}/{level}/{i}.jsonl"
        agent_profile_file = f"{save_path}/profile/{level}/profile.csv"
        task_instructions = [(row['Question'],row['choices'],row['Answer'],row['orc'],row['caption']) for row in scienceqa]
        if os.path.exists(agent_save_file):
            sessions = utils.get_all_agent_sessions(agent_save_file)
            completed_tasks = utils.get_non_error_tasks(sessions)
            print(f"{level}:{len(completed_tasks)}")
            task_instructions = [task for task in task_instructions if task[0] not in completed_tasks]
            utils.delete_error(agent_save_file)

        llm_plan = get_llm_backend(plan_agent)
        llm_action = get_llm_backend(action_agent)
        llm_reflect = get_llm_backend(reflect_agent)

        agent_cls = get_agent(agent_name)
        agents = [agent_cls(ques, choices, ans, caption, orc, llm_plan.run, llm_action.run, llm_reflect.run, max_context_len) for ques,choices,ans,orc,caption in task_instructions]
        for agent in agents:
            e2e_s = time.perf_counter()
            process_agent_run_step(agent)
            e2e_e = time.perf_counter()
            process_agent_clear_kvcache([llm_plan, llm_action, llm_reflect])
            utils.log_agent(agent, agent_save_file)
            if profile:
                utils.log_profile(agent, e2e_e - e2e_s, agent_profile_file)
        print(f'Finished Trial. Total: {len(agents)}')

def main():
    test_num = 20 if not args.profile else 10
    if args.task == "Hotpotqa":
        levels = ['hard', 'medium', 'easy']
        for level in levels:
            os.makedirs(f"{save_path}/{level}", exist_ok=True)
            os.makedirs(f"{save_path}/profile/{level}", exist_ok=True)
            run_one_complex_level_hotpotqa(level, test_num, args.profile)
    elif args.task == "Scienceqa":
        levels = ['9-12', '5-8', '1-4']
        for level in levels:
            os.makedirs(f"{save_path}/{level}", exist_ok=True)
            os.makedirs(f"{save_path}/profile", exist_ok=True)
            run_one_complex_level_scienceqa(level, test_num, args.profile)
if __name__ == '__main__':
    main()
