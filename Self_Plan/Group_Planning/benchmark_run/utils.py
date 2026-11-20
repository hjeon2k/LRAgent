"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import joblib
import csv
import json
import requests

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    not_finish = [a for a in agents if not a.is_finished()]
    return correct, incorrect, not_finish

def remove_fewshot(prompt: str) -> str:
    prefix = prompt.split('Here are some examples:')[0]
    suffix = prompt.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n' +  suffix.strip('\n').strip()

def log_trial(agents, trial_n):
    correct, incorrect, not_finish = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)} , Not Finished: {len(not_finish)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        # log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        # log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN NOT_FINISH AGENTS -----------\n\n'
    for agent in not_finish:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    return log

def summarize_trial_detailed(agents):
    correct = [a.is_correct() for a in agents]
    reward = [a.reward()[0] for a in agents]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    error = [a.run_error for a in agents]
    return correct, reward, error, halted, incorrect

def log_agent(agent, file_path):
    question = agent.question
    g_truth = agent.key
    correct = agent.is_correct()
    reward = agent.reward()[0]
    halted = agent.is_halted()
    error = agent.run_error
    prompt = agent._build_agent_prompt()
    save_dict = {"question":question, "answer":g_truth, "correct":correct, "reward":reward,
                 "halted":halted, "error":error,"prompt":prompt}
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        json.dump(save_dict, f)
        f.write("\n")


def log_profile(agent, e2e_time, file_path):
    data = [[t[-1], t[0], s[0], s[1], s[2]] for (t, s) in zip(agent.profile_t, agent.profile_s)] + [["e2e", e2e_time, 0, 0, 0]]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)


def get_all_agent_sessions(file_name):
    sessions = []
    with open(file_name) as f:
        for line in f:
            session = json.loads(line)
            sessions.append(session)
    return sessions

def get_error_tasks(sessions):
    error_tasks = []
    for sess in sessions:
        if sess["error"]:
            task = (sess["question"], sess["answer"])
            error_tasks.append(task)
    error_tasks = list(set(error_tasks))
    return error_tasks

def get_non_error_tasks(sessions):
    tasks = []
    for sess in sessions:
        if not sess["error"]:
            task = (sess["question"], sess["answer"])
            tasks.append(task)
    tasks = list(set(tasks))
    return tasks

def delete_error(file_name):
    sessions = get_all_agent_sessions(file_name)
    non_error_sessions = [sess for sess in sessions if not sess["error"]]
    with open(file_name+'.back', 'a') as b_f:
        for sess in sessions:
            json.dump(sess, b_f)
            b_f.write('\n')
    with open(file_name, 'w') as f:
        for sess in non_error_sessions:
            json.dump(sess, f)
            f.write('\n')

def summarize_react_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted

def summarize_react_trial_detailed(agents):
    correct = [a.is_correct() for a in agents]
    reward = [a.reward()[0] for a in agents]
    return correct, reward

def log_react_trial(agents, trial_n):
    correct, incorrect, halted = summarize_react_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN HALTED AGENTS -----------\n\n'
    for agent in halted:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    return log

def save_agents(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        agent.enc = None
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))

def load_agents(dir:str):
    import tiktoken
    agents = []
    for f in os.listdir(dir):
        agent = joblib.load(os.path.join(dir, f))
        agent.enc = tiktoken.encoding_for_model("text-davinci-003")
        agents.append(agent)
    return agents

def _validate_server(address):
    if not address:
        raise ValueError('Must provide a valid server for search')
    if address.startswith('http://') or address.startswith('https://'):
        return address
    PROTOCOL = 'http://'
    print(f'No protocol provided, using "{PROTOCOL}"')
    return f'{PROTOCOL}{address}'

def parse_result(result, field="snippet"):
    return [
        item[field]
        for item in result
        if isinstance(item, dict) and isinstance(item.get(field), str) and item[field]
    ]


def call_web_search(query, count: int = 5):
    provider = os.environ.get("LRAGENT_SEARCH_PROVIDER", "google")
    query = query.strip('"').strip("'")
    headers = {}

    if provider == "google":
        key = os.environ.get("GOOGLE_API_KEY")
        cx = os.environ.get("GOOGLE_CSE_ID")
        if not key or not cx:
            return None
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": key, "cx": cx, "q": query, "num": count}
        field = "snippet"
    elif provider == "brave":
        key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if not key:
            return None
        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": count}
        headers["X-Subscription-Token"] = key
        field = "description"
    else:
        raise ValueError("LRAGENT_SEARCH_PROVIDER must be google or brave")

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        results = response.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(results, dict):
        return None
    if provider == "brave":
        results = results.get("web")
        if not isinstance(results, dict):
            return None
        items = results.get("results")
    else:
        items = results.get("items")

    return parse_result(items, field) if isinstance(items, list) else None
