import json

key_to_print = "prompt"

with open("Self_Plan/Group_Planning/output/autoact_test_data/hotpotqa/7b-easy.jsonl", "r") as f:
    for line in f:
        data = json.loads(line.strip())
        print(data.get(key_to_print, f"Key '{key_to_print}' not found"))
        exit()

