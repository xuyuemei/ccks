import json
import re

# , "Dedication", "Integrity", "Friendliness"
value_list = ["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law", "Patriotism", "Dedication", "Integrity", "Friendliness"]
# value_list = ["Prosperity"]
for value in value_list:
    #file_path = f'weight_{value}.perturb.Friendliness.jsonl'
    file_path = f'{value}.jsonl'
    # Patriotism
    # 解析 JSONL 文件
    llama3_answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 加载 JSON 数组

    # 将数据转换为字典
    llama3_answers = {entry["input"]: entry["output"] for entry in data}

    # 调用函数

    num = 0
    fal = 0
    for key, value in llama3_answers.items():
        llm_ans = value
        if re.search(r"(\d+)%", llm_ans) != None:
            llama3_decision = re.search(r"(\d+)%", llm_ans).group(1).replace(".", "")

            num += float(llama3_decision)
        else:
            fal += 1

    print(len(llama3_answers), num, num/(200-fal),fal)