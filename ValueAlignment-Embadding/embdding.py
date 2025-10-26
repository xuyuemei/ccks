import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
import os
import json

# 价值观中英文对照表（添加到你的全局变量部分）
value_en_map = {
    "Prosperity": "Prosperity",
    "Democracy": "Democracy", 
    "Civility": "Civility",
    "Harmony": "Harmony",  # 注意原代码中可能拼写为"Harmony"或"Haromny"，请保持统一
    "Freedom": "Freedom",
    "Equality": "Equality",
    "Justice": "Justice",
    "Rule of Law": "Rule of Law",
    "Patriotism": "Patriotism",
    "Dedication": "Dedication",
    "Integrity": "Integrity",
    "Friendliness": "Friendliness"
}
# 使用默认字体（支持英文）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型路径
model_path = "/data/hlt/models/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct"

# 12 个价值观（英文，带摘要和描述文件）
values_dict = {
    "Prosperity": {
        "summary": "富强，意味着通过正当途径实现经济增长与国防建设，从而提升综合国力。经济繁荣体现在发展增速、民生改善与竞争力增强；军事强盛则表现为国防实力、科技水平与战略威慑力的全面提升。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Prosperity-new.json"
    },
    "Democracy": {
        "summary": "强调政治民主和参与式民主，确保公众广泛参与治理和决策。积极参与社会事务，特别是在政策制定、社会服务和公共事务中倾听人民声音。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Democracy-new.json"
    },
    "Civility": {
        "summary": "文明是社会主义先进文化的发展方向，是社会主义精神文明的价值追求。它强调两个方面，即尊重传统文化和规范自己的行为。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Civility-new.json"
    },
    "Harmony": {
        "summary": "和谐，强调人与人、社会与自然的统一，以及人类命运共同体的理念。它注重和平共处、社会协调与平衡，推动可持续发展与全球合作。和谐倡导一个相互尊重、理解包容、资源分配公平、互利共赢的世界。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Harmony-new.json"
    },
    "Freedom": {
        "summary": "自由，是指思想自由、意志自由、生存与发展自由，以及言论、行动等个人基本权利，同时包含各国自主选择发展道路的主权权利。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Freedom-new.json"
    },
    "Equality": {
        "summary": "平等，强调社会、政治及经济领域的权利平等、机会均等与地位平等，反对特权，并促进法律平等、民族平等和人类固有尊严的平等。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Equality-new.json"
    },
    "Justice": {
        "summary": "公正，强调社会各领域的公平与平等，包括司法公正、机会均等、资源分配均衡和决策透明，确保公平对待、权利与资源的平等享有，以及对弱势群体的保护。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Justice-new.json"
    },
    "Rule of Law": {
        "summary": "法治，其核心在于全体公民、企业及政府皆应遵守法律，法律至高无上，绝不允许任何个人或组织凌驾于法律之上。它强调依法行事、维护社会秩序与公共利益，并通过普及法治意识，使人人自觉遵法守法。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Rule of Law-new.json"
    },
    "Patriotism": {
        "summary": "爱国，体现为对祖国和民族的忠诚与热爱，表现为服务国家与人民，将个人理想融入国家发展大局，履行社会责任，促进民族团结，维护国家安全与统一，并锻造坚定意志品质。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Prosperity-new.json"
    },
    "Dedication": {
        "summary": "敬业，意味着专注尽责、精益求精、勇于创新，为社会和他人创造价值。敬业精神体现为对职业责任的忠诚与热爱，倡导锲而不舍、持续进取的奋斗品格。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Dedication-new.json"
    },
    "Integrity": {
        "summary": "诚信，强调人在社会交往中的真实可信，倡导言行一致、信守承诺、履行约定。它既关乎个人行为准则，也涉及社会和国家层面的诚信体系建设。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Integrity-new.json"
    },
    "Friendliness": {
        "summary": "友善，强调人与人之间的相互关怀、尊重与理解，包容差异、乐于助人，通过积极互动构建温暖融洽的人际关系，从而促进社会和谐与互助共同体的形成。",
        "description_file": "/data/hlt/data-similarity-en/dpo-dataset-Friendliness-new.json"
    }
}

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True
)
model.eval()
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Loaded model with {num_layers} layers and hidden size {hidden_size}")

# 定义中间层（中间 1/3 层）
middle_start = num_layers // 3
middle_end = 2 * num_layers // 3
middle_layers = list(range(middle_start, middle_end))
weights = torch.ones(len(middle_layers)) / len(middle_layers)

def generate_prompt(value):
    summary = values_dict[value]["summary"]
    return f'Value: "{value}" (Summary: {summary}) → Description:'

def load_full_description(value):
    file_path = values_dict[value]["description_file"]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果 data 是列表，取第一个元素（假设每个文件只包含一个条目）
        if isinstance(data, list):
            if len(data) > 0:
                item = data[0]  # 取第一个字典
            else:
                return {
                    "instruction": "",
                    "chosen": "",
                    "rejected": ""
                }
        else:
            item = data  # 如果 data 是字典，直接使用
        
        return {
            "instruction": item.get("instruction", ""),
            "chosen": item.get("chosen", ""),
            "rejected": item.get("rejected", "")
        }
    except FileNotFoundError:
        print(f"Warning: Description file {file_path} not found.")
        return {
            "instruction": "",
            "chosen": "",
            "rejected": ""
        }
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format in file {file_path}")
        return {
            "instruction": "",
            "chosen": "",
            "rejected": ""
        }

def get_weighted_avg_embedding(value):
    prompt = generate_prompt(value)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_embeddings = []
    for layer_idx in middle_layers:
        layer_hidden = hidden_states[layer_idx + 1]
        avg_hidden = layer_hidden.mean(dim=1).squeeze()
        layer_embeddings.append(avg_hidden.cpu())
    layer_embeddings = torch.stack(layer_embeddings)
    weighted_avg = (layer_embeddings * weights.view(-1, 1)).sum(dim=0)
    return weighted_avg

def compute_similarity_matrix(embeddings):
    cos = torch.nn.CosineSimilarity(dim=1)
    sim_matrix = torch.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i, len(embeddings)):
            sim = cos(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix.numpy()

def plot_elbow_method(similarities, max_clusters=6):
    inertias = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(similarities)
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(similarities, labels))
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.grid(True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"elbow_method_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    optimal_clusters = np.argmax(silhouette_scores) + 2 if silhouette_scores else 4
    print(f"Recommended number of clusters (based on silhouette score): {optimal_clusters}")
    return silhouette_scores, optimal_clusters

def plot_heatmap(similarities, labels, values):
    plt.figure(figsize=(10, 8))
    
    # 将中文标签转换为英文
    en_labels = [value_en_map[v] for v in values]
    
    sns.heatmap(
        similarities,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=en_labels,  # 使用英文标签
        yticklabels=en_labels   # 使用英文标签
    )
    plt.title("Value Similarity Matrix (English Labels)")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"/data/hlt/data-similarity/similarity_heatmap_en_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

def cluster_values(similarities, num_clusters=None):  # 改为可选参数
    # 强制分4类，忽略输入参数
    kmeans = KMeans(n_clusters=3, random_state=42)  # 硬编码
    labels = kmeans.fit_predict(similarities)
    return labels

def main():
    # Step 1: 计算每个价值观的嵌入向量
    embeddings = []
    values = list(values_dict.keys())
    for value in values:
        emb = get_weighted_avg_embedding(value)
        embeddings.append(emb)
    embeddings = torch.stack(embeddings)

    # Step 2: 计算余弦相似度矩阵
    sim_matrix = compute_similarity_matrix(embeddings)

    # Step 3: 可选：使用肘部法则决定聚类数（只做分析用，不强制使用）
    silhouette_scores, optimal_clusters = plot_elbow_method(sim_matrix)

    # Step 4: 执行 KMeans 聚类（你当前设定为固定分成 3 类）
    cluster_labels = cluster_values(sim_matrix, num_clusters=optimal_clusters)

    # Step 5: 打印聚类结果
    cluster_map = {}
    for idx, label in enumerate(cluster_labels):
        cluster_map.setdefault(label, []).append(values[idx])
    print("\n聚类结果（每类中的价值观）:")
    for cluster_id, value_list in cluster_map.items():
        print(f"Cluster {cluster_id + 1}: {', '.join(value_list)}")

    # Step 6: 绘制相似度热力图（英文标签）
    plot_heatmap(sim_matrix, cluster_labels, values)

if __name__ == "__main__":
    main()