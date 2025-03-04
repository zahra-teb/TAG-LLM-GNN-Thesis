from openai import OpenAI
from utils.load_data import get_products
from prompts import ZERO_SHOT_PRODUCTS_PROMPT
from active_selection import single_score
import pandas as pd
import json
from tqdm import tqdm
import yaml
import torch
import re
import warnings

warnings.filterwarnings('ignore')

# Load API key from YAML file
def load_api_key(config_path="/home/ahmadi/zahra/thesis_stuff/TAG-LLM-GNN-Thesis/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["openrouter"]["api_key"]

# Initialize OpenAI client with OpenRouter
api_key = load_api_key()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

def generate_pred_conf_explanations(selected_nodes, selected_texts, output_path="../data/llm_responses_subgraph2_llama.json"):
    results = {}

    # Try loading existing results
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)  # Load full JSON
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: JSON file is partially corrupted.")
        # # Try to recover valid JSON by reading line by line
        # with open(output_path, "r", encoding="utf-8") as f:
        #     valid_json_str = ""
        #     for line in f:
        #         valid_json_str += line.strip()
        #     try:
        #         results = json.loads(valid_json_str)
        #     except json.JSONDecodeError:
        #         print("Recovery failed. Starting fresh.")
        #         results = {}
        results = {}

    # Iterate through selected nodes
    for node_id in tqdm(selected_nodes, desc="Processing selected nodes", total=len(selected_nodes)):
        node_id = node_id.item()  # Ensure it's a Python integer
        if str(node_id) in results:
            continue  # Skip already processed nodes

        node_text_attr = selected_texts[node_id]
        if 'Description: nan' in node_text_attr:
            print("Nan in text. Skipping...")
            continue  # Skip nodes with empty or NaN text
        
        prompt = ZERO_SHOT_PRODUCTS_PROMPT.format(description=node_text_attr)

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        res = response.choices[0].message.content

        # Validate and clean JSON response
        try:
            cleaned_res = re.sub(r"^```json\n?|```$", "", res.strip())  # Remove backticks
            json_res = json.loads(cleaned_res)
        except json.JSONDecodeError:
            json_res = {"error": "Invalid JSON response", "node_id": node_id, "raw_response": res}

        # Store response
        results[str(node_id)] = json_res

        print(f"Response for node {node_id} saved.")

    # Save the entire JSON properly
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"All responses saved correctly in '{output_path}'")

if __name__ == "__main__":
    # Load data and texts
    data, texts = get_products("../data/subgraph_data2.pt", "../data/subgraph_texts2.csv")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select top `b` nodes
    b = 3000  
    seed = 42  

    selected_nodes = single_score(
        b=b,
        x=data.x.to(device),
        data=data,
        seed=seed,
        device=device
    )

    # # Pick only the first 10 nodes
    # selected_nodes = selected_nodes[:10]


    # Extract texts for selected nodes
    selected_texts = {node.item(): texts[node.item()] for node in selected_nodes if node.item() < len(texts)}
    # Run LLM annotation
    generate_pred_conf_explanations(selected_nodes, selected_texts)
