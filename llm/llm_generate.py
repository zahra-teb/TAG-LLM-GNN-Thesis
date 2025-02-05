from together import Together
from utils.load_data import get_products
from prompts import ZERO_SHOT_PRODUCTS_PROMPT
import os
import json
from tqdm import tqdm
import yaml

# Load API key from YAML file
def load_api_key(config_path="/home/ahmadi/zahra/thesis_stuff/TAG-LLM-GNN-Thesis/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["together"]["api_key"]

# Initialize Together client
api_key = load_api_key()
client = Together(api_key=api_key)

def generate_pred_conf_explanations(data, texts, output_path="../data/llm_responses_subgraph.json"):
    results = {}
    
    # Try loading existing results if the file already exists
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # File doesn't exist or is corrupted, start fresh
    
    with open(output_path, "a", encoding="utf-8") as f:
        # Initialize tqdm progress bar
        for node_id, node_text_attr in tqdm(zip(data.n_id, texts), desc="Processing nodes", total=len(data.n_id)):
            if str(node_id) in results:
                continue  # Skip already processed nodes

            prompt = ZERO_SHOT_PRODUCTS_PROMPT.format(description=node_text_attr)
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            res = response.choices[0].message.content

            # Convert the JSON string to a Python dictionary
            try:
                json_res = json.loads(res)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for node {node_id}: {e}")
                json_res = {"error": "Invalid JSON response", "node_id": node_id, "raw_response": res}

            # Store response using the actual node ID
            results[f"{node_id}"] = json_res

            # Save each response to the JSON file immediately
            f.seek(0)  # Move to the beginning of the file
            json.dump(results, f, ensure_ascii=False, indent=4)
            f.truncate()  # Ensure old content is removed if necessary

            print(f"Response for node {node_id} saved.")

    print(f"All responses saved in '{output_path}'")

if __name__ == "__main__":
    # Load data from products
    data, texts = get_products("../data/subgraph_data.pt", "../data/subgraph_texts.csv")
    
    # Limit to the first 5 nodes
    # data.n_id = data.n_id[:5]
    # texts = texts[:5]

    # Run the function on a small subset
    generate_pred_conf_explanations(data=data, texts=texts)

        

