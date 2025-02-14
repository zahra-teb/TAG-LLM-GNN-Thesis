import json
from tqdm import tqdm
from together import Together
from prompts import ZERO_SHOT_PRODUCTS_PROMPT
import yaml

# Load API key
def load_api_key(config_path="../../config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["together"]["api_key"]

api_key = load_api_key()
client = Together(api_key=api_key)

# LLM Annotation Function
def generate_pred_conf_explanations(selected_nodes, texts, output_path="../data/llm_responses.json"):
    results = {}

    # Try loading existing results
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    with open(output_path, "a", encoding="utf-8") as f:
        for node_id in tqdm(selected_nodes, desc="Querying LLM"):
            if str(node_id) in results:
                continue  # Skip already processed nodes
            
            node_text_attr = texts[node_id]
            prompt = ZERO_SHOT_PRODUCTS_PROMPT.format(description=node_text_attr)
            
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            res = response.choices[0].message.content
            
            try:
                json_res = json.loads(res)
            except json.JSONDecodeError:
                json_res = {"error": "Invalid JSON response", "node_id": node_id, "raw_response": res}

            results[f"{node_id}"] = json_res
            f.seek(0)
            json.dump(results, f, ensure_ascii=False, indent=4)
            f.truncate()

            print(f"Saved LLM annotation for node {node_id}")

    print(f"All responses saved in '{output_path}'")


if __name__ == "__main__":
    # Load data from products
    data, texts = get_products("../data/subgraph_data.pt", "../data/subgraph_texts.csv")
    
    node_features = data.x.to(torch.float32)  # Ensure correct dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate confidence tensor (set default to 0.5 if unavailable)
    confidences = torch.full((node_features.shape[0],), 0.5, dtype=torch.float32, device=device)

    # Perform active learning (select nodes for annotation)
    selected_indices = single_score(
        b=500,  # Select 500 nodes
        x=node_features, 
        data=graph_data, 
        seed=42, 
        conf=confidences, 
        device=device
    )

    # Convert selected indices to list
    selected_nodes = selected_indices.cpu().numpy().tolist()
    print(f"Selected {len(selected_nodes)} nodes for LLM annotation.")






# from together import Together
# from utils.load_data import get_products
# from prompts import ZERO_SHOT_PRODUCTS_PROMPT
# import os
# import json
# from tqdm import tqdm
# import yaml
# import torch
# import warnings

# warnings.filterwarnings('ignore')

# # Load API key from YAML file
# def load_api_key(config_path="/home/ahmadi/zahra/thesis_stuff/TAG-LLM-GNN-Thesis/config.yaml"):
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config["together"]["api_key"]

# # Initialize Together client
# api_key = load_api_key()
# client = Together(api_key=api_key)

# # def generate_pred_conf_explanations(data, texts, output_path="../data/llm_responses_subgraph.json"):
# #     results = {}
    
# #     # Try loading existing results if the file already exists
# #     try:
# #         with open(output_path, "r", encoding="utf-8") as f:
# #             results = json.load(f)
# #     except (FileNotFoundError, json.JSONDecodeError):
# #         pass  # File doesn't exist or is corrupted, start fresh
    
# #     with open(output_path, "a", encoding="utf-8") as f:
# #         # Initialize tqdm progress bar
# #         for node_id, node_text_attr in tqdm(zip(data.n_id, texts), desc="Processing nodes", total=len(data.n_id)):
# #             if str(node_id) in results:
# #                 continue  # Skip already processed nodes

# #             prompt = ZERO_SHOT_PRODUCTS_PROMPT.format(description=node_text_attr)
# #             response = client.chat.completions.create(
# #                 model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
# #                 messages=[{"role": "user", "content": prompt}],
# #                 response_format={"type": "json_object"},
# #                 temperature=0
# #             )
# #             res = response.choices[0].message.content

# #             # Validate response
# #             if not res or not isinstance(res, str):
# #                 print(f"Invalid response for node {node_id}, skipping...")
# #                 json_res = {"error": "Invalid response", "node_id": node_id, "raw_response": str(res)}
# #             else:
# #                 # Convert the JSON string to a Python dictionary
# #                 try:
# #                     json_res = json.loads(res)
# #                 except json.JSONDecodeError as e:
# #                     print(f"Error decoding JSON for node {node_id}: {e}")
# #                     json_res = {"error": "Invalid JSON response", "node_id": node_id, "raw_response": res}

# #             # Convert tensors to Python types
# #             if isinstance(node_id, torch.Tensor):
# #                 node_id = node_id.item()
# #             json_res = json.loads(json.dumps(json_res, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x))

# #             # Store response using the actual node ID
# #             results[f"{node_id}"] = json_res

# #             # Save each response to the JSON file immediately
# #             f.seek(0)  # Move to the beginning of the file
# #             json.dump(results, f, ensure_ascii=False, indent=4)
# #             f.truncate()  # Ensure old content is removed if necessary

# #             print(f"Response for node {node_id} saved.")

# #     print(f"All responses saved in '{output_path}'")

# def generate_pred_conf_explanations(data, texts, output_path="../data/llm_responses_subgraph.json"):
#     results = {}
    
#     # Try loading existing results if the file already exists
#     try:
#         with open(output_path, "r", encoding="utf-8") as f:
#             results = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         pass  # File doesn't exist or is corrupted, start fresh
    
#     with open(output_path, "a", encoding="utf-8") as f:
#         # Initialize tqdm progress bar with an index to track position
#         for idx, (node_id, node_text_attr) in enumerate(tqdm(zip(data.n_id, texts), desc="Processing nodes", total=len(data.n_id))):
#             if idx < 76:  # Skip the first 76 nodes
#                 continue
            
#             if str(node_id) in results:
#                 continue  # Skip already processed nodes

#             prompt = ZERO_SHOT_PRODUCTS_PROMPT.format(description=node_text_attr)
#             response = client.chat.completions.create(
#                 model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0
#             )
#             res = response.choices[0].message.content

#             # Validate response
#             if not res or not isinstance(res, str):
#                 print(f"Invalid response for node {node_id}, skipping...")
#                 json_res = {"error": "Invalid response", "node_id": node_id, "raw_response": str(res)}
#             else:
#                 # Convert the JSON string to a Python dictionary
#                 try:
#                     json_res = json.loads(res)
#                 except json.JSONDecodeError as e:
#                     print(f"Error decoding JSON for node {node_id}: {e}")
#                     json_res = {"error": "Invalid JSON response", "node_id": node_id, "raw_response": res}

#             # Convert tensors to Python types
#             if isinstance(node_id, torch.Tensor):
#                 node_id = node_id.item()
#             json_res = json.loads(json.dumps(json_res, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x))

#             # Store response using the actual node ID
#             results[f"{node_id}"] = json_res

#             # Save each response to the JSON file immediately
#             f.seek(0)  # Move to the beginning of the file
#             json.dump(results, f, ensure_ascii=False, indent=4)
#             f.truncate()  # Ensure old content is removed if necessary

#             print(f"Response for node {node_id} saved.")

#     print(f"All responses saved in '{output_path}'")

# if __name__ == "__main__":
#     # Load data from products
#     data, texts = get_products("../data/subgraph_data.pt", "../data/subgraph_texts.csv")
    
#     # Limit to the first 5 nodes
#     # data.n_id = data.n_id[:5]
#     # texts = texts[:5]

#     # Run the function on a small subset
#     generate_pred_conf_explanations(data=data, texts=texts)

        

