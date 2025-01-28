from together import Together
from utils.load_data import get_dataset
from prompts import ZERO_SHOT_BOOKS_PROMPT
import os
import json
from tqdm import tqdm

client = Together(api_key="3e**************************************************************")

def generate_pred_conf_explanations(data, output_dir="../data/llm_responses"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tqdm progress bar
    for i, node_text_attr in enumerate(tqdm(data.text_nodes, desc="Processing nodes")):
        prompt = ZERO_SHOT_BOOKS_PROMPT.format(description=node_text_attr)
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
            print(f"Error decoding JSON for node {i}: {e}")
            json_res = {"error": "Invalid JSON response", "node_index": i, "raw_response": res}
        
        # Define a unique filename for each JSON response
        file_name = f"response_node_{i}.json"
        file_path = os.path.join(output_dir, file_name)
        
        # Save the response as a JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_res, f, ensure_ascii=False, indent=4)

    print(f"All responses saved in '{output_dir}'")

if __name__ == "main":
    # for now the original data but later it must change to the sample data
    data = get_dataset("../../goodreads_data/goodreads_crime/processed/crime.pkl")
    generate_pred_conf_explanations(data=data)
        

