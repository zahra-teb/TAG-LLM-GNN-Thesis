ZERO_SHOT_BOOKS_PROMPT = """
**Description**: {description}
**Question**: Which of the following 10 genres of book does this book belong to: 'non-fiction', 'history, historical fiction, biography', 'poetry', 'mystery, thriller, crime', 'comics, graphic', 'children', 'fantasy, paranormal', 'romance', 'fiction', 'young-adult'?
Give at most 5 likely genres as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100 in the form of a list of python dicts like [{{"genre": <answer_here>, "confidence": <confidence_here>}}, ...].
You also must provide your reasoning.
**Note**: Only if the **Description** is 'user', just return -1 as genre with a 100 confidence.
The final output format must be ONLY a JSON object as follows:
{{
    "genres_and_confs": [{{"genre": <answer_here>, "confidence": <confidence_here>}}, ...],
    "reasoning": <your provided reasoning>
}}
"""