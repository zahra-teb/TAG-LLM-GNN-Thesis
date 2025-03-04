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

ZERO_SHOT_PRODUCTS_PROMPT = """
Product description: {description}
Question: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NAN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? 
Give the most likely category with a confidence score ranging from 0 to 100 in the form of a python dict like {{"category:":<answer_here>, "confidence": <confidence_here>}}.
The category you predict must not contain any number, just the category itself.
You also must provide your reasoning.
The final output format MUST be a JSON object as follows:
{{
    "cat_and_conf": {{"category:":<answer_here>, "confidence": <confidence_here>}},
    "reasoning": <your provided reasoning>
}}
Note: Just generate the json without any explanations and aditional characters.
Answer:
"""