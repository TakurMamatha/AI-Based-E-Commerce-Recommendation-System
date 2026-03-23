# # import pandas as pd

# # data = pd.read_csv('data/products.csv')
# import os
# import pandas as pd

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# file_path = os.path.join(BASE_DIR, 'data', 'products.csv')

# data = pd.read_csv(file_path)

# def get_recommendations(product_name):
#     product_name = product_name.lower()
#     data['name'] = data['name'].str.lower()

#     if product_name not in data['name'].values:
#         return ["No product found"]

#     # Simple recommendation (same category)
#     category = data[data['name'] == product_name]['category'].values[0]
#     recommendations = data[data['category'] == category]['name'].tolist()

#     return recommendations[:4]
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, 'data', 'products.csv')

data = pd.read_csv(file_path)

# Combine features
# data['combined'] = data['name'] + " " + data['description']
data['combined'] = data['name'] + " " + data['category'] + " " + data['description']

# Convert text → vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['combined'])

# Similarity matrix
similarity = cosine_similarity(vectors)

def get_recommendations(product_name):
    product_name = product_name.lower().strip()
    data['name'] = data['name'].str.lower().str.strip()

    if product_name not in data['name'].values:
        return ["No product found"]

    index = data[data['name'] == product_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    result = [data.iloc[i[0]]['name'] for i in scores[1:5]]
    return result