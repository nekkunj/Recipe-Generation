import pandas as pd
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer,util
from tensorflow.keras.models import Model
from flask import Flask, render_template, url_for, request, redirect,jsonify

app=Flask(__name__)
model = SentenceTransformer('LaBSE')

def check_ingredients(ingredients, itemset):
    return any(ingre in itemset for ingre in ingredients)
def find_replacement(df,ingredient):
    # Filter rows where the food_label column contains the ingredient
    mask = df['Food label'].str.contains(ingredient, case=False)
    replacements = df[mask]['Substitution label']

    if not replacements.empty:
        return replacements.tolist()[:3]
    else:
        return "No replacement found"
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('tabs.html')
 
# Function that searches the corpus and prints the results
def search(inp_question,model,embeddings,recipes):
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, embeddings)
    hits = hits[0]  #Get the hits for the first query

    # print("Input question:", inp_question)
    # print("Results (after {:.3f} seconds):".format(end_time-start_time))
    # for hit in hits[0:10]:
    #     print("\t{:.3f}\t{}".format(hit['score'], recipes["text_features"][hit['corpus_id']]))
    return hits
 
@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    recipe_quality = request.form['recipe_quality']
    print(f"Received recipe quality: {recipe_quality}")
    ### howto load it for later
    model.load_state_dict(torch.load('./bertmodel/model.pth'))
    model.eval()  # Set the model to evaluation mode if it's for inference
    embeddings = torch.load('./bertmodel/embeddings.pth')
    recipes = pd.read_csv('keywordsandimagesfinaldone.csv')
    # recipes.drop(["Unnamed: 0"],axis=1,inplace=True)
    hits=search(recipe_quality,model,embeddings,recipes)
    print(hits)
    received_Obj = [recipes.iloc[hit['corpus_id']].to_dict() for hit in hits]

    for recipe in received_Obj:
        recipe['CookTime'] = str(recipe['CookTime'])
    for i, hit in enumerate(hits):
        received_Obj[i]['Score'] = hit['score']

    return json.dumps({'recipe': received_Obj})

@app.route('/suggest_ingredient', methods=['POST'])
def suggest_ingredient():
    ingredient_list = request.form['ingredient_Name'].strip().split(',')
    rules = pd.read_csv('association_rules.csv')
    filtered_rules = rules[
        rules['antecedents'].apply(lambda x: check_ingredients(ingredient_list, x)) |
        rules['consequents'].apply(lambda x: check_ingredients(ingredient_list, x))
    ]

    filtered_rules = filtered_rules.sort_values(by='lift', ascending=False)


    filtered_rules = filtered_rules[
        ~filtered_rules.apply(lambda row: any(ingre in row['antecedents'] and ingre in row['consequents'] for ingre in ingredient_list), axis=1)
    ]
    most_associated_ingredient = filtered_rules.iloc[0]['consequents']
    most_associated_ingredient_next = filtered_rules.iloc[1]['consequents']
    sugg=most_associated_ingredient+","+most_associated_ingredient_next
    recipe = f"You can also buy {sugg}"
    return jsonify({'recipe': sugg})



@app.route('/receipe_replacement', methods=['POST'])
def receipe_replacement():
    user_ingredient = request.form['ingredient_Name']
    df=pd.read_csv('final_substitution.csv')
    df = df.drop(columns=['Food id','Substitution id'])
    replacements = find_replacement(df,user_ingredient)
    sugg = f"The replacements are:  {replacements}"
    return jsonify({'recipe': sugg})







if __name__ == "__main__":
    app.run(debug=True)