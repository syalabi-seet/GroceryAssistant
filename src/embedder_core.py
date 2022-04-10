# ArgumentParser imports
import argparse
import multiprocessing as mp

# Recipe1MDataset imports
import regex as re
import random
import json
from itertools import compress
from rich.progress import track
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Embedder imports
import os
import io
import json
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial.distance import cosine
from loguru import logger
import streamlit as st


def ArgumentParser():
    # Dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_state', default=42)
    parser.add_argument('--window_size', default=5)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--vocab_size', default=10000)
    parser.add_argument('--sequence_length', default=4)
    parser.add_argument('--num_sampled', default=64)
    parser.add_argument('--embed_size', default=128)
    parser.add_argument('--n_shards', default=5)
    parser.add_argument('--test_size', default=0.2)
    parser.add_argument('--buffer_size', default=1024)
    parser.add_argument('--workers', default=(mp.cpu_count() // 4))
    return parser.parse_known_args()[0]


@st.cache(show_spinner=False)
def load_embedder():
    args = ArgumentParser()
    embedder = Embedder(args)
    return embedder


class Recipe1MDataset:
    def __init__(self, args):
        self.random_state = args.random_state
        self.vocab_size = args.vocab_size
        self.lemma = WordNetLemmatizer()

    def load_recipes(self):
        with open("Recipe1M+/det_ingrs.json") as f:
            recipes = json.load(f)

        random.seed(self.random_state)
        
        corpus = []
        for i, recipe in track(enumerate(recipes), total=len(recipes)):
            ingredients = [x['text'] for x in compress(recipe['ingredients'], recipe['valid'])]
            ingredients = self.process_ingredients(ingredients)
            recipes[i]['ingredients'] = random.sample(ingredients, k=len(ingredients))
            corpus.extend(ingredients)
            del recipes[i]['valid']   

        self.save_corpus(corpus)
        self.save_recipes(recipes)

    def clean_ingredient(self, ingredient):
        ingredient = ingredient.lower()
        ingredient = ingredient.replace(" - ", "-")
        ingredient = ingredient.replace(" and ", " & ")
        return ingredient
    
    def lemmatize_ingredient(self, ingredient):
        ingredient = " ".join(
            list(map(self.lemma.lemmatize, ingredient.split())))
        ingredient = ingredient.replace("cooky", "cookie")
        return ingredient

    def process_ingredients(self, ingredients):
        hypen_patterns = [
            r"[^|\s]?(low)\s*(fat|sodium|salt)",
            r"[^|\s]?(fat)\s*(free)",
            r"[^|\s]?(non)\s*(fat)",
            r"[^|\s]?(trans)\s*(fat-free)",
            r"[^|\s]?(sugar)\s*(free)",
            r"[^|\s]?(all)\s*(purpose)",
            r"[^|\s]?(multi)\s*(grain)",
            r"[^|\s]?(stir)\s*(fry)",
            r"[^|\s]?(stone)\s*(ground)"]

        apostrophe_patterns = [
            r"(lawry[ |']?[s]?\b)\s",
            r"(zata[rian]+[ |']?[s]?\b)\b",
            r"(pig[ |']?[s]?\b)",
            r"(mccormick[ |']?[s]?\b)",
            r"(campbell[ |']?[s]?\b)"]

        misc_patterns = [
            r"\d+[%\"\- inch\ in.]+",
            r"[\-|\w+]*cooked\b\s",
            r"^(prepared\s\w+\s)",
            r"^(healthy\s\w+\s)"]

        cleaned_ingredients = []
        for ingredient in sorted(ingredients):
            ingredient = self.clean_ingredient(ingredient)

            # Hypen words
            for pattern in hypen_patterns:
                if re.match(pattern, ingredient):
                    ingredient = re.sub(pattern, r"\g<1>-\g<2>", ingredient)
                    break

            # Apostrophe words
            for pattern in apostrophe_patterns:
                if re.match(pattern, ingredient):
                    if not "pig" in pattern:
                        replace = r""
                    else:
                        replace = r"pig's "                    
                    ingredient = re.sub(pattern, replace, ingredient)                   
                    break

            # Misc words
            for pattern in misc_patterns:
                if re.match(pattern, ingredient):
                    ingredient = re.sub(pattern, r"", ingredient)         

            # Split zest & rind & juice ingredients
            if ',' in ingredient:
                matches = re.search(r"^(.+),\s(.+)\sof$", ingredient)
                ingredient, types = matches.groups()
                for method in types.split(" & "):
                    cleaned_ingredients.append(f"{ingredient} {method}")

            # Lemmatize
            ingredient = self.lemmatize_ingredient(ingredient)    

            # Remove keywords
            ingredient = re.sub(r"^(head\s)", r"", ingredient)
            ingredient = ingredient.replace("head of ", "")
            ingredient = ingredient.replace("store-bought ", "")
            ingredient = re.sub(r"(.*(black\spepper\b))", r"\g<2>", ingredient)
            ingredient = re.sub(r"(whole\s(black\speppercorn\b))", r"\g<2>", ingredient)

            # Remove ambiguous ingredients
            ambiguous = ["ziploc bag", "water", "tap water", "pig's", "ice", "pen"]
            if (re.match(r"your\sfavorite\s.+", ingredient) or 
                    ingredient in ambiguous):
                ingredient = ""

            # Split `&` words
            markers = [
                "salt", "tomato", "pork", "kosher salt", "frozen pea",
                "rotel tomato", "pea", "garlic"]
            ends = ["sauce", "paste", "blend", "mix", "seasoning", "filling"]

            if " & " in ingredient:
                for marker in markers:
                    if ingredient.startswith(marker):
                        for end in ends:
                            if not ingredient.endswith(end):
                                cleaned_ingredients.extend(ingredient.split(" & "))
                                break
                        break
            elif ingredient:
                cleaned_ingredients.append(ingredient)

        return set(cleaned_ingredients)
    
    def save_corpus(self, corpus):
        counter = Counter(corpus).most_common()[:self.vocab_size-2]
        corpus_dict = []

        for i, (ingredient, frequency) in enumerate(counter):
            corpus_dict.append({
                'token': ingredient, 'frequency': frequency, 'token_id': i})
                
        with open("data/corpus.json", 'w') as f:
            json.dump(corpus_dict, f, indent=4)
    
    def save_recipes(self, recipes):
        with open("data/recipes.json", "w") as f:
            json.dump(recipes, f, indent=4)


class Embedder:
    def __init__(self, args):
        self.args = args
        self.model_path = 'data/word2vec.kv'
        self.model = self.get_model()
        self.vocab = list(self.model.key_to_index.keys())
        logger.debug("Vocabulary loaded")
        self.vectors = self.get_document_vectors()

    def load_raw_data(self):
        with open("data/recipes.json") as f:
            data = json.load(f)
        return data
    
    def load_cleaned_data(self):
        with open("data/clean_recipes.json") as f:
            data = json.load(f)   
        logger.debug("Recipe database loaded")   
        return data

    def get_model(self):
        if not os.path.exists(self.model_path):
            ingredients = [x['ingredients'] for x in self.load_raw_data()]
            logger.debug("Embedding model trained from scratch")
            embedding_model = Word2Vec(
                sentences=ingredients, vector_size=self.args.embed_size, 
                window=self.args.window_size, min_count=2, workers=self.args.workers,
                sg=0, hs=0, negative=self.args.num_sampled, cbow_mean=1, 
                epochs=10).wv
            logger.debug("Embedding model trained from scratch")
            embedding_model.save(self.model_path)
            logger.debug(f"Word vectors saved to {self.model_path}")
            self.save_tsv(embedding_model)
            logger.debug("Vectors saved for embedding projection")
        else:
            embedding_model = KeyedVectors.load(self.model_path)
            logger.debug("Embedder model loaded")
        return embedding_model

    def save_tsv(self, model):
        out_v = io.open('viz/vectors.tsv', 'w', encoding='utf-8')
        out_m = io.open('viz/metadata.tsv', 'w', encoding='utf-8')
        vectors = model.get_normed_vectors()
        corpus = model.key_to_index

        for word in model.key_to_index:
            vector = vectors[corpus[word]]
            out_v.write('\t'.join([str(x) for x in vector]) + "\n")
            out_m.write(word + "\n")

        out_v.close()
        out_m.close()

    def get_document_vectors(self):
        if not os.path.exists("data/doc_vectors.npy"):
            vectors = []
            for recipe in track(self.recipes, total=len(self.load_data())):
                ingredients = recipe['ingredients']
                if ingredients:
                    document_vector = self.model.vectors_for_all(ingredients).get_normed_vectors()
                    document_vector = np.mean(document_vector, axis=0)
                    vectors.append(document_vector)
            vectors = np.stack(vectors, axis=0)
            np.save("data/doc_vectors", vectors)
            logger.debug("Document vectors saved")
        else:
            vectors = np.load("data/doc_vectors.npy")
        logger.debug("Document vectors loaded")
        return vectors


    @st.cache(show_spinner=False, persist=True)
    def get_results(self, document, document_vectors):
        def compute_vector(document):
            document_vectors = self.model.vectors_for_all(document).get_normed_vectors()
            document_vector = np.mean(document_vectors, axis=0)
            return document_vector

        def clean_results(results):
            recipes = self.load_raw_data()
            cleaned_recipes = self.load_cleaned_data()
            results = [recipes[x]['id'] for x in np.argsort(results) if results[x] != 0.0][:10]
            results = list(filter(lambda x: x['id'] in results, cleaned_recipes))
            return results

        selected_vector = compute_vector(document)

        length = len(document_vectors)
        inputs = zip([selected_vector]*length, document_vectors)
        with mp.Pool(processes=self.args.workers) as pool:
            results = pool.starmap(cosine, inputs)
        
        logger.debug("Similarities calculated")
        return clean_results(results)