import pandas as pd
import string
import spacy
from tqdm import tqdm
tqdm.pandas()

# Load spacy model
nlp = spacy.load("en_core_web_sm", disable=['ner', 'senter', 'parser'])

# Remove punctuation and numbers by replacing them with whitespace
def remove_punct_num(text):
    punct = string.punctuation
    digit = string.digits
    characters = []
    
    for c in text:
        if c not in punct and c not in digit:
            characters.append(c)
        elif c in digit:
            characters.append("")
        else:
            characters.append(" ")

    no_punct = "".join(characters)
    
    return no_punct


def lemmatizer(text):
    doc = nlp(text)
    
    tokens = [token.lemma_ for token in doc]

    lemmatized_text = " ".join(tokens)
    
    return lemmatized_text


def preprocess(text):
    
    clean_text = remove_punct_num(text)
    preprocessed_text = lemmatizer(clean_text)
    
    return preprocessed_text


if __name__ == "__main__":
    # Requires the prior execution of 'build_data.ipynb' as the raw data is not included in the git repository.
    
    # Load raw data
    df = pd.read_json("../data/data.json", orient="records")

    # Preprocess data
    df["description_lemma"] = df["description"].progress_apply(preprocess)

    # Save preprocessed data
    data = df.to_json("../data/data_preprocessed.json", orient="records")



