import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import preprocess as prep



class BookSearch():

    def __init__(self):

        self.df = pd.DataFrame()
        self.vectorizer = TfidfVectorizer()
        self.vectors = None    # vector representation of book descriptions
        self.vocab = None    # vocabulary from the descriptions
        self.inverted_index = defaultdict(list)
        self.query = ""    # user query
        self.query_vector = None    # vector representation of user query
        self.query_vocab = None    # vocabulary from the user query

        self.chat = {
            "head": "\n-----------------------------------------------------\n               WELCOME TO BOOK-SEARCH\nwhere your new favourite book is just one search away\n-----------------------------------------------------\n",

            "start": "🧙:\tHi, I've heard you're looking for something to read.\nWell, you're at the right place. \nJust tell me what you want to read about, \nand I'll be happy to recommend you a few books you might like.",

            "loading": ["… blowing dust off the rare manuscripts …",
                   "… brewing tea in the reading room for maximum inspiration …",
                   "… untangling the bookmark collection …",
                   "… asking the mystery novels who stole the missing chapter …",
                   "… rewinding the plot twists so they’re ready for another reader …",
                   "… teaching the dictionaries a few new words …",
                   "… tucking the fairytales in for a midday nap …",
                   "… negotiating a truce between the romance and horror sections …",
                   "… updating the time-travel section to avoid paradoxes …",
                   "… organizing the biographies from 'most scandalous' to 'least scandalous' …",
                   "… waiting for the philosophy section to agree on the meaning of 'loading' …",
                   "… rebinding the space-time continuum in leather …",
                   "… casting protective spells on the ancient scrolls …",
                   "… turning the pages of the universe one second at a time …"],

            "searching": ["… looking through my archive …",
                     "… negotiating with the index cards for faster alphabetical order …",
                     "… checking the margins for secret annotations …",
                     "… summoning the ghost of the head librarian for consultation …",
                     "… cross-referencing your curiosity with the entire Dewey Decimal System …",
                     "… locating the correct shelf… eventually …",
                     "… whispering your request to the card catalog … shhh …",
                     "… pulling out a rolling ladder for dramatic effect …",
                     "… checking behind the oversized atlases …",
                     "… following a trail of paperclips through the stacks …",
                     "… consulting the library mouse for insider tips …",
                     "… unlocking the basement archives and hoping nothing bites …",
                     "… taking the long way because the fiction section is gossiping again …",
                     "… asking the biography section if it knows anyone who’s seen your book …",
                     "… removing a stack of completely unrelated books that insisted on being picked first …",
                     "… politely asking a reference book to give up its secrets …",
                     "… finding a handwritten note in another book that points to your request …"],

            "yes_no": ["That answer is like a missing page – mysterious, but unhelpful. Please choose between 'yes' or 'no'.",
                       "Your answer is so rare, even the rare books section hasn't heard of it. Please choose between 'yes' or 'no'.",
                       "That sounds like the start of a wonderful story, but all I need is 'yes' or 'no'.",
                       "That answer might require its own special shelf, for now, please pick 'yes' or 'no'.",
                       "I'd love to discuss that further, but for now, please select 'yes' or 'no'."],

            "no_match": ["I searched every shelf, but couldn’t find a match for that request.",
                     "The archives are quiet… nothing here fits that description exactly.",
                     "I checked the stacks twice, but no book quite matches your keywords.",
                     "It seems our shelves don’t hold that story… yet.",
                     "No exact matches this time…",
                     "That description is intriguing, but the catalog has nothing quite like it.",
                     "I even checked the dusty corners of the rare books room – no luck this time.",
                     "Even the library mouse says he hasn’t seen a book like that around here."],

            "goodbye": ["Thanks for stopping by. I‘ll be here, dusting the shelves, until you return.",
                    "Well then, I‘ll just reshelve my notes. Goodbye for now.",
                    "It’s been a pleasure hunting down books for you. Drop by anytime.",
                    "Thank you for visiting the archives. The books and I will be waiting.",
                    "Go on then… I’ll just be here arguing with the thesaurus again.",
                    "Goodbye then, and don’t worry, the library mouse will keep me company until you’re back.",
                    "Come back whenever the mood for a good book strikes.",
                    "Alright, may your next chapter be a good one.",
                    "I’ll be here, ready to help you find the next adventure."]}


    # Load and vectorize data
    def load_and_fit(self):
        # Load DataFrame with preprocessed descriptions
        self.df = pd.read_json("../data/data_preprocessed.json", orient="records")

        # Vectorize descriptions
        self.vectors = self.vectorizer.fit_transform(self.df["description_lemma"])

        # Extract vocabulary
        self.vocab = self.vectorizer.get_feature_names_out()

        return True


    # Build inverted index
    def get_inverted_index(self):

        pairs = np.argwhere(self.vectors > 0) # of shape [doc_index, word_index]
        for doc_idx, word_idx in pairs:
            word = self.vocab[word_idx]
            self.inverted_index[word].append(doc_idx)

        return self.inverted_index


    # Preprocess and vectorize query
    def process_query(self,query_input):

        query = prep.preprocess(query_input)
        self.query_vector = self.vectorizer.transform([query])
        self.query_vocab = [self.vocab[word_idx] for x, word_idx in np.argwhere(self.query_vector > 0)]

        return True


    # Find candidate docs with the inverted index
    def get_candidate_docs(self):

        candidate_docs = []

        for word in self.query_vocab:
            doc_idx = self.inverted_index[word]
            candidate_docs += doc_idx

        candidate_docs = set(candidate_docs)

        return candidate_docs


    # Calculate cosine similarity
    # between the query vector and each of the candidate document vectors
    def get_cossim(self,candidate_docs):

        cossim = {}

        message = random.choice(self.chat["searching"])
        print("\n" + message)

        for doc_idx in tqdm(candidate_docs, bar_format="{l_bar}{bar}|",ncols=37):
            cossim[doc_idx] = cosine_similarity(self.query_vector[0], self.vectors[doc_idx])[0][0]
        cossim_sorted = pd.Series(cossim).sort_values(ascending=False)

        return cossim_sorted


    # Print matches with highest cosine similarity
    def results(self,cossim):
        if len(cossim) < 5:
            best_matches = cossim.index
        else:
            best_matches = cossim.head(5).index

        for idx in best_matches:
            title = self.df.iloc[idx]["title"]
            author = self.df.iloc[idx]["authors"]
            year = self.df.iloc[idx]["pub_date"]
            description = self.df.iloc[idx]["description"]

            print(f'"{title}" by {author}')
            print(f"published in {year}")
            print(description, "\n")


    # Call once to start the Chatbot and load the model
    def start(self):

        print(self.chat["head"])
        message = random.choice(self.chat["loading"])
        print("\n" + message + "\n")

        self.load_and_fit()
        self.inverted_index = self.get_inverted_index()

        print(self.chat["start"])

        return None



    def run(self):
        while True:
            # Ask for book query
            query_input = input("\n🧙:\tWith what keywords would you describe your perfect next read?\n\n👤:\t")
            self.process_query(query_input)

            # Retrieve vectors of relevant documents
            candidate_docs = self.get_candidate_docs()

            # Check if any documents have been found, if not -> clarification question & restart
            if len(candidate_docs) == 0:
                message = random.choice(self.chat["no_match"])
                print(f"\n🧙:\t{message} Perhaps try a different set of keywords?")
                continue

            # Calculate cosine similarity
            cossim = self.get_cossim(candidate_docs)

            # Results
            print("\n🧙:\tHere's what I found for you:\n")
            self.results(cossim)

            # Another query?
            next_query = input("\n🧙:\tDo you want me to find some more books for you?\n\n👤:\t")
            while True:
                if "yes" in next_query.lower():
                    break # goes into a new iteration of the outer while loop

                elif "no" in next_query.lower():
                    message = random.choice(self.chat["goodbye"])
                    print(f"\n🧙:\t{message}")
                    return None # ends function

                else: # clarification question
                    message = random.choice(self.chat["yes_no"])
                    next_query = input(f"\n🧙:\t{message}\n\n👤:\t")
