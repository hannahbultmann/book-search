# 📚 BookSearch
![Python](https://img.shields.io/badge/python-<3.13-blue?logo=python&logoColor=white)
![License](https://img.shields.io/github/license/hannahbultmann/book-search)

A command-line chatbot that helps you discover your next favourite book.
It uses tf-idf vectorization and cosine-similarity to compare your query with book descriptions stored in the dataset.
Instead of browsing through endless lists, talk to the playful "librarian" chatbot.

Based on data from the [Open Library](https://openlibrary.org/developers/dumps).

## How to use
To clone this app, you need [Git Large File Storage (LFS)](https://git-lfs.com) installed on your computer. 
Then, from your command line run:

```bash
# Initialize Git LFS
git lfs install

# Clone this repository
git clone https://github.com/hannahbultmann/book-search.git

# Go into the repository
cd book-search/src

# If necessary, install dependencies
pip install -r ../requirements.txt

# Run the app
python run_chatbot.py
```
