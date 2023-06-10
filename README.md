# Coding Chatbot with Scratchpad

This Python script is a command-line interface for interacting with OpenAI's GPT-4 model. It allows users to have a conversation with the chatbot and update a scratchpad with relevant information that the chatbot can use to provide more accurate and context-aware responses.

## Features

- **User Input**: The script provides two input modes for the user:
  - *Normal Mode*: Single-line input for regular conversation with the chatbot.
  - *Multi-line Mode*: Allows users to enter multiple lines of text to update the scratchpad. To enter this mode, type `SCRATCHPAD` and press Enter. Type `END` to save and exit the multi-line input mode.

- **Scratchpad**: The scratchpad is a text file (`scratchpad.txt`) that can be updated by the user during the conversation. The chatbot uses the information in the scratchpad to provide more context-aware responses.

- **System Message**: A system message is used to provide the chatbot with instructions on how to use the scratchpad. The content of the system message is stored in a separate text file (`system_message.txt`).

- **Conversation History**: The script maintains a conversation history, which is used to provide context to the chatbot. If the conversation becomes too long, the oldest message is removed to fit within the model's token limit.

- **API Communication**: The script communicates with OpenAI's API to generate responses from the chatbot. It handles errors and retries if necessary.

Enjoy your conversation with the chatbot and make use of the scratchpad to provide additional context for more accurate and helpful responses!

## Setup

1. Install OpenAI if not already, should have no other dependencies: `pip install openai --upgrade`, or alternatively see instructions below for virtual environment setup.
2. Create `key_openai.txt` and populate your OpenAI key. Note, you need to have access to GPT-4 API. Note, some operating systems require 

### Setup using virtual environment
1. cd into the project
2. run `python3 -m venv .venv` - creates a folder that will isolate all the dependencies
3. run `source .venv/bin/activate` (on linux) - activates the virtual environment
4. run `pip install -r requirements.txt` - installs all the dependencies

## Usage

1. Run `python chat.py`
2. Type `SCRATCHPAD` to enter scratchpad update mode. This will allow you to copy/paste multiple lines. Type `END` on the last line to save and exit.

## Example Output:

Below is an example output from the chatbot. It uses a combination of textwrap and newlines to ensure code is preserved in console output but not too wide. Note, the scratchpad was previously updated with another python script. You can see that this was able to understand the code and request well enough to understand what other functions were necessary.

``````
****** IMPORTANT: ******

Type SCRATCHPAD to enter multi line input mode to update scratchpad. Type END to save and exit.



[NORMAL] USER:
write a new search_kb function. assume that each KB article is a YAML file in a folder called kb. the elements in the KB ar article and vector. vector is a high dimensional semantic embedding that can be used to calculate cosine similarity with numpy. we will be using Google USEv5 for the embeddings



CHATBOT:
First, let's import the necessary libraries:

```python
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
```

Now, let's create a function to load all the KB articles and their embeddings:

```python
def load_kb_articles(kb_folder='kb'):
    kb_articles = []
    for filename in os.listdir(kb_folder):
        if filename.endswith('.yaml'):
            filepath = os.path.join(kb_folder, filename)
            article = open_yaml(filepath)
            kb_articles.append(article)
    return kb_articles
```

Next, let's create a function to calculate the cosine similarity between the query and the KB articles:

```python
def calculate_similarity(query, kb_articles):
    # Load the Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/5")

    # Calculate the query embedding
    query_embedding = embed([query])[0].numpy()

    # Calculate the cosine similarity between the query and the KB articles
    similarities = []
    for article in kb_articles:
        article_embedding = np.array(article['vector'])
        similarity = cosine_similarity(query_embedding.reshape(1, -1), article_embedding.reshape(1, -1))
        similarities.append(similarity[0][0])

    return similarities
```

Finally, let's update the `search_kb` function to use the above functions and return the most relevant article:

```python
def search_kb(query):
    # Load the KB articles
    kb_articles = load_kb_articles()

    # Calculate the cosine similarity between the query and the KB articles
    similarities = calculate_similarity(query, kb_articles)

    # Find the index of the most similar article
    most_similar_index = np.argmax(similarities)

    # Return the most similar article
    return kb_articles[most_similar_index]['article']
```

Now, the `search_kb` function should return the most relevant KB article based on the cosine similarity between the
query and the article embeddings.
``````
