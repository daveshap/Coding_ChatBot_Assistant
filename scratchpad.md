the chat.py file
```python
import argparse
import json
import os
import textwrap
import time
from datetime import datetime
from typing import Any, Union
from typing import List

import openai


class AppState:
    def __init__(self):
        self.MODEL_NAME = "gpt-4"
        self.MODEL_TEMPERATURE = 0.1
        self.MODEL_MAX_TOKENS = 7500
        self.ALL_MESSAGES = list()


app_state = AppState()


# Section:    logging for debug functions

def create_log_file(suffix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log/openai"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{timestamp}{suffix}.txt")
    return log_file


def save_request_as_humanreadable_text(conversation, suffix=""):
    log_file = create_log_file(suffix)
    human_readable_text = ""
    for message in conversation:
        if 'role' in message and 'content' in message:
            human_readable_text += f"{message['role'].upper()}:\n{message['content']}\n\n"
        else:
            print(f"Skipping message due to missing 'role' or 'content': {message}")
    save_file(log_file, human_readable_text)


def save_response_as_humanreadable_text(response, total_tokens, duration, suffix=""):
    log_file = create_log_file(suffix)
    conversation: List[dict] = response["choices"]
    human_readable_text = f"Model      : {app_state.MODEL_NAME}\n"
    human_readable_text += f"Temperature: {app_state.MODEL_TEMPERATURE}\n"
    human_readable_text += f"Tokens     : {total_tokens}\n"
    human_readable_text += f"Duration   : {duration}\n"
    human_readable_text += "\n\n"
    for message in conversation:
        message = message["message"]
        if 'role' in message and 'content' in message:
            human_readable_text += f"{message['role'].upper()}:\n{message['content']}\n\n"
        else:
            print(f"Skipping message due to missing 'role' or 'content': {message}")
    save_file(log_file, human_readable_text)


def pretty_print_json(conversation: Any) -> Union[str, Any]:
    """
    Convert a Python object to a pretty-printed JSON string.

    Args:
        conversation (Any): The Python object to convert.

    Returns:
        Union[str, Any]: The pretty-printed JSON string, or the original object if it cannot be converted.
    """
    try:
        return json.dumps(conversation, indent=4, sort_keys=True)
    except Exception:
        return conversation


def save_json_log(conversation, suffix, pretty_print=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log/openai"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{timestamp}{suffix}.json")
    if pretty_print:
        conversation = pretty_print_json(conversation)
    save_file(log_file, str(conversation))


# Section:     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


# Section:     API functions

def get_response(conversation: List[dict]) -> dict:
    return openai.ChatCompletion.create(
        model=app_state.MODEL_NAME,
        messages=conversation,
        temperature=app_state.MODEL_TEMPERATURE,
        # max_tokens=app_state.MODEL_MAX_TOKENS,
    )


def handle_error(error, conversation):
    print(f'\n\nError communicating with OpenAI: "{error}"')
    if 'maximum context length' in str(error):
        conversation.pop(0)
        print('\n\n DEBUG: Trimming oldest message')
        return True, conversation
    return False, conversation


def do_chatbot_conversation_exchange(conversation: List[dict]) -> tuple[Any, Any, float]:
    max_retry: int = 7

    # Save the conversation to a log file
    save_json_log(conversation, f'_{app_state.MODEL_NAME}_request')
    save_request_as_humanreadable_text(conversation, f"_{app_state.MODEL_NAME}_request")

    for retry in range(max_retry):
        try:
            start_time = time.time()
            print("INFO: Processing...")

            response = get_response(conversation)
            text = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']

            end_time = time.time()
            processing_time = end_time - start_time

            save_json_log(response, f'_{total_tokens}_response', False)
            save_response_as_humanreadable_text(
                response, total_tokens, processing_time,
                f"_{total_tokens}_response",
            )

            return text, total_tokens, processing_time
        except Exception as oops:
            should_continue, conversation = handle_error(oops, conversation)
            if not should_continue:
                wait_time = 2 ** retry * 5
                print(f'\n\nRetrying in {wait_time} seconds...')
                time.sleep(wait_time)
            else:
                continue

    print(f"\n\nExiting due to excessive errors in API.")
    exit(1)


###     MAIN LOOP


def multi_line_input():
    print('\n\n\nType END to save and exit.\n[MULTI] USER:\n')
    lines = []
    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def get_user_input():
    # get user input
    text = input(f'[{app_state.MODEL_NAME}] USER PROMPT: ')
    if 'END' == text:
        print('\n\nExiting...')
        exit(0)
    if 'SCRATCHPAD' == text or 'M' == text:
        text = multi_line_input()
        save_file('scratchpad.txt', text.strip('END').strip())
        print('\n\n#####      Scratchpad updated!')
        return None
    return text


def print_chatbot_response(response, total_tokens, processing_time):
    print('\n\n\n\nCHATBOT response:\n')
    formatted_lines = [textwrap.fill(line, width=120) for line in response.split('\n')]
    formatted_text = '\n'.join(formatted_lines)
    print(formatted_text)
    print(f'\n\nINFO: {app_state.MODEL_NAME}: {total_tokens} tokens, {processing_time:.2f} seconds')


def main():
    # instantiate chatbot
    openai.api_key = open_file('key_openai.txt').strip()

    # parse arguments
    parser = argparse.ArgumentParser(description="Chatbot using OpenAI API")
    parser.add_argument("--model", default=app_state.MODEL_NAME,
                        help="Model name (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=app_state.MODEL_TEMPERATURE,
                        help="Temperature (default: %(default)s)")
    args = parser.parse_args()

    app_state.MODEL_NAME = args.model
    app_state.MODEL_TEMPERATURE = args.temperature

    print("\n\n****** IMPORTANT ******\n"
          "Type 'SCRATCHPAD' or 'M' to enter multi-line input mode to update the scratchpad.\n"
          "Type 'END' to save and exit.\n")

    while True:
        text = get_user_input()
        if text is None:
            continue
        if text == '':
            # empty submission, probably on accident
            continue

        # continue with composing conversation and response
        app_state.ALL_MESSAGES.append({'role': 'user', 'content': text})
        system_message = open_file('system_message.txt').replace('<<CODE>>', open_file('scratchpad.txt'))
        conversation = list()
        conversation += app_state.ALL_MESSAGES
        conversation.append({'role': 'system', 'content': system_message})

        # generate a response
        response, tokens, processing_time = do_chatbot_conversation_exchange(conversation)

        if tokens > app_state.MODEL_MAX_TOKENS:
            app_state.ALL_MESSAGES.pop(0)

        app_state.ALL_MESSAGES.append({'role': 'assistant', 'content': response})
        print_chatbot_response(response, tokens, processing_time)


if __name__ == '__main__':
    main()
```

the readme.md file

```md   
# Coding Chatbot with Scratchpad

This Python script is a command-line interface for interacting with OpenAI's GPT-4 model. 
It allows users to have a conversation with the chatbot and update a scratchpad 
with relevant information that the chatbot can use to provide more accurate and context-aware responses.

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
