import json

import openai
from time import time, sleep
import textwrap
import os
from datetime import datetime

MODEL_NAME = "gpt-4"
# MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.1
MODEL_MAX_TOKENS = 7500


###     logging for debug functions

def save_humanreadable_log(conversation, suffix=""):
    if isinstance(conversation, dict) and "choices" in conversation:
        conversation = conversation["choices"][0]["message"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log/openai"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{timestamp}{suffix}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        for message in conversation:
            if 'role' in message and 'content' in message:
                f.write(f"{message['role'].upper()}:\n{message['content']}\n\n")
            else:
                print(f"Skipping message due to missing 'role' or 'content': {message}")


def pretty_print_json(conversation):
    try:
        parsed_json = json.loads(str(conversation))
        pretty_json = json.dumps(parsed_json, indent=4, sort_keys=True)
        return pretty_json
    except json.JSONDecodeError:
        return conversation


def save_json_log(conversation, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log/openai"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{timestamp}{suffix}.json")
    conversation = pretty_print_json(conversation)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(str(conversation))


###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


###     API functions

def get_response(conversation, model, temperature):
    return openai.ChatCompletion.create(model=model, messages=conversation, temperature=temperature)


def handle_error(error, conversation):
    print(f'\n\nError communicating with OpenAI: "{error}"')
    if 'maximum context length' in str(error):
        conversation.pop(0)
        print('\n\n DEBUG: Trimming oldest message')
        return True, conversation
    return False, conversation


def do_chatbot_conversation_exchange(conversation, model="gpt-4", temperature=0.0):
    max_retry = 7

    # Save the conversation to a log file
    save_json_log(conversation, f'_{model}_request')
    save_humanreadable_log(conversation, f"_{model}_request")

    for retry in range(max_retry):
        try:
            response = get_response(conversation, model, temperature)
            text = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']
            save_json_log(response, f'_{total_tokens}_response')
            return text, total_tokens
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
    text = input(f'\n\n\n{MODEL_NAME}: [NORMAL] USER:\n')
    if 'SCRATCHPAD' == text or 'M' == text:
        text = multi_line_input()
        save_file('scratchpad.txt', text.strip('END').strip())
        print('\n\n#####      Scratchpad updated!')
        return None
    return text


def print_chatbot_response(response):
    print('\n\n\n\nCHATBOT:\n')
    formatted_lines = [textwrap.fill(line, width=120) for line in response.split('\n')]
    formatted_text = '\n'.join(formatted_lines)
    print(formatted_text)


def main():
    # instantiate chatbot
    openai.api_key = open_file('key_openai.txt').strip()
    ALL_MESSAGES = list()
    print("\n\n****** IMPORTANT ******\n"
          "Type 'SCRATCHPAD' to enter multi-line input mode to update the scratchpad.\n"
          "Type 'END' to save and exit.\n")

    while True:
        text = get_user_input()
        if text is None:
            continue
        if text == '':
            # empty submission, probably on accident
            continue

        # continue with composing conversation and response
        ALL_MESSAGES.append({'role': 'user', 'content': text})
        system_message = open_file('system_message.txt').replace('<<CODE>>', open_file('scratchpad.txt'))
        conversation = list()
        conversation += ALL_MESSAGES
        conversation.append({'role': 'system', 'content': system_message})

        # generate a response
        response, tokens = do_chatbot_conversation_exchange(conversation, MODEL_NAME, MODEL_TEMPERATURE)
        if tokens > MODEL_MAX_TOKENS:
            ALL_MESSAGES.pop(0)
        ALL_MESSAGES.append({'role': 'assistant', 'content': response})
        print_chatbot_response(response)


if __name__ == '__main__':
    main()
