import argparse
import json
import os
import textwrap
import time
from datetime import datetime
from typing import Any, Union
from typing import List

import anthropic


class AppState:
    def __init__(self):
        self.MODEL_NAME = "claude-1-100k"
        self.MODEL_TEMPERATURE = 0.1
        self.MODEL_MAX_TOKENS = 7500
        self.MODEL_MAX_MESSAGES = 20
        self.ALL_MESSAGES = list()


app_state = AppState()


# Section:    logging for debug functions

def create_log_file(suffix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log/openai"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{timestamp}{suffix}")
    return log_file


def save_request_as_human_readable_text(conversation, suffix):
    log_file = create_log_file(suffix)
    human_readable_text = ""
    for message in conversation:
        if 'role' in message and 'content' in message:
            human_readable_text += f"# {message['role'].upper()}:\n{message['content']}\n\n"
        else:
            print(f"Skipping message due to missing 'role' or 'content': {message}")
    save_content_to_file(log_file, human_readable_text)


def save_response_as_human_readable_text(response, duration, suffix=""):
    log_file = create_log_file(suffix)
    conversation = response['completion']
    human_readable_text = f"- Model      : {app_state.MODEL_NAME}\n"
    human_readable_text += f"- Temperature: {app_state.MODEL_TEMPERATURE}\n"
    #human_readable_text += f"- Tokens     : {total_tokens}\n"
    human_readable_text += f"- Duration   : {duration}\n"
    human_readable_text += "\n\n"
    for message in conversation:
        message = message["message"]
        if 'role' in message and 'content' in message:
            human_readable_text += f"# {message['role'].upper()}:\n{message['content']}\n\n"
        else:
            print(f"Skipping message due to missing 'role' or 'content': {message}")
    save_content_to_file(log_file, human_readable_text)


def pretty_print_json(conversation: Any) -> Union[str, Any]:
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
    save_content_to_file(log_file, str(conversation))


# Section:     file operations

def save_content_to_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def read_file_content(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


# Section:     API functions

def fetch_chatbot_response(conversation: List[dict]) -> dict:
    return openai.completion(
        model=app_state.MODEL_NAME,
        prompt=f"\n\nHuman:{str(conversation)}\n\nAssistant:",
        temperature=app_state.MODEL_TEMPERATURE,
        max_tokens_to_sample=app_state.MODEL_MAX_TOKENS,
    )


def handle_error(error, conversation):
    print(f'\n\nError communicating with OpenAI: "{error}"')
    if 'maximum context length' in str(error):
        conversation.pop(0)
        print('\n\n DEBUG: Trimming oldest message')
        return True, conversation
    return False, conversation


def perform_chatbot_conversation(conversation: List[dict]) -> tuple[Any, Any, float]:
    max_retry: int = 7

    # Save the conversation to a log file
    save_json_log(conversation, f'_{app_state.MODEL_NAME}_request')
    save_request_as_human_readable_text(conversation, f"_{app_state.MODEL_NAME}_request.md")

    for retry in range(max_retry):
        try:
            start_time = time.time()
            print("INFO: Processing...")

            response = fetch_chatbot_response(conversation)
            text = response['completion']
            #total_tokens = response['usage']['total_tokens']

            end_time = time.time()
            processing_time = end_time - start_time

            #save_json_log(response, f'_{total_tokens}_response', False)
            #save_response_as_human_readable_text(
                #response, processing_time, #total_tokens,
                #f"_{total_tokens}_response.md",
            #)


            return text, processing_time, #total_tokens,
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{response}"')
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
        save_content_to_file('scratchpad.md', text.strip('END').strip())
        print('\n\n#####      Scratchpad updated!')
        return None
    return text


def print_chatbot_response(response,  processing_time):
    print('\n\n\n\nCHATBOT response:\n')
    formatted_lines = [textwrap.fill(line, width=120) for line in response.split('\n')]
    formatted_text = '\n'.join(formatted_lines)
    print(formatted_text)
    print(f'\n\nINFO: {app_state.MODEL_NAME}: {processing_time:.2f} seconds')



def main():
    # instantiate chatbot

    # parse arguments
    parser = argparse.ArgumentParser(description="Chatbot using Anthropic Claude API")
    parser.add_argument("--model", default=app_state.MODEL_NAME,
                        help="Model name (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=app_state.MODEL_TEMPERATURE,
                        help="Temperature (default: %(default)s)")
    args, unknown = parser.parse_known_args()

    app_state.MODEL_NAME = args.model
    app_state.MODEL_TEMPERATURE = args.temperature

    print(f"Current settings:\n"
          f"Model: {app_state.MODEL_NAME}\n"
          f"Temperature: {app_state.MODEL_TEMPERATURE}")
    print("Sample app usage: python chat.py --model gpt-3.5-turbo --temperature 0.2")


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
        system_message = read_file_content('system_message.txt').replace('<<CODE>>', read_file_content('scratchpad.md'))
        conversation = list()
        conversation += app_state.ALL_MESSAGES
        conversation.append({'role': 'system', 'content': system_message})
        prompt = f"\n\nHuman:{str(conversation)}\n\nAssistant:"

        # generate a response
        response, processing_time = perform_chatbot_conversation(conversation)

        #Claude doesn't count tokens. We should instead set a max message count.
        #if tokens > app_state.MODEL_MAX_TOKENS:
        messagecount = len(app_state.ALL_MESSAGES)
        if messagecount > app_state.MODEL_MAX_MESSAGES:
            app_state.ALL_MESSAGES.pop(0)

        print(f'Message count: {messagecount}')

        app_state.ALL_MESSAGES.append({'role': 'assistant', 'content': response})
        print_chatbot_response(response, processing_time)


if __name__ == '__main__':
    api_key = read_file_content('key_claude.txt').strip()
    openai = anthropic.Client(api_key)
    main()
