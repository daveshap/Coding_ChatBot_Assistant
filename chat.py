import openai
from time import time, sleep
import textwrap
import sys
import yaml


###     file operations


def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)


def open_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


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


def chatbot(conversation, model="gpt-4", temperature=0):
    max_retry = 7

    for retry in range(max_retry):
        try:
            response = get_response(conversation, model, temperature)
            text = response['choices'][0]['message']['content']
            return text, response['usage']['total_tokens']
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
    text = input('\n\n\n[NORMAL] USER:\n')
    if 'SCRATCHPAD' in text:
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
        response, tokens = chatbot(conversation)
        if tokens > 7500:
            ALL_MESSAGES.pop(0)
        ALL_MESSAGES.append({'role': 'assistant', 'content': response})
        print_chatbot_response(response)


if __name__ == '__main__':
    main()
