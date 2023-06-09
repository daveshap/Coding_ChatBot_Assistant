import openai
from time import time, sleep
import textwrap
import sys


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


def chatbot(conversation, model="gpt-4", temperature=0):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=conversation, temperature=temperature)
            text = response['choices'][0]['message']['content']
            return text, response['usage']['total_tokens']
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = conversation.pop(0)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)


###     MAIN LOOP

a = '''def multi_line_input():
    print('\n\n\nUSER:\n')
    lines = []
    while True:
        try:
            line = input()
            if sys.stdin.isatty():
                # If in a terminal, check for Ctrl+Enter by checking if line is empty
                if not line:
                    break
            else:
                # If not in a terminal (e.g., piping input), break on EOF
                if not line:
                    break
            lines.append(line)
        except KeyboardInterrupt:
            # Ctrl+Enter will raise a KeyboardInterrupt, so we break the loop
            break
    return "\n".join(lines).strip()'''


def multi_line_input():
    print('\n\n\nType END to save and exit.\n[MULTI] USER:\n')
    lines = []
    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)
    return "\n".join(lines)


if __name__ == '__main__':
    # instantiate chatbot
    openai.api_key = open_file('key_openai.txt')
    ALL_MESSAGES = list()
    print('\n\n****** IMPORTANT: ******\n\nType SCRATCHPAD to enter multi line input mode to update scratchpad. Type END to save and exit.')
    
    while True:
        # get user input
        text = input('\n\n\n[NORMAL] USER:\n')
        
        # check if scratchpad updated, continue
        if 'SCRATCHPAD' in text:
            text = multi_line_input()
            save_file('scratchpad.txt', text.strip('END').strip())
            print('\n\n#####      Scratchpad updated!')
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
        print('\n\n\n\nCHATBOT:')
        formatted_lines = [textwrap.fill(line, width=120) for line in response.split('\n')]
        formatted_text = '\n'.join(formatted_lines)
        print(formatted_text)