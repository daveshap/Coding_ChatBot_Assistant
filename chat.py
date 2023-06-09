import openai
from time import time, sleep
import textwrap


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




if __name__ == '__main__':
    # instantiate chatbot
    openai.api_key = open_file('key_openai.txt')
    ALL_MESSAGES = list()
    
    while True:
        # get user input
        text = input('\n\n\n\nUSER: ').strip()
        
        # check if scratchpad updated, continue
        if 'SCRATCHPAD' in text:
            save_file('scratchpad.txt', text.replace('SCRATCHPAD', '').strip())
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
        response = chatbot(conversation)
        ALL_MESSAGES.append({'role': 'assistant', 'content': response})
        print('\n\n\n\nCHATBOT:')
        #formatted_text = textwrap.fill(response, width=100, initial_indent='    ', subsequent_indent='    ')
        formatted_text = textwrap.fill(response, width=120)
        print(formatted_text)