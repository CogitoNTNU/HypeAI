import os
import sys

# Add the parent directory of 'openai' to sys.path
sys.path.append(os.path.abspath('../openai'))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from openai_api import generate_text

'''
This is for generating quiz questions and answers based on
a text file containing knowledge about a certain topic, and related processing.
The result is then used as the informational content of the quiz video.
'''

def read_text_from_file(file_path):
    """
    Read text from a file.

    :param file_path: Path to the file.
    :return: Text content of the file.
    """
    with open(file_path, 'r') as file:
        return file.read()
    
def save_text_to_file(file_path, text):
    """
    Save text to a file.

    :param file_path: Path to the file.
    :param text: Text content to save.
    """
    with open(file_path, 'w') as file:
        file.write(text)
        
def generate_and_save(knowledge_file, output_file, model="gpt-3.5-turbo", max_tokens=750):
    

# Generate quiz questions and answers based on the knowledge text
def generate_quiz_qna(knowledge_file, output_file, model="gpt-3.5-turbo", max_tokens=500):
    """
    Generate quiz questions and answers based on the knowledge text.

    :param knowledge_text: The text containing knowledge about a certain topic.
    :param model: The model to use for text generation.
    :param max_tokens: The maximum number of tokens to generate.
    :return: Generated quiz questions and answers.
    """

    # Change system_prompt to generate better/different questions and answers
    system_prompt = "Your job is to generate 5 extremely concise quiz questions and answers based on the following knowledge. The questions should be no more than 10 words and the answers should be no more than two words. The questions MUST begin with 'Q1:', 'Q2:', and so on until Q5. and the answers MUST begin with 'A1:', 'A2:', and so on until A5."
    user_prompt = read_text_from_file(knowledge_file)
    
    output = generate_text(system_prompt, user_prompt, model=model, max_tokens=max_tokens)
    save_text_to_file(output_file, output)
    #return output.strip()
    #return generate_text(system_prompt, user_prompt, model=model, max_tokens=max_tokens)
    

# Generate a keyword based on the knowledge text
def generate_quiz_keyword(knowledge_file, output, model="gpt-3.5-turbo", max_tokens=20):
    """
    Generate a keyword based on the knowledge text.

    :param knowledge_file: Path to the file containing knowledge about a certain topic.
    :param model: The model to use for text generation.
    :param max_tokens: The maximum number of tokens to generate.
    :return: Generated keyword.
    """
    # Read the knowledge text from the file
    knowledge_text = read_text_from_file(knowledge_file)
    
    # Define the system prompt to generate a keyword
    system_prompt = "Your job is to generate a single tag/keyword that captures the essence of the following knowledge. It must not be longer than two words."
    user_prompt = knowledge_text
    
    # Generate the keyword
    keyword = generate_text(system_prompt, user_prompt, model=model, max_tokens=max_tokens)
    
    return keyword.strip()

# Generate a music prompt for suno based on the knowledge text
def generate_quiz_music_prompt(knowledge_file, model="gpt-3.5-turbo", max_tokens=100):
    """
    Generate a short music prompt based on the knowledge text.

    :param knowledge_file: Path to the file containing knowledge about a certain topic.
    :param model: The model to use for text generation.
    :param max_tokens: The maximum number of tokens to generate.
    :return: Generated music prompt.
    """
    # Read the knowledge text from the file
    knowledge_text = read_text_from_file(knowledge_file)
    
    # Define the system prompt to generate a music prompt
    system_prompt = "Your job is to generate a short music prompt that captures the essence of the following knowledge."
    user_prompt = knowledge_text
    
    # Generate the music prompt
    music_prompt = generate_text(system_prompt, user_prompt, model=model, max_tokens=max_tokens)
    
    return music_prompt.strip()
    

# Only for testing
def main():
    generate_quiz_qna('knowledge/source_1.txt', 'quiz_output/quiz_output_2.txt')
    

if __name__ == "__main__":
    result = main()
    print(result)