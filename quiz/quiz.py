import os
import sys

# Add the parent directory of 'openai' to sys.path
sys.path.append(os.path.abspath('../openai'))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from openai_api import generate_text

'''
This is for generating quiz questions and answers based on
a text file containing knowledge about a certain topic.
This is then used as the main content of the quiz video.
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

def generate_quiz(knowledge_file, output_file, model="gpt-3.5-turbo", max_tokens=500):
    """
    Generate quiz questions and answers based on the knowledge text.

    :param knowledge_text: The text containing knowledge about a certain topic.
    :param model: The model to use for text generation.
    :param max_tokens: The maximum number of tokens to generate.
    :return: Generated quiz questions and answers.
    """

    # Change system_prompt to generate better/different questions and answers
    system_prompt = "Your job is to generate 5 extremely concise quiz questions and answers based on the following knowledge. The questions should be no more than 10 words and the answers should be no more than two words."
    user_prompt = read_text_from_file(knowledge_file)
    
    output = generate_text(system_prompt, user_prompt, model=model, max_tokens=max_tokens)
    save_text_to_file(output_file, output)
    #return generate_text(system_prompt, user_prompt, model=model, max_tokens=max_tokens)

def main():
    # Placeholder file path (change this manually)
    #file_path = 'knowledge/source_1.txt'
    
    # Read the knowledge text from the file
    #knowledge_text = read_text_from_file(file_path)
    
    # Generate quiz questions and answers
    #quiz_content = generate_quiz(knowledge_text)
    generate_quiz('knowledge/source_1.txt', 'quiz_output/quiz_output_1.txt')
    
    # Store the result in a file
    #output_file_path = 'quiz_output/quiz_output_1.txt'
    #with open(output_file_path, 'w') as output_file:
    #    output_file.write(quiz_content)
    
    # Return the result
    # return quiz_content

if __name__ == "__main__":
    result = main()
    print(result)