import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

#api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


# Load the API key from the environment


#if not api_key:
 #   raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI client

def generate_text(system_prompt, user_prompt, model="gpt-3.5-turbo", max_tokens=100):
    """
    Generate text using OpenAI's API with a system prompt.

    :param system_prompt: The system prompt to set the behavior of the assistant.
    :param user_prompt: The user prompt to send to the API.
    :param model: The model to use for text generation.
    :param max_tokens: The maximum number of tokens to generate.
    :return: The generated text.
    """
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=max_tokens)

if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "Once upon a time"
    generated_text = generate_text(system_prompt, user_prompt)
    print(f"Generated text: {generated_text}")

    # Example with an empty user prompt
    print(generate_text(system_prompt, ""))

#print(generate_text("You are a helpful assistant.", "Once upon a time"))