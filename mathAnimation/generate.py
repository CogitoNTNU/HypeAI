from openai import OpenAI

client = OpenAI()

QUESTION_SYSTEM_PROMPT = "You are a math olympiad coach that creates interesting geometry questions. The questions are high school level difficulty. That are easy to understand, and should have a clever step by step proof."

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": "Create a geometry question."}
    ]
)

# MATH_TUTOR_SYSTEM_PROMPT = "You are a helpful math tutor that provides short step-by-step solutions to math problems."

question = response.choices[0].message.content.strip()

MANIM_SYSTEM_PROMPT = "You are an expert in creating math animations using the Manim python library. You are excellent at taking a question and step-by-step solution as input, and animating it in a way that is entertainging and easy to understand. You will only output the python code for the animation, nothing else. The Python code should be surrounded with triple backticks."

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
        {"role": "system", "content": MANIM_SYSTEM_PROMPT},
        {"role": "user", "content": "Generate the Python code for animating the solution to this geometry question: " + question}
    ]
)

manim_code = response.choices[0].message.content.strip()

print("------------  Question  ------------")
print(question)

print("------------  Manim Code  ------------")
print(manim_code)
