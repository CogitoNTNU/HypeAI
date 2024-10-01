from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

QUESTION_SYSTEM_PROMPT = "You are a math olympiad coach that creates interesting geometry questions. The questions are high school level difficulty. That are easy to understand, and should have a clever step by step proof."

question_messages = [
    SystemMessage(QUESTION_SYSTEM_PROMPT),
    HumanMessage("Create a geometry question.")
]

result = model.invoke(question_messages)
question = parser.invoke(result)

MANIM_SYSTEM_PROMPT = "You are an expert in creating math animations using the Manim python library. You are excellent at taking a question and step-by-step solution as input, and animating it in a way that is entertainging and easy to understand. You will only output the python code for the animation, nothing else."

manim_messages = [
    SystemMessage(MANIM_SYSTEM_PROMPT),
    HumanMessage("Generate the Python code for animating the solution to this geometry question: " + question)
]

result = model.invoke(manim_messages)
python_code = parser.invoke(result)

print("------------  Question  ------------")
print(question)

print("------------  Manim Code  ------------")
print(python_code)