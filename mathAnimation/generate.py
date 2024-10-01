from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

PROBLEM_SYSTEM_PROMPT = "You are a math olympiad coach that creates interesting geometry problems. The problems are high school level difficulty. That are easy to understand, and should have a clever step by step proof."
MANIM_SYSTEM_PROMPT = "You are an expert in creating math animations using the Manim python library. You are excellent at taking a problem and step-by-step solution as input, and animating it in a way that is entertainging and easy to understand. You will only output the python code for the animation, nothing else."


def generate_problem():
    problem_messages = [
        SystemMessage(PROBLEM_SYSTEM_PROMPT),
        HumanMessage("Create a geometry problem."),
    ]
    result = model.invoke(problem_messages)
    problem = parser.invoke(result)
    return problem


def generate_manim_code(problem):
    manim_messages = [
        SystemMessage(MANIM_SYSTEM_PROMPT),
        HumanMessage(
            "Generate the Python code for animating the solution to this geometry problem: "
            + problem
        ),
    ]
    result = model.invoke(manim_messages)
    python_code = parser.invoke(result)
    return parse_python_output(python_code)


def parse_python_output(python_code):
    # Remove triple backticks if present
    python_code = python_code.replace("```python", "")
    python_code = python_code.replace("```py", "")
    python_code = python_code.replace("```", "")
    return python_code
