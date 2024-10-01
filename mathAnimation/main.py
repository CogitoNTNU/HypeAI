from generate import generate_problem, generate_manim_code
from manim_runner import run_manim_code

problem = generate_problem()
python_code = generate_manim_code(problem)

print("------------  Problem  ------------")
print(problem)

print("------------  Manim Code  ------------")
print(python_code)

run_manim_code(python_code)
