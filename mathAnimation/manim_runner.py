import subprocess

PYTHON_TEMP_FILE = "manim_temp.py"


def run_manim_code(python_code):
    with open(PYTHON_TEMP_FILE, "w") as f:
        f.write(python_code)

    subprocess.run(["manim", "-pql", PYTHON_TEMP_FILE, "GeneratedProblem"], check=True)
