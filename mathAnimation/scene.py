from manim import *

class TriangleAngleBisector(Scene):
    def construct(self):
        # Triangle points
        A = np.array([-3, 2, 0])
        B = np.array([-5, -2, 0])
        C = np.array([1, -2, 0])
        D = np.array([-1, -2, 0])  # Point D on line BC

        # Create triangle
        triangle = Polygon(A, B, C, color=BLUE)
        self.play(Create(triangle))

        # Labels
        label_A = MathTex("A").next_to(A, UP)
        label_B = MathTex("B").next_to(B, LEFT)
        label_C = MathTex("C").next_to(C, RIGHT)
        self.play(Write(label_A), Write(label_B), Write(label_C))

        # Highlight angle A
        angle_A = Arc(radius=0.5, start_angle=PI/2, angle=-PI/3, color=YELLOW)
        angle_label = MathTex(r"\angle A").next_to(angle_A, UP)
        self.play(Create(angle_A), Write(angle_label))

        # Angle bisector
        angle_bisector = Line(A, D, color=RED, stroke_width=2)
        self.play(Create(angle_bisector))

        # Point D label
        label_D = MathTex("D").next_to(D, DOWN)
        self.play(Write(label_D))

        # Add segments BD and CD
        segment_BD = Line(B, D, color=GREEN)
        segment_CD = Line(C, D, color=GREEN)
        self.play(Create(segment_BD), Create(segment_CD))

        # Lengths labels
        label_AB = MathTex("AB=8").next_to(A, LEFT)
        label_AC = MathTex("AC=6").next_to(A, RIGHT)
        label_BD = MathTex("BD=4").next_to(B, LEFT)
        label_CD = MathTex("CD=?").next_to(C, RIGHT)

        self.play(Write(label_AB), Write(label_AC), Write(label_BD), Write(label_CD))

        # Show the Angle Bisector Theorem
        theorem_text = MathTex(r"\frac{AB}{AC} = \frac{BD}{CD}").to_edge(UP)
        self.play(Write(theorem_text))

        # Substituting known values
        substitute_text = MathTex(r"\frac{8}{6} = \frac{4}{CD}").next_to(theorem_text, DOWN)
        self.play(Write(substitute_text))

        # Cross-multiplying
        cross_multiply_text = MathTex(r"8 \cdot CD = 6 \cdot 4").next_to(substitute_text, DOWN)
        self.play(Write(cross_multiply_text))

        # Simplifying
        simplify_text_1 = MathTex(r"8 \cdot CD = 24").next_to(cross_multiply_text, DOWN)
        simplify_text_2 = MathTex(r"CD = \frac{24}{8} = 3").next_to(simplify_text_1, DOWN)
        self.play(Write(simplify_text_1), Write(simplify_text_2))

        # Final result
        result_text = MathTex("CD = 3").scale(1.5).to_edge(DOWN)
        self.play(Write(result_text))

        # Wait before ending
        self.wait(2)