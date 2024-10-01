from manim import *

class HairyBallsOnSineWave(Scene):
    def construct(self):
        # Create the sine wave
        sine_wave = FunctionGraph(
            lambda x: np.sin(x),
            x_range=[-4*PI, 4*PI, 0.01],
            color=BLUE
        )

        # Create two "hairy" balls
        ball1 = Circle(radius=0.3, fill_opacity=1, color=RED)
        ball2 = Circle(radius=0.3, fill_opacity=1, color=GREEN)

        # Add "hair" to the balls (small lines radiating from the center)
        for ball in [ball1, ball2]:
            for angle in np.linspace(0, 2*PI, 16, endpoint=False):
                hair = Line(ORIGIN, 0.3*UP).rotate(angle, about_point=ORIGIN)
                ball.add(hair)

        # Position balls on the sine wave
        ball1.move_to(sine_wave.point_from_proportion(0))
        ball2.move_to(sine_wave.point_from_proportion(0.25))

        # Add everything to the scene
        self.add(sine_wave, ball1, ball2)

        # Animate the balls moving along the sine wave
        self.play(
            MoveAlongPath(ball1, sine_wave, rate_func=linear),
            MoveAlongPath(ball2, sine_wave, rate_func=linear),
            run_time=10,
            rate_func=linear
        )

# To render this animation, run:
# manim -pql manim_script.py HairyBallsOnSineWave