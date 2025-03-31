from manim import *

class TritonAdditionKernel(Scene):
    def construct(self):
        # Title
        title = Text("Triton Parallel Vector Addition", font_size=40)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        self.wait()

        # Create input vectors
        n_elements = 16
        block_size = 4
        
        # Create input arrays visualization
        x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        y_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        x_array = VGroup(*[Square(side_length=0.5).add(Text(str(x), font_size=20)) 
                          for x in x_values]).arrange(RIGHT, buff=0.1)
        y_array = VGroup(*[Square(side_length=0.5).add(Text(str(y), font_size=20)) 
                          for y in y_values]).arrange(RIGHT, buff=0.1)
        
        # Labels
        x_label = Text("X Array:", font_size=24).next_to(x_array, LEFT)
        y_label = Text("Y Array:", font_size=24).next_to(y_array, LEFT)
        
        # Position arrays
        array_group = VGroup(x_label, x_array, y_label, y_array).arrange(DOWN, buff=0.5)
        array_group.move_to(ORIGIN)
        
        # Show input arrays
        self.play(Create(x_array), Write(x_label))
        self.play(Create(y_array), Write(y_label))
        self.wait()

        # Demonstrate parallel processing blocks
        colors = [RED, BLUE, GREEN, YELLOW]
        blocks = []
        
        for i in range(n_elements // block_size):
            block_x = VGroup(*x_array[i*block_size:(i+1)*block_size])
            block_y = VGroup(*y_array[i*block_size:(i+1)*block_size])
            
            # Highlight blocks
            block_highlight = VGroup(
                SurroundingRectangle(block_x, color=colors[i]),
                SurroundingRectangle(block_y, color=colors[i])
            )
            blocks.append(block_highlight)
            
            self.play(Create(block_highlight))
            
            # Show parallel processing text
            process_text = Text(f"Program {i}: Processing block {i*block_size}:{(i+1)*block_size}", 
                              font_size=24,
                              color=colors[i])
            process_text.to_edge(DOWN, buff=1 + i*0.5)
            self.play(Write(process_text))
            self.wait(0.5)

        # Show result
        result_values = [x + y for x, y in zip(x_values, y_values)]
        result_array = VGroup(*[Square(side_length=0.5).add(Text(str(r), font_size=20)) 
                               for r in result_values]).arrange(RIGHT, buff=0.1)
        result_label = Text("Result:", font_size=24).next_to(result_array, LEFT)
        
        result_group = VGroup(result_label, result_array).next_to(array_group, DOWN, buff=1)
        
        self.play(Create(result_array), Write(result_label))
        self.wait()

        # Final message
        final_text = Text("Each block processed in parallel!", 
                         font_size=32,
                         color=YELLOW)
        final_text.to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)

