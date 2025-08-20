import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.font_manager import FontProperties
import ipywidgets as widgets
from IPython.display import display, HTML
import json
import re


class DiffusionModelVisualizer:
    """
    Interactive visualization tool for diffusion language models with fixed token length.
    Includes slider controls for step navigation.
    """

    def __init__(self, cmap_name='viridis'):
        """
        Initialize the visualizer with color map for confidence scores.

        Args:
            cmap_name (str): Name of the matplotlib colormap for confidence scores
        """
        self.cmap = plt.get_cmap(cmap_name)
        self.responses = None
        self.confidence_scores = None
        self.current_step = 0
        self.num_steps = 0
        self.font_size = 10
        self.tokens_per_line = 20
        self.line_spacing = 1.5

    def load_data(self, responses, confidence_scores, answers_correct=None, inputs=None):
        """
        Load and validate the diffusion model data.

        Args:
            responses (list): List of responses for each diffusion step, each with the same number of tokens
            confidence_scores (list): List of confidence scores for each step and token

        Returns:
            bool: True if data is valid, False otherwise
        """
        if len(responses) != len(confidence_scores):
            print(
                f"Error: Number of responses ({len(responses)}) doesn't match confidence scores ({len(confidence_scores)})")
            return False

        # Check that all responses have the same number of tokens
        num_tokens = len(responses[0])
        for step, resp in enumerate(responses):
            if len(resp) != num_tokens:
                print(f"Error at step {step}: Expected {num_tokens} tokens, got {len(resp)}")
                return False
            if len(resp) != len(confidence_scores[step]):
                print(
                    f"Error at step {step}: Token count ({len(resp)}) doesn't match confidence scores ({len(confidence_scores[step])})")
                return False

        self.responses = responses
        self.inputs = inputs
        self.confidence_scores = confidence_scores
        self.num_steps = len(responses)
        self.answers_correct = answers_correct
        self.current_step = 0
        return True

    def _format_text_with_color(self, tokens, scores):
        """
        Format tokens with HTML spans colored by confidence scores.

        Args:
            tokens (list): List of tokens to display
            scores (list): Confidence scores for each token

        Returns:
            str: HTML-formatted string with colored tokens
        """
        html_text = ""
        for token, score in zip(tokens, scores):
            # Convert confidence score to RGB color
            # Using a color gradient from blue (low confidence) to red (high confidence)
            r = int(255 * score)
            g = int(100 + (155 * (1 - abs(score - 0.5) * 2)))
            b = int(255 * (1 - score))

            # Add the token with appropriate color
            html_text += f'<span style="color: rgb({r},{g},{b});">{token}</span>'

            # Add space after punctuation
            if token in ['.', ',', '!', '?', ';', ':']:
                html_text += ' '

        return html_text

    def visualize_interactive_html(self):
        """
        Create an interactive HTML visualization with slider controls.
        Works in Jupyter notebooks.

        Returns:
            IPython.display.HTML: Interactive HTML visualization
        """
        if self.responses is None or self.confidence_scores is None:
            print("No data loaded. Call load_data() first.")
            return None

        # Create the slider widget
        step_slider = widgets.IntSlider(
            min=1,
            max=self.num_steps,
            value=1,
            step=1,
            description='Step:',
            continuous_update=False,
            layout=widgets.Layout(width='600px')
        )

        # Container for the response text
        response_container = widgets.HTML(
            value=self._get_response_html(0),
            layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0',
                                  width='800px', min_height='200px')
        )

        # Legend for confidence scores
        legend_html = """
        <div style="text-align: center; margin-top: 10px;">
            <h4>Confidence Score Legend</h4>
            <div style="display: flex; justify-content: center; margin: 10px 0;">
                <div style="background: linear-gradient(to right, rgb(0,100,255), rgb(150,150,150), rgb(255,100,0)); 
                     width: 400px; height: 20px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; width: 400px; margin: 0 auto;">
                <span>Low Confidence (0.0)</span>
                <span>High Confidence (1.0)</span>
            </div>
        </div>
        """
        legend_container = widgets.HTML(value=legend_html)

        # Function to update the display when the slider changes
        def update_display(change):
            step_idx = change['new'] - 1  # Convert from 1-indexed to 0-indexed
            response_container.value = self._get_response_html(step_idx)

        # Connect the slider to the update function
        step_slider.observe(update_display, names='value')

        # Create a title
        title = widgets.HTML(value='<h2 style="text-align: center;">Diffusion Model Response Visualization</h2>')

        # Assemble the complete widget
        display(title)
        display(step_slider)
        display(legend_container)
        display(response_container)

        # Return the main components for further customization if needed
        return {'slider': step_slider, 'container': response_container}

    def _get_response_html(self, step_idx):
        """
        Generate HTML for a specific diffusion step.

        Args:
            step_idx (int): Index of the diffusion step to display

        Returns:
            str: HTML content for the response
        """
        tokens = self.responses[step_idx]
        scores = self.confidence_scores[step_idx]

        # Format the tokens with line breaks for readability
        formatted_html = '<div style="line-height: 1.5; font-family: monospace;">'

        # Display step information
        formatted_html += f'<p style="font-weight: bold;">Step {step_idx + 1}/{self.num_steps}</p>'

        # Add the colored tokens with proper formatting
        current_line = ""
        if self.answers_correct is not None:
            current_line += "Correct Answer: " if self.answers_correct[step_idx] else "False Answer: "
        word_count = 0

        for token, score in zip(tokens, scores):
            # Convert confidence score to RGB color
            r = int(255 * score)
            g = int(100 + (155 * (1 - abs(score - 0.5) * 2)))
            b = int(255 * (1 - score))

            # Format token with color
            colored_token = f'<span style="color: rgb({r},{g},{b});">{token}</span>'

            # Add space after punctuation
            if token in ['.', ',', '!', '?', ';', ':']:
                colored_token += ' '

            # Add to current line
            current_line += colored_token
            word_count += 1

            # Check if we need a new line
            if word_count % self.tokens_per_line == 0:
                formatted_html += current_line + '<br>'
                current_line = ""

        # Add any remaining content
        if current_line:
            formatted_html += current_line

        formatted_html += '</div>'
        return formatted_html

    def visualize_matplotlib(self, step_idx=0, figsize=(12, 8)):
        """
        Create a static matplotlib visualization for a specific step.

        Args:
            step_idx (int): Index of the diffusion step to display
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.responses is None or self.confidence_scores is None:
            print("No data loaded. Call load_data() first.")
            return None

        if step_idx < 0 or step_idx >= self.num_steps:
            print(f"Step index {step_idx} out of range (0-{self.num_steps - 1})")
            return None

        tokens = self.responses[step_idx]
        scores = self.confidence_scores[step_idx]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        # Title
        plt.title(f"Diffusion Step {step_idx + 1}/{self.num_steps}", fontsize=14)

        # Layout parameters
        x_pos = 0.05
        y_pos = 0.95
        x_max = 0.95
        line_height = 0.05

        # Display tokens with colors based on confidence
        for i, (token, score) in enumerate(zip(tokens, scores)):
            # Skip empty tokens
            if not token or token.isspace():
                continue

            # Get color based on confidence
            color = self.cmap(score)

            # Estimate token width (rough approximation)
            token_width = len(token) * 0.01

            # Check if we need a new line
            if x_pos + token_width > x_max or i % self.tokens_per_line == 0 and i > 0:
                x_pos = 0.05
                y_pos -= line_height

            # Add the token with appropriate color
            ax.text(x_pos, y_pos, token, color=color, fontsize=self.font_size)

            # Move position for next token
            x_pos += token_width

            # Add space after punctuation
            if token in ['.', ',', '!', '?', ';', ':']:
                x_pos += 0.01

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Confidence Score')

        plt.tight_layout()
        return fig

    def create_interactive_slider_plot(self, figsize=(14, 10)):
        """
        Create an interactive matplotlib plot with a slider to navigate steps.

        Args:
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: The interactive figure
        """
        if self.responses is None or self.confidence_scores is None:
            print("No data loaded. Call load_data() first.")
            return None

        # Create figure with space for slider
        fig, (ax, slider_ax) = plt.subplots(2, 1, figsize=figsize,
                                            gridspec_kw={'height_ratios': [20, 1]})
        plt.subplots_adjust(bottom=0.15)

        # Initial plot
        ax.axis('off')
        ax.set_title(f"Diffusion Step 1/{self.num_steps}", fontsize=14)

        # Layout parameters
        step_idx = 0
        tokens = self.responses[step_idx]
        scores = self.confidence_scores[step_idx]

        # Text elements for each token
        text_elements = []
        x_pos = 0.05
        y_pos = 0.95
        x_max = 0.95
        line_height = 0.05

        for i, (token, score) in enumerate(zip(tokens, scores)):
            # Skip empty tokens
            if not token or token.isspace():
                continue

            # Get color based on confidence
            color = self.cmap(score)

            # Estimate token width
            token_width = len(token) * 0.01

            # Check if we need a new line
            if x_pos + token_width > x_max or i % self.tokens_per_line == 0 and i > 0:
                x_pos = 0.05
                y_pos -= line_height

            # Add the token with appropriate color
            text_elem = ax.text(x_pos, y_pos, token, color=color, fontsize=self.font_size)
            text_elements.append((text_elem, token))

            # Move position for next token
            x_pos += token_width

            # Add space after punctuation
            if token in ['.', ',', '!', '?', ';', ':']:
                x_pos += 0.01

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Confidence Score')

        # Create slider
        slider = Slider(
            ax=slider_ax,
            label='Diffusion Step',
            valmin=1,
            valmax=self.num_steps,
            valinit=1,
            valstep=1
        )

        # Update function for slider
        def update(val):
            step_idx = int(slider.val) - 1  # Convert to 0-indexed
            tokens = self.responses[step_idx]
            scores = self.confidence_scores[step_idx]

            # Clear previous text elements
            for text_elem, _ in text_elements:
                text_elem.remove()
            text_elements.clear()

            # Update title
            ax.set_title(f"Diffusion Step {step_idx + 1}/{self.num_steps}", fontsize=14)

            # Redraw tokens
            x_pos = 0.05
            y_pos = 0.95

            for i, (token, score) in enumerate(zip(tokens, scores)):
                # Skip empty tokens
                if not token or token.isspace():
                    continue

                # Get color based on confidence
                color = self.cmap(score)

                # Estimate token width
                token_width = len(token) * 0.01

                # Check if we need a new line
                if x_pos + token_width > x_max or i % self.tokens_per_line == 0 and i > 0:
                    x_pos = 0.05
                    y_pos -= line_height

                # Add the token with appropriate color
                text_elem = ax.text(x_pos, y_pos, token, color=color, fontsize=self.font_size)
                text_elements.append((text_elem, token))

                # Move position for next token
                x_pos += token_width

                # Add space after punctuation
                if token in ['.', ',', '!', '?', ';', ':']:
                    x_pos += 0.01

            fig.canvas.draw_idle()

        # Connect the slider to the update function
        slider.on_changed(update)

        return fig, slider

    def save_visualization(self, fig, filename="diffusion_step_visualization.png", dpi=300):
        """
        Save the visualization to a file.

        Args:
            fig (matplotlib.figure.Figure): Figure to save
            filename (str): Output filename
            dpi (int): Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Visualization saved to {filename}")

    def create_web_visualization(self, output_file="diffusion_visualization.html"):
        """
        Create a standalone HTML file with interactive visualization.

        Args:
            output_file (str): Filename for the HTML output

        Returns:
            str: Path to the created HTML file
        """
        if self.responses is None or self.confidence_scores is None:
            print("No data loaded. Call load_data() first.")
            return None

        # Convert data to JSON for JavaScript
        data_json = json.dumps({
            'responses': self.responses,
            'confidence_scores': self.confidence_scores,
            "answers_correct": self.answers_correct,
            'num_steps': self.num_steps,
            "inputs": self.inputs,
            "confidence_scores_inputs": [[0] * len(self.confidence_scores[0])] + self.confidence_scores[:-1],
        })

        # Create HTML with embedded JavaScript
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Diffusion Model Response Visualization</title>
            <style>
            /* 1) Base (light) theme */
                :root {{
                  --bg: #fff;
                  --fg: #111;
                  --border: #ccc;
                  --note: #666;
                  --control-bg: #f5f5f5;
                  --control-fg: #111;
                }}
    
                /* 2) Dark theme overrides */
                @media (prefers-color-scheme: dark) {{
                  :root {{
                    --bg: #121212;
                    --fg: #eee;
                    --border: #333;
                    --note: #aaa;
                    --control-bg: #1e1e1e;
                    --control-fg: #eee;
                  }}
                }}
                body {{
                    background-color: var(--bg);
                    color: var(--fg);
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    max-width: 900px;
                    margin: 0 auto;
                }}
                .container {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .response-container {{
                    border: 1px solid #ddd;
                    padding: 20px;
                    margin: 10px 0;
                    width: 100%;
                    min-height: 300px;
                    font-family: monospace;
                    white-space: pre-wrap;
                    line-height: 1.5;
                }}
                .controls {{
                    margin: 20px 0;
                    width: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .slider {{
                    width: 80%;
                    margin: 0 10px;
                }}
                .step-display {{
                    font-weight: bold;
                    min-width: 80px;
                    text-align: center;
                }}
                .color-legend {{
                    margin-top: 20px;
                    text-align: center;
                }}
                .legend-gradient {{
                    background: linear-gradient(to right, rgb(0,100,255), rgb(150,150,150), rgb(255,100,0));
                    height: 20px;
                    width: 400px;
                    margin: 10px auto;
                }}
                .legend-labels {{
                    display: flex;
                    justify-content: space-between;
                    width: 400px;
                    margin: 0 auto;
                }}
                button {{
                    margin: 0 5px;
                    padding: 5px 10px;
                    cursor: pointer;
                }}
                .token {{
                    display: inline-block;
                    padding: 0 2px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="color-legend">
                  <h3>Confidence Score Legend</h3>
                  <div class="legend-gradient"></div>
                  <div class="legend-labels">
                      <span>Low Confidence (0.0)</span>
                      <span>High Confidence (1.0)</span>
                  </div>
              </div>

                <div class="controls">
                    <button id="prev-btn">&lt; Prev</button>
                    <span class="step-display" id="step-display">Step 1/{self.num_steps}</span>
                    <input type="range" min="1" max="{self.num_steps}" value="1" class="slider" id="step-slider">
                    <button id="next-btn">Next &gt;</button>
                </div>

                <div class="response-container" id="response-container"></div>
            </div>

            <script>
                // Load the data
                const data = {data_json};
                const tokensPerLine = {self.tokens_per_line};

                // Get elements
                const slider = document.getElementById('step-slider');
                const stepDisplay = document.getElementById('step-display');
                const responseContainer = document.getElementById('response-container');
                const prevBtn = document.getElementById('prev-btn');
                const nextBtn = document.getElementById('next-btn');

                // Convert confidence score to RGB color
                function scoreToColor(score) {{
                    const r = Math.floor(255 * score);
                    const g = Math.floor(100 + (155 * (1-Math.abs(score-0.5)*2)));
                    const b = Math.floor(255 * (1-score));
                    return `rgb(${{r}},${{g}},${{b}})`;
                }}

                // Update the display for a specific step
                function updateDisplay(stepIdx) {{
                    const tokens = data.responses[stepIdx];
                    const scores = data.confidence_scores[stepIdx];
                    const reward = data.answers_correct[stepIdx];
                    // Update step display
                    stepDisplay.textContent = `Step ${{stepIdx + 1}}/${{data.num_steps}}`;

                    // Clear previous content
                    responseContainer.innerHTML = '';

                    // Add tokens with colors
                    let html = '';
                    html += `Answer ${{reward}}:`;
                    for (let i = 0; i < tokens.length; i++) {{
                        const token = tokens[i];
                        const score = scores[i];

                        // Add line break every tokensPerLine tokens
                        if (i > 0 && i % tokensPerLine === 0) {{
                            html += '<br>';
                        }}

                        // Add colored token
                        const color = scoreToColor(score);
                        html += `<span class="token" style="color: ${{color}}">${{token}}</span>`;

                        // Add space after punctuation
                        if (['.', ',', '!', '?', ';', ':'].includes(token)) {{
                            html += ' ';
                        }}
                    }}

                    responseContainer.innerHTML = html;
                }}
                // Update the input for a specific step
                function updateInput(stepIdx) {{
                    const tokens = data.inputs[stepIdx];
                    const scores = data.confidence_scores_inputs[stepIdx];
                    const reward = data.answers_correct[stepIdx];
                    // Update step display
                    stepDisplay.textContent = `Step ${{stepIdx + 1}}/${{data.num_steps}}`;

                    // Clear previous content
                    responseContainer.innerHTML = '';

                    // Add tokens with colors
                    let html = '';
                    html += `Answer ${{reward}}:`;
                    for (let i = 0; i < tokens.length; i++) {{
                        const token = tokens[i];
                        const score = scores[i];

                        // Add line break every tokensPerLine tokens
                        if (i > 0 && i % tokensPerLine === 0) {{
                            html += '<br>';
                        }}

                        // Add colored token
                        const color = scoreToColor(score);
                        html += `<span class="token" style="color: ${{color}}">${{token}}</span>`;

                        // Add space after punctuation
                        if (['.', ',', '!', '?', ';', ':'].includes(token)) {{
                            html += ' ';
                        }}
                    }}

                    responseContainer.innerHTML = html;
                }}

                // Initialize with the first step
                updateInput(0);
                input_mode = false;
                // Slider event listener
                slider.addEventListener('input', function() {{
                    const stepIdx = parseInt(this.value) - 1;
                    if (stepIdx > 1) {{
                        updateDisplay(stepIdx);
                        input_mode = true;
                    }}
                    else {{
                        updateInput(stepIdx);
                        input_mode = false;
                    }}
                }});

                // Button event listeners
                prevBtn.addEventListener('click', function() {{
                    const currentStep = parseInt(slider.value);
                    if (input_mode) {{ 
                            updateInput(currentStep - 1);
                            input_mode = false;
                        }}
                    else {{
                        if (currentStep > 1) {{
                            slider.value = currentStep - 1;
                            updateDisplay(currentStep - 2);
                            input_mode = true;         
                            }}             
                    }}
                }});

                nextBtn.addEventListener('click', function() {{
                    const currentStep = parseInt(slider.value);
                     if (input_mode) {{
                        if (currentStep < data.num_steps) {{ 
                            slider.value = currentStep + 1;
                            updateInput(currentStep);
                            input_mode = false;
                            }}
                        }}
                    else {{
                        updateDisplay(currentStep - 1);
                        input_mode = true;
                    }}
                }});
            </script>
        </body>
        </html>
        """

        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Web visualization saved to {output_file}")
        return output_file


# Function to generate example data with 512 tokens for testing
def generate_example_data(num_steps=10, num_tokens=512):
    """
    Generate example data for testing visualization.

    Args:
        num_steps (int): Number of diffusion steps
        num_tokens (int): Number of tokens per step

    Returns:
        tuple: (responses, confidence_scores)
    """
    responses = []
    confidence_scores = []

    # Create a vocabulary of example tokens
    vocabulary = ["the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
                  "and", "or", "but", "because", "if", "when", "where", "how", "what", "why",
                  "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                  "do", "does", "did", "can", "could", "will", "would", "shall", "should",
                  "may", "might", "must", ",", ".", "!", "?", ";", ":", "(", ")",
                  "this", "that", "these", "those", "there", "here"]

    # Generate responses for each step
    for step in range(num_steps):
        # At each step, we'll have more defined tokens and higher confidence
        # For early steps, many tokens will be random and low confidence
        step_tokens = []
        step_scores = []

        # Confidence increases with step number
        base_confidence = min(0.2 + step * (0.8 / num_steps), 0.95)

        for token_idx in range(num_tokens):
            # Early tokens in the sequence tend to be more defined earlier
            position_factor = 1.0 - (token_idx / num_tokens)

            # Determine if this token is "defined" at this step
            token_defined = (token_idx / num_tokens) < (step / num_steps * 1.5)

            if token_defined:
                # Use a more stable token with higher confidence
                token_choice = vocabulary[token_idx % len(vocabulary)]
                confidence = min(base_confidence + position_factor * 0.3, 0.95)
            else:
                # Use a more random token with lower confidence
                token_choice = vocabulary[np.random.randint(0, len(vocabulary))]
                confidence = max(0.1, base_confidence - 0.3)

            step_tokens.append(token_choice)
            step_scores.append(confidence)

        responses.append(step_tokens)
        confidence_scores.append(step_scores)

    return responses, confidence_scores


# Example usage
if __name__ == "__main__":
    # Generate example data
    responses, confidence_scores = generate_example_data(num_steps=20, num_tokens=512)

    # Create visualizer
    visualizer = DiffusionModelVisualizer(cmap_name='plasma')

    # Load data
    visualizer.load_data(responses, confidence_scores)

    # For Jupyter notebooks, use this to create interactive widget
    # visualizer.visualize_interactive_html()

    # For standalone visualization
    # fig = visualizer.visualize_matplotlib(step_idx=5)
    # plt.show()

    # For interactive matplotlib visualization (works in notebook with %matplotlib widget)
    # fig, slider = visualizer.create_interactive_slider_plot()
    # plt.show()

    # Create standalone HTML visualization
    visualizer.create_web_visualization("diffusion_visualization.html")

    print("Done!")


# Function to load real data from files
def load_diffusion_data(responses_file, confidence_scores_file):
    """
    Load diffusion model data from files.

    Args:
        responses_file (str): Path to JSON file containing response tokens
        confidence_scores_file (str): Path to JSON file containing confidence scores

    Returns:
        tuple: (responses, confidence_scores)
    """
    with open(responses_file, 'r') as f:
        responses = json.load(f)

    with open(confidence_scores_file, 'r') as f:
        confidence_scores = json.load(f)

    return responses, confidence_scores


# Function to visualize real data
def visualize_real_data(responses_file, confidence_scores_file, output_file="diffusion_visualization.html"):
    """
    Load and visualize real diffusion model data.

    Args:
        responses_file (str): Path to JSON file with response tokens
        confidence_scores_file (str): Path to JSON file with confidence scores
        output_file (str): Path to save HTML visualization

    Returns:
        str: Path to the created HTML file
    """
    # Load data
    responses, confidence_scores = load_diffusion_data(responses_file, confidence_scores_file)

    # Create visualizer
    visualizer = DiffusionModelVisualizer(cmap_name='plasma')

    # Load data
    visualizer.load_data(responses, confidence_scores)

    # Create web visualization
    return visualizer.create_web_visualization(output_file)