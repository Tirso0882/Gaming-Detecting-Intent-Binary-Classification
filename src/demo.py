import argparse
import os

import gradio as gr
import torch

from src.inference import GamingIntentPredictor


def create_demo(model_path):
    """
    Create a Gradio demo for the gaming intent classifier.
    
    Args:
        model_path (str): Path to the saved model checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    predictor = GamingIntentPredictor(model_path, device)
    
    def predict_intent(text):
        prediction, probabilities = predictor.predict(text, return_probabilities=True)
        
        non_gaming_prob = probabilities[0]
        gaming_prob = probabilities[1]
        
        result = {
            "Non-Gaming": float(non_gaming_prob),
            "Gaming": float(gaming_prob)
        }
        
        if prediction == 1:
            label = "Gaming-Related"
        else:
            label = "Non-Gaming"
        
        return result, label
    
    # Create Gradio interface
    with gr.Blocks(title="Gaming Intent Detector") as demo:
        gr.Markdown("# Gaming Intent Detector")
        gr.Markdown("This demo detects whether a text input is related to gaming or not.")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter text",
                    placeholder="Type your message here...",
                    lines=5
                )
                submit_button = gr.Button("Analyze")
            
            with gr.Column():
                label_output = gr.Textbox(label="Prediction")
                prob_output = gr.Label(label="Confidence")
        
        submit_button.click(
            fn=predict_intent,
            inputs=text_input,
            outputs=[prob_output, label_output]
        )
        
        examples = [
            ["When is the next Elden Ring DLC coming out?"],
            ["I need help with the final boss in God of War Ragnarök."],
            ["What's the best graphics card for playing Cyberpunk 2077?"],
            ["Can someone recommend a good restaurant in downtown?"],
            ["My phone battery drains too quickly, any solutions?"],
            ["What's the weather forecast for tomorrow?"]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=text_input
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Gradio demo for the gaming intent classifier")
    parser.add_argument("--model_path", type=str, default="output/best_model.pt",
                        help="Path to the saved model checkpoint")
    parser.add_argument("--share", action="store_true",
                        help="Whether to create a publicly shareable link")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}. Please train the model first.")
        exit(1)
    
    demo = create_demo(args.model_path)
    demo.launch(share=args.share) 