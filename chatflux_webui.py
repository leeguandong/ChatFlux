import gradio as gr
from openai import AzureOpenAI
import torch
from chatdit import ChatDiT
import os
from PIL import Image
import io
import base64
from typing import List, Union, Tuple

# Initialize Azure OpenAI client
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set.")

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

# Initialize ChatDiT with Azure OpenAI client
app = ChatDiT(
    client=client,
    device=torch.device('cuda')
)

# Store conversation history
conversation_history = []


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for Gradio display."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def process_output(output: Union[Image.Image, List[Image.Image], str], is_markdown: bool = False) -> Tuple[
    List[Tuple[str, str]], List[Image.Image]]:
    """Process ChatDiT output into chatbot format and gallery images."""
    chatbot_output = []
    gallery_images = []

    if is_markdown:
        chatbot_output = [(output, None)]  # Markdown text for chatbot
    elif isinstance(output, list):
        chatbot_output = [("Generated images displayed below:", None)]
        gallery_images = output  # Multiple images for gallery
    else:
        chatbot_output = [("Generated image displayed below:", None)]
        gallery_images = [output]  # Single image for gallery

    return chatbot_output, gallery_images


def chat_with_dit(message: str, image_files: List[str], mode: str, return_markdown: bool = False) -> Tuple[
    List[Tuple[str, str]], List[Image.Image]]:
    """Handle chat interaction with ChatDiT based on mode."""
    global conversation_history

    # Convert uploaded image files to PIL Images
    images = []
    if image_files:
        for file_path in image_files:
            try:
                img = Image.open(file_path).convert("RGB")
                images.append(img)
            except Exception as e:
                conversation_history.append((f"Error loading image: {str(e)}", None))
                return conversation_history, []

    # Append user message to history
    user_message = (message, None)
    conversation_history.append(user_message)

    try:
        if mode == "text-to-image":
            output = app.chat(message, return_markdown=return_markdown)
        elif mode == "image-to-image":
            if not images:
                conversation_history.append(("Please upload an image for image-to-image mode.", None))
                return conversation_history, []
            output = app.chat(message, images=images, return_markdown=return_markdown)[0]
        elif mode == "image-to-images":
            if not images:
                conversation_history.append(("Please upload an image for image-to-images mode.", None))
                return conversation_history, []
            output = app.chat(message, images=images, return_markdown=return_markdown)
        elif mode == "images-to-image":
            if len(images) < 3:
                conversation_history.append(("Please upload three images for images-to-image mode.", None))
                return conversation_history, []
            output = app.chat(message, images=images, return_markdown=return_markdown)
        elif mode == "images-to-images":
            if len(images) < 2:
                conversation_history.append(("Please upload two images for images-to-images mode.", None))
                return conversation_history, []
            output = app.chat(message, images=images, return_markdown=return_markdown)
        elif mode == "text-image-article":
            output = app.chat(message, return_markdown=True)
            # Save article if requested
            output.save('./innovative_building_designing/')

        # Process output for chatbot and gallery
        bot_response, gallery_images = process_output(output, is_markdown=return_markdown)
        conversation_history.extend(bot_response)

    except Exception as e:
        conversation_history.append((f"Error: {str(e)}", None))
        gallery_images = []

    return conversation_history, gallery_images


def clear_conversation():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []
    return conversation_history, []


with gr.Blocks(title="ChatDiT Multi-Turn Conversation") as demo:
    gr.Markdown("# ChatDiT Multi-Turn Conversation Interface")
    gr.Markdown(
        "Interact with ChatDiT for text-to-image, image-to-image, and article generation. Select a mode, input text, and upload images as needed.")

    with gr.Row():
        mode = gr.Radio(
            choices=[
                "text-to-image",
                "image-to-image",
                "image-to-images",
                "images-to-image",
                "images-to-images",
                "text-image-article"
            ],
            label="Interaction Mode",
            value="text-to-image"
        )

    chatbot = gr.Chatbot(label="Conversation", height=400)
    gallery = gr.Gallery(label="Generated Images", height=300)

    with gr.Row():
        with gr.Column(scale=4):
            text_input = gr.Textbox(label="Your Message", placeholder="Type your prompt here...")
        with gr.Column(scale=1):
            return_markdown = gr.Checkbox(label="Return Markdown (for articles)", value=False)

    with gr.Row():
        image_input = gr.File(label="Upload Images", file_count="multiple", file_types=["image"])

    with gr.Row():
        submit_button = gr.Button("Send")
        clear_button = gr.Button("Clear Conversation")

    # Event handlers
    submit_button.click(
        fn=chat_with_dit,
        inputs=[text_input, image_input, mode, return_markdown],
        outputs=[chatbot, gallery]
    )
    clear_button.click(
        fn=clear_conversation,
        outputs=[chatbot, gallery]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9008)
