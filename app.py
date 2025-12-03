import base64
import os
from io import BytesIO

import gradio as gr
import openai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

api_key = os.getenv("AI_GATEWAY_API_KEY")
client = openai.OpenAI(
    api_key="NONE",
    base_url="https://filip-max-marc-modal-hackathon--iris-vllm-serve-dev.modal.run/v1",
)
client_finetuned = openai.OpenAI(
    api_key="NONE",
    base_url="https://filip-max-marc-modal-hackathon--iris-vllm-serve-dev.modal.run/v1",
)


def generate_image(prompt: str):
    """
    Generate an image using Google Imagen via the image generation API.
    """
    try:
        response = client.images.generate(
            model="google/imagen-4.0-ultra-generate-001",
            prompt=prompt,
            n=1,
            response_format="b64_json",
        )

        if response.data and len(response.data) > 0:
            image_obj = response.data[0]

            if hasattr(image_obj, "b64_json") and image_obj.b64_json:
                image_bytes = base64.b64decode(image_obj.b64_json)
                image = Image.open(BytesIO(image_bytes))
                return image

        # If no image found, return error message
        return "Image generation failed - no image in response"
    except Exception as e:
        return f"Image generation error: {str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generates an image based on a text prompt. Use this whenever the user asks you to create, generate, draw, make, or imagine an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the image to generate",
                    },
                },
                "required": ["prompt"],
            },
        },
    }
]


def normalize_message(msg):
    """
    Convert Gradio message format to OpenAI format.
    """
    role = msg.get("role")
    content = msg.get("content")

    if isinstance(content, list):
        text_parts = [
            block.get("text", "") for block in content if isinstance(block, dict)
        ]
        content = " ".join(text_parts)

    return {"role": role, "content": content}


def agent_respond(history, temperature=0.7, max_tokens=512):
    """
    Takes a chat history and returns the assistant's reply content.
    """
    system_message = {"role": "system", "content": "You are a friendly agent."}

    normalized_history = [normalize_message(msg) for msg in history]
    messages = [system_message] + normalized_history

    print(messages)
    response = client_finetuned.chat.completions.create(
        model="llm",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def interact_with_agent(user_message, history, temperature, max_tokens):
    """
    Interface function for Gradio.
    Takes the latest user message and chat history, returns assistant response.
    """
    if history is None:
        history = []

    history = history + [{"role": "user", "content": user_message}]

    response = agent_respond(history, temperature, max_tokens)

    return response


with gr.Blocks(title="Custom LLM") as demo:
    gr.Markdown("<p style='padding: 20px 0;'></p>")

    with gr.Row():
        temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.01, label="Temperature")
        max_tokens = gr.Slider(16, 2048, value=512, step=16, label="Max Tokens")

    chatbot = gr.Chatbot(label="Custom LLM", height=600)

    gr.ChatInterface(
        interact_with_agent,
        chatbot=chatbot,
        additional_inputs=[temperature, max_tokens],
        textbox=gr.Textbox(placeholder="Ask me to generate an image..."),
        examples=[
            [
                "Give me an image of a potato man and his potato wife in their potato house"
            ],
            ["Generate an image of a banana cat"],
            ["Draw a sunset over mountains with a lake"],
            ["Create a futuristic robot design"],
            ["Paint a dragon in a fantasy landscape"],
        ],
        fill_height=True,
    )

demo.launch()
