import gradio as gr
import openai
from dotenv import load_dotenv

from PIL import Image
import base64
from io import BytesIO
import os

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


# -------------------
# Define your tools
# -------------------
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


# -------------------
# Agent loop
# -------------------
def agent_respond(history):
    """
    Takes a chat history in Gradio's messages format (list of {"role", "content"})
    and returns an updated history including the assistant's reply.
    """
    system_message = {
        "role": "system",
        "content": """You are an AI agent with image generation capabilities.

IMPORTANT: You have access to the generate_image tool.

If the user asks you to:
- Generate, create, draw, make, produce, or imagine an image
- Show, create, or design a picture
- Create a visual representation

YOU MUST IMMEDIATELY use the generate_image tool with their request as the prompt.

Do not refuse image generation requests. Do not say you cannot generate images. Always use the tool.""",
    }

    messages = [system_message] + history

    response = client_finetuned.chat.completions.create(
        model="llm", messages=messages, tools=tools, tool_choice="auto"
    )

    msg = response.choices[0].message

    # Tool call path
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        name = tool_call.function.name
        args = eval(tool_call.function.arguments)

        if name == "generate_image":
            result = generate_image(**args)

            # Check if result is an image or an error message
            is_image = isinstance(result, Image.Image)
            if is_image:
                tool_content = (
                    f"Image generated successfully ({result.size[0]}x{result.size[1]})"
                )
            else:
                tool_content = str(result)

            # Let the LLM see the tool result
            followup = client_finetuned.chat.completions.create(
                model="llm",
                messages=messages
                + [
                    msg,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    },
                ],
            )

            assistant_text = followup.choices[0].message.content

            # Return the image if generated, otherwise return the text response
            if is_image:
                # Convert PIL Image to base64 data URL for Gradio
                buffered = BytesIO()
                result.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_url = f"data:image/png;base64,{img_base64}"
                content = f"![Generated Image]({image_url})"
            else:
                content = assistant_text

            history = history + [
                {
                    "role": "assistant",
                    "content": content,
                }
            ]
            return history

    # Normal response path (no tool calls)
    assistant_content = msg.content
    history = history + [{"role": "assistant", "content": assistant_content}]
    return history


# -------------------
# Gradio UI
# -------------------


def interact_with_agent(user_message, history):
    """
    Interface function for Gradio ChatInterface.
    Takes the latest user message and chat history, returns updated history.
    """
    if history is None:
        history = []

    # Append the latest user message in messages format
    history = history + [{"role": "user", "content": user_message}]

    # Let the agent generate and append the assistant reply
    history = agent_respond(history)
    return history


demo = gr.ChatInterface(
    interact_with_agent,
    chatbot=gr.Chatbot(label="Image Generation Agent"),
    textbox=gr.Textbox(placeholder="Ask me to generate an image..."),
    examples=[
        ["Give me an image of a potato man and his potato wife in their potato house"],
        ["Generate an image of a banana cat"],
        ["Draw a sunset over mountains with a lake"],
        ["Create a futuristic robot design"],
        ["Paint a dragon in a fantasy landscape"],
    ],
)

demo.launch()
