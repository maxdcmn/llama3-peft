import gradio as gr
import openai

client_finetuned = openai.OpenAI(
    api_key="NONE",
    base_url="https://filip-max-marc-modal-hackathon--llama3-finetome-serve.modal.run/v1",
)


def agent_respond(history, temperature=0.7, max_tokens=512):
    """
    Takes a chat history and returns the assistant's reply content.
    """
    system_message = {"role": "system", "content": "You are a friendly agent."}
    messages = [system_message] + history

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
        max_tokens = gr.Slider(16, 1024, value=512, step=16, label="Max Tokens")

    chatbot = gr.Chatbot(label="Custom LLM", height=600)

    gr.ChatInterface(
        interact_with_agent,
        chatbot=chatbot,
        additional_inputs=[temperature, max_tokens],
        textbox=gr.Textbox(placeholder="Ask me anything..."),
        fill_height=True,
    )

demo.launch()
