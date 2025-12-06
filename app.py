import json

import gradio as gr
import openai
from ddgs import DDGS

MODELS = {
    "FineTome": "https://filip-max-marc-modal-hackathon--llama-3-2-3b-finetome-serve.modal.run/v1",
    "DDGS": "https://filip-max-marc-modal-hackathon--llama-3-2-3b-ddg-serve.modal.run/v1",
}
TOOL = {
    "type": "function",
    "function": {
        "name": "search_duckduckgo",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def search(query: str) -> tuple[str, list]:
    """
    Searches the web for the given query and returns the results.

    Args:
        query (str): The query to search for

    Returns:
        tuple[str, list]: A tuple containing the search results and the sources
    """
    try:
        results = list(DDGS().text(query, max_results=3))
        full = "\n".join(f"{r['title']}: {r['body']}" for r in results) or "No results"
        sources = [{"title": r["title"], "url": r["href"]} for r in results]
        return full, sources
    except:
        return "Search failed", []


def chat(
    messages: list,
    model_url: str,
    use_tools: bool,
    temp: float,
    max_tokens: int,
    sources: list = None,
) -> str:
    """
    Chats with the model and returns the response.

    Args:
        messages (list): The messages to chat with
        model_url (str): The URL of the model
        use_tools (bool): Whether to use tools
        temp (float): The temperature
        max_tokens (int): The maximum tokens
        sources (list): The sources

    Returns:
        str: The response
    """
    if sources is None:
        sources = []

    client = openai.OpenAI(api_key="NONE", base_url=model_url)
    kwargs = {
        "model": "llm",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    if use_tools:
        kwargs["tools"] = [TOOL]
    resp = client.chat.completions.create(**kwargs).choices[0].message

    if resp.tool_calls:
        args = json.loads(resp.tool_calls[0].function.arguments)
        query = args.get("query", "")
        result, new_sources = search(query)
        sources.extend(new_sources)
        messages.append(
            {
                "role": "user",
                "content": f"Search results: {result}\n\nAnswer based on this.",
            }
        )
        return chat(messages, model_url, False, temp, max_tokens, sources)

    response = resp.content or ""
    if sources:
        source_links = " ".join(f"[[{i+1}]]({s['url']})" for i, s in enumerate(sources))
        return f"{response}\n\nSources: {source_links}"
    return response


def respond(message: str, history: list, model: str, temp: float, max_tok: int) -> str:
    """
    Responds to a message with a model and returns the response.

    Args:
        message (str): The message to respond to
        history (list): The chat history
        model (str): The model to use
        temp (float): The temperature
        max_tok (int): The maximum tokens

    Returns:
        str: The response
    """
    if model == "DDGS":
        system = "You are a helpful assistant with access to a web search tool. Use the search tool ONLY when you need current information (news, prices, weather, recent events). For general knowledge questions, answer directly without searching. Do not mention the search tool in your response."
    else:
        system = "You are a helpful assistant."

    messages = [{"role": "system", "content": system}]
    for h in (history or [])[-6:]: # keep the last 6 messages as context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        return chat(messages, MODELS[model], model == "DDGS", temp, max_tok)
    except Exception as e:
        return f"Error: {e}"


# gradio chat interface with model selection, temperature and the max tokens limit
with gr.Blocks(title="Llama 3.2 3B Chat") as demo:
    gr.Markdown("# Finetuned Llama-3.2-3B-Instruct Chat")
    with gr.Tabs():
        with gr.Tab("FineTome"):
            with gr.Row():
                temp1 = gr.Slider(0.0, 1.0, 0.7, label="Temperature")
                max_tok1 = gr.Slider(64, 1024, 256, label="Max Tokens")
            # m=message, h=history, t=temp, mt=max_tokens
            gr.ChatInterface(
                lambda m, h, t, mt: respond(m, h, "FineTome", t, mt),
                additional_inputs=[temp1, max_tok1],
                submit_btn="Send",
                chatbot=gr.Chatbot(height=500),
                examples=[
                    ["Explain quantum computing in simple terms"],
                    ["Write a haiku about programming"],
                    ["What are the pros and cons of remote work?"],
                    ["How do I make a good cup of coffee?"],
                ],
            )
        with gr.Tab("DDGS"):
            with gr.Row():
                temp2 = gr.Slider(0.0, 1.0, 0.7, label="Temperature")
                max_tok2 = gr.Slider(64, 512, 256, label="Max Tokens")
            gr.ChatInterface(
                lambda m, h, t, mt: respond(m, h, "DDGS", t, mt),
                additional_inputs=[temp2, max_tok2],
                submit_btn="Send",
                chatbot=gr.Chatbot(height=500),
                examples=[
                    ["What is the current price of Bitcoin?"],
                    ["How will the weather in Barcelona be tomorrow?"],
                    ["Who won the latest Champions League final?"],
                    ["What are today's top news headlines?"],
                ],
            )
    gr.Markdown("<small>*First request may take ~30s while Modal spins up the GPU.*</small>")

demo.launch()
