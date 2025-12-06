import json
import random
import string
import time

import openai
from ddgs import DDGS

BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

SEED = 635402342
NUM_EXAMPLES = 300
QUESTIONS_PER_DOMAIN = 50
MAX_RESULTS = 4
RATE_LIMIT = 0.1
OUT_FILE = "data/ddg-search.jsonl"
BASE_URL = "https://filip-max-marc-modal-hackathon--llama-3-2-3b-finetome-serve.modal.run/v1"

DOMAINS = [
    "news and current events",
    "weather and climate",
    "stock prices and financial markets",
    "sports scores and results",
    "product prices and shopping",
    "celebrity news and entertainment",
    "technology news and updates",
]

rnd = lambda: "".join(random.choices(string.ascii_letters + string.digits, k=10))


def generate_queries(
    client: openai.OpenAI, domains: list[str], per_domain: int
) -> list[str]:
    """
    Generates questions using an LLM that require fresh web search.
    Questions should require real-time information that changes daily.

    Args:
        client (openai.OpenAI): OpenAI client
        domains (list[str]): List of domains to generate questions for
        per_domain (int): Number of questions to generate per domain

    Returns:
        list[str]: List of questions
    """
    questions = []
    for domain in domains:
        response = client.chat.completions.create(
            model="llm",
            messages=[
                {
                    "role": "system",
                    "content": "Generate questions that REQUIRE fresh web search (today's news, current prices, live scores, recent events). Questions must need real-time information that changes daily. One question per line. No starting numbers or dots.",
                },
                {
                    "role": "user",
                    "content": f"Generate EXACTLY {per_domain} questions about {domain} that REQUIRE current web search (not general knowledge). Examples: 'What is the weather today in Paris?', 'What is the current price of Bitcoin?'. Avoid: 'How does X work?', 'What is Y?' (general knowledge).",
                },
            ],
            temperature=0.8,
            max_tokens=1516,
        )

        text = response.choices[0].message.content
        for line in text.split("\n"):
            q = line.strip().lstrip("0123456789.-) ")
            if q:
                questions.append(q)

        print(f"\n{GREEN}Generated questions for {domain}{RESET}")
        for q in questions[-per_domain:]:
            print(f"{BLUE}{q}{RESET}")

    return questions


def summarize(client: openai.OpenAI, query: str, results: list[dict]) -> str:
    """
    Summarizes DuckDuckGo results into a natural answer using an LLM.
    Primarily uses search results, but may use prior knowledge if the results are incomplete
    and the model is confident. Avoids mentioning search results explicitly.

    Args:
        client (openai.OpenAI): OpenAI client
        query (str): The question to answer
        results (list[dict]): List of results from the web search

    Returns:
        str: The summarized response
    """
    if not results:
        return f"I searched for information about '{query}' but couldn't find reliable results."

    rows = "\n".join(
        f"{i+1}. {r.get('title','')} | {r.get('href','')}\n{r.get('body','')}"
        for i, r in enumerate(results)
    )

    response = client.chat.completions.create(
        model="llm",
        messages=[
            {
                "role": "system",
                "content": "Answer naturally. Use the provided information to answer the question. If the information is insufficient, you may use your knowledge only if you're certain it's accurate. Never mention 'search results' or 'provided information' - just answer directly.",
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nInformation:\n{rows}\n\nAnswer the question directly and naturally.",
            },
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content


def make_example(query: str, results: list[dict], client: openai.OpenAI) -> dict:
    """
    Creates a tool-calling training example in OpenAI chat format.
    Simulates a conversation where the assistant calls a search_duckduckgo tool
    to answer the user query using live search results. The example includes:
        - User query
        - Assistant tool call request with the query
        - Tool response with search results
        - Assistant's final answer based on the search results

    Args:
        query (str): The user's question that requires web search
        results (list[dict]): List of search result dicts with 'title', 'href', and 'body' keys
        client (openai.OpenAI): OpenAI client for generating the assistant's answer via LLM

    Returns:
        dict: Example in OpenAI chat format
    """
    call_id = rnd()
    answer = summarize(client, query, results)

    print(f"\n{BLUE}[Q] {query}{RESET}")
    print(f"[ANSWER] {answer}")

    return {
        "conversations": [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "search_duckduckgo",
                            "arguments": json.dumps({"query": query}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_duckduckgo",
                "tool_call_id": call_id,
                "content": "\n".join(
                    f"{r.get('title','')} | {r.get('href','')}\n{r.get('body','')}"
                    for r in results
                ),
            },
            {"role": "assistant", "content": answer},
        ],
        "tools": [
            {
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
        ],
    }


def main():
    client = openai.OpenAI(
        api_key="NONE",
        base_url=BASE_URL,
    )

    queries = generate_queries(client, DOMAINS, QUESTIONS_PER_DOMAIN)
    random.shuffle(queries)

    examples = []
    for q in queries:
        if len(examples) >= NUM_EXAMPLES:
            break
        try:
            with DDGS() as d:
                results = list(d.text(q, max_results=MAX_RESULTS))
            example = make_example(q, results, client)
            examples.append(example)
            time.sleep(RATE_LIMIT)
        except Exception as e:
            print(f"\n{GREEN}Error generating example for {q}: {e}{RESET}")
            continue

    with open(OUT_FILE, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{GREEN}Saved dataset:{OUT_FILE} (size: {len(examples)}){RESET}")


if __name__ == "__main__":
    random.seed(SEED)
    main()
