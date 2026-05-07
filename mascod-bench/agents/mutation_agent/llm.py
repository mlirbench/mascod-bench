import requests
from agents.mutation_agent.prompts.system_prompt import getSystemPrompt

API_URL = "http://old-coms-4020-040.class.las.iastate.edu:8080/generate-mutation/gpt"

def generate_mutation(prompt: str) -> str:
    system_prompt = getSystemPrompt()
    user_prompt = prompt

    response = requests.post(
        API_URL,
        json={
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        },
        # timeout=60
    )

    data = response.json()

    if "output" not in data:
        raise Exception(f"API Error: {data}")

    return data["output"].strip()