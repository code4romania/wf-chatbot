import openai
from openai import RateLimitError, APIError
from LLM import ChatSession
import time


class OpenAIChatSession(ChatSession):
    """
    OpenAI implementation of ChatSession using the ChatCompletion API.
    """
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        if api_key:
            openai.api_key = api_key
        self.model = model
        self.temperature = temperature

    def send(self, prompt: str) -> tuple[str, str | None]:
        """
        Send a single-prompt string to the LLM and return (response, error_code).
        Retries on rate limits.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            text = resp.choices[0].message['content'].strip()
            return text, None
        except RateLimitError as e:
            time.sleep(3)
            return self.send(prompt)
        except APIError as e:
            return "", f"APIError: {e}"
        except Exception as e:
            return "", f"Error: {e}"
        
