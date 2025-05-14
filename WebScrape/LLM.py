# File: chat_session.py
from abc import ABC, abstractmethod
import time
import openai
from openai import RateLimitError, APIError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ChatSession(ABC):
    """
    Abstraction for a chat-based LLM session.  
    send() takes a prompt string and returns a tuple (response_text, error_code).
    error_code is None on success, or an exception name/description on failure.
    """

    @abstractmethod
    def send(self, prompt: str) -> tuple[str, str | None]:
        pass


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
        

class DeepSeek(ChatSession):
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
        device_map: str = "auto",
        max_input_length: int = 1024,
        load_in_4bit: bool = True,
        bnb_quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.float16,
        use_double_quant: bool = True,
        # Default generation parameters
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        early_stopping: bool = True
    ):
        """
        Initialize the DeepSeek model with quantization and generation settings.
        """
        # Store generation settings
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.early_stopping = early_stopping
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # BitsAndBytes quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        ).eval()

        # Disable cache to save VRAM
        self.model.config.use_cache = False

        self.max_input_length = max_input_length

    def send(self, prompt: str) -> str:
        """
        Send a single-prompt string to the DeepSeek model and return the response text.
        Conforms to ChatSession.send(prompt: str) -> str interface.
        """
        # Build conversation prompt
        full_prompt = (
            f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
        )

        # Tokenize input
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_length
        ).to(self.model.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                num_return_sequences=1,
                early_stopping=self.early_stopping
            )

        # Decode newly generated tokens
        gen_tokens = outputs[0, inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return response
