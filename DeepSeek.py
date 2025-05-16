import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class DeepSeek:
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-llm-7b-chat",
        device_map: str = "auto",
        max_input_length: int = 1024,
        load_in_4bit: bool = True,
        bnb_quant_type: str = "nf4",
        compute_dtype: torch.dtype = torch.float16,
        use_double_quant: bool = True
    ):
        """
        Initialize the DeepSeek model with optional 4-bit quantization settings.
        """
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

    def send_message(
        self,
        message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        early_stopping: bool = True
    ) -> str:
        """
        Send a user message to the model and return the assistant's response text.
        """
        # Build prompt
        prompt = (
            f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{message}\n<|assistant|>\n"
        )

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_length
        ).to(self.model.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                early_stopping=early_stopping
            )

        # Decode only the newly generated tokens
        gen_tokens = outputs[0, inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return response


if __name__ == "__main__":
    ds = DeepSeek()
    print("DeepSeek assistant ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        reply = ds.send_message(user_input)
        print(f"DeepSeek: {reply}\n")
