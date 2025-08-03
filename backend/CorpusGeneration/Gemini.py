import abc
import os
import time
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError, InvalidArgument

from LLM import ChatSession


class GeminiChat(ChatSession):
    """
    Google Gemini implementation of ChatSession using the GenerativeModel API.
    """
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash-preview-05-20", temperature: float = 0.7):

        resolved_api_key = api_key if api_key else os.environ.get("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY environment variable not set.")
        
        try:
            genai.configure(api_key=resolved_api_key)
        except Exception as e:
            # This can catch issues if the API key is invalid format before any API call
            raise ValueError(f"Failed to configure Google AI SDK: {e}")
        
        self.model_name = model
        self.temperature = temperature
        
        # Define generation configuration
        self.generation_config = GenerationConfig(
            temperature=self.temperature
            # You can add other parameters like top_p, top_k, max_output_tokens here
            # e.g., max_output_tokens=2048
        )

        try:
            self.model = genai.GenerativeModel(
                self.model_name
            )
        except Exception as e:
            # Catch errors during model initialization (e.g., invalid model name)
            raise ValueError(f"Failed to initialize GenerativeModel '{self.model_name}': {e}")


    def send(self, prompt: str) -> tuple[str, str | None]:
        """
        Send a single-prompt string to the Gemini LLM and return (response, error_code).
        Retries on rate limits (ResourceExhausted).

        Args:
            prompt: The user's prompt string.

        Returns:
            A tuple containing the LLM's response text (str) and an error message (str | None).
            If successful, the error message is None.
        """
        try:
            # For single-turn, generate_content is appropriate.
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            
            # Check if the prompt itself was blocked
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason.name
                detailed_ratings = ""
                if response.prompt_feedback.safety_ratings:
                    detailed_ratings = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in response.prompt_feedback.safety_ratings])
                return "", f"PromptBlocked: The prompt was blocked. Reason: {block_reason_msg}. Details: {detailed_ratings if detailed_ratings else 'N/A'}"

            # Check if candidates exist and process the first one
            if not response.candidates:
                # This case might occur if prompt_feedback didn't catch a block, or other issues.
                return "", "NoCandidates: The model returned no candidates. The prompt might have been blocked or an issue occurred."

            candidate = response.candidates[0]
            
            # Check finish reason for the generated content
            # Common finish reasons: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
            if candidate.finish_reason.name not in ("STOP", "MAX_TOKENS"):
                error_message = f"GenerationIssue: Content generation finished due to {candidate.finish_reason.name}."
                if candidate.finish_reason.name == "SAFETY":
                    safety_ratings_info = "Blocked due to safety concerns in generated content."
                    if candidate.safety_ratings:
                        safety_ratings_info += " Details: " + ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in candidate.safety_ratings])
                    error_message = f"ContentBlocked: {safety_ratings_info}"
                return "", error_message

            # Safely extract text content
            if candidate.content and candidate.content.parts:
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if not text_parts:
                     return "", "NoTextContent: Model returned content parts but no text."
                full_text = "".join(text_parts)
                return full_text.strip(), None
            else:
                # This state (candidate exists, finish_reason is OK, but no content/parts) should be rare
                return "", "NoContentParts: Model returned a candidate but no parsable content parts."

        except ResourceExhausted as e:
            # Specific error for rate limits
            # print(f"Rate limit hit for Gemini. Retrying in 3 seconds. Error: {e}") # Optional: for debugging
            print(e)
            time.sleep(3)
            return self.send(prompt) # Recursive call for retry
        except InvalidArgument as e:
            # Often related to invalid API key, malformed request, or unsupported model features
            return "", f"InvalidArgumentError: {e}"
        except GoogleAPIError as e:
            # General Google API error
            return "", f"GoogleAPIError: {e}"
        except Exception as e:
            # Catch-all for other unexpected errors during API call or response processing
            return "", f"Error: {e}"

if __name__ == '__main__':
    # Example Usage (requires GOOGLE_API_KEY to be set in environment or passed)
    # Ensure you have the google-generativeai package installed: pip install google-generativeai

    print("Attempting to initialize GeminiChatSession...")
    try:
        # To test, set your GOOGLE_API_KEY environment variable
        # or pass it directly: api_key="YOUR_GOOGLE_API_KEY"
        gemini_session = GeminiChat(model="gemini-2.5-flash-latest") # Using flash for faster, cheaper tests
        print("GeminiChatSession initialized successfully.")

        prompt1 = "Hello, Gemini! Tell me a fun fact about programming."
        print(f"\nSending prompt: '{prompt1}'")
        response_text, error = gemini_session.send(prompt1)

        if error:
            print(f"Error: {error}")
        else:
            print(f"Gemini Response: {response_text}")

    except ValueError as ve:
        print(f"Initialization Error: {ve}")
        print("Please ensure GOOGLE_API_KEY is set or passed correctly, and the model name is valid.")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")

