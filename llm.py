from langchain.chat_models import init_chat_model

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, TIMEOUT


def build_model():
    return init_chat_model(
        MODEL_NAME,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
    )
