from typing import Any, Dict, Optional
from dotenv import load_dotenv
import httpx
from llama_index.embeddings.openai import (OpenAIEmbedding,
                                           OpenAIEmbeddingMode,
                                           OpenAIEmbeddingModelType)
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE
from llama_index.callbacks.base import CallbackManager
from llama_index.bridge.pydantic import PrivateAttr

load_dotenv()


class OpenAIEmbeddingProxy(OpenAIEmbedding):
    """Custom OpenAIEmbedding with proxy http_client"""
    _http_client: httpx.Client = PrivateAttr()

    def __init__(
            self,
            mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
            model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
            additional_kwargs: Optional[Dict[str, Any]] = None,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            max_retries: int = 10,
            timeout: float = 60.0,
            callback_manager: Optional[CallbackManager] = None,
            http_client: Optional[httpx.Client] = None,  # Add http_client as an argument
            **kwargs: Any,
    ) -> None:
        self._http_client = http_client  # Store http_client as an attribute

        super().__init__(
            mode=mode,
            model=model,
            embed_batch_size=embed_batch_size,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        credential_kwargs = super()._get_credential_kwargs()
        credential_kwargs["http_client"] = self._http_client
        return credential_kwargs
