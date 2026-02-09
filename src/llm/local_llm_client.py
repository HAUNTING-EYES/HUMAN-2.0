"""
HUMAN 2.0 - Local LLM Client
Wrapper for Ollama API to run models locally on AMD/NVIDIA GPUs.

Supports:
- DeepSeek-Coder 6.7B (recommended for 8GB VRAM)
- CodeLlama 7B
- Qwen 7B

For AMD GPUs, use ollama-for-amd fork:
https://github.com/likelovewant/ollama-for-amd/releases
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import requests
except ImportError:
    requests = None


@dataclass
class GenerationStats:
    """Statistics for a generation request."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    tokens_per_second: float


class LocalLLMClient:
    """
    Client for local LLM inference via Ollama.

    Features:
    - Automatic availability detection
    - Model switching
    - Generation statistics tracking
    - Graceful fallback when unavailable
    """

    DEFAULT_MODEL = "deepseek-coder:6.7b"
    CODING_MODELS = [
        "deepseek-coder:6.7b",
        "deepseek-coder:1.3b",
        "codellama:7b",
        "qwen2.5-coder:7b",
        "starcoder2:3b"
    ]

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: int = 10,
        auto_check: bool = True
    ):
        """
        Initialize local LLM client.

        Args:
            model: Model name to use (e.g., "deepseek-coder:6.7b")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            auto_check: Whether to check availability on init
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Stats tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_tokens_generated = 0

        # Check availability
        self._available = False
        self._available_models: List[str] = []

        if auto_check:
            self._check_availability()

    @property
    def available(self) -> bool:
        """Check if Ollama is available."""
        return self._available

    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        return self._available_models

    def _check_availability(self) -> bool:
        """Check if Ollama server is running and responsive."""
        # Skip Ollama if FORCE_CLOUD_ONLY is set
        if os.environ.get('FORCE_CLOUD_ONLY', '').lower() in ('1', 'true', 'yes'):
            self.logger.info("FORCE_CLOUD_ONLY: Skipping Ollama check")
            self._available = False
            return False

        if requests is None:
            self.logger.warning("requests library not installed")
            self._available = False
            return False

        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )

            if resp.status_code == 200:
                data = resp.json()
                self._available_models = [
                    m.get('name', '') for m in data.get('models', [])
                ]
                self._available = True
                self.logger.info(
                    f"Ollama available with {len(self._available_models)} models"
                )

                # Check if requested model is available
                if self.model not in self._available_models:
                    self.logger.warning(
                        f"Model {self.model} not found. "
                        f"Available: {self._available_models[:5]}"
                    )
                    # Try to find a coding model
                    for coding_model in self.CODING_MODELS:
                        if coding_model in self._available_models:
                            self.logger.info(f"Switching to {coding_model}")
                            self.model = coding_model
                            break

                return True
            else:
                self._available = False
                return False

        except requests.exceptions.ConnectionError:
            self.logger.debug("Ollama not running (connection refused)")
            self._available = False
            return False
        except requests.exceptions.Timeout:
            self.logger.debug("Ollama not responding (timeout)")
            self._available = False
            return False
        except Exception as e:
            self.logger.error(f"Error checking Ollama: {e}")
            self._available = False
            return False

    def refresh_availability(self) -> bool:
        """Refresh availability status."""
        return self._check_availability()

    def pull_model(self, model: str = None) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model to pull (uses self.model if not specified)

        Returns:
            True if successful
        """
        model = model or self.model

        if not self._available:
            self.logger.error("Ollama not available")
            return False

        try:
            self.logger.info(f"Pulling model {model}...")
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=30  # 30 seconds max for local models
            )

            if resp.status_code == 200:
                self.logger.info(f"Model {model} pulled successfully")
                self._check_availability()  # Refresh model list
                return True
            else:
                self.logger.error(f"Failed to pull model: {resp.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        stop: List[str] = None,
        system: str = None,
        timeout: int = None
    ) -> Optional[str]:
        """
        Generate text using local LLM.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop: Stop sequences
            system: System prompt
            timeout: Request timeout (uses self.timeout if not specified)

        Returns:
            Generated text or None if failed
        """
        if not self._available:
            self.logger.debug("Local LLM not available, skipping")
            return None

        if requests is None:
            return None

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self.timeout

        self.total_requests += 1
        start_time = time.time()

        try:
            # Build request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }

            if stop:
                payload["options"]["stop"] = stop

            if system:
                payload["system"] = system

            # Make request
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=request_timeout
            )

            if resp.status_code != 200:
                self.logger.error(f"Ollama error: {resp.status_code} - {resp.text}")
                return None

            data = resp.json()
            response_text = data.get("response", "")

            # Track stats
            self.successful_requests += 1
            duration_ms = (time.time() - start_time) * 1000

            # Ollama provides token counts in response
            eval_count = data.get("eval_count", len(response_text) // 4)
            self.total_tokens_generated += eval_count

            self.logger.debug(
                f"Generated {eval_count} tokens in {duration_ms:.0f}ms "
                f"({eval_count * 1000 / duration_ms:.1f} tok/s)"
            )

            return response_text

        except requests.exceptions.Timeout:
            self.logger.warning(f"Request timed out after {self.timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.warning("Connection lost to Ollama")
            self._available = False
            return None
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return None

    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate code with specialized prompting.

        Args:
            prompt: Code generation prompt
            language: Programming language
            max_tokens: Maximum tokens

        Returns:
            Generated code or None
        """
        system_prompt = f"""You are an expert {language} programmer.
Generate clean, well-documented code.
Return ONLY the code, no explanations.
Ensure the code is syntactically correct."""

        result = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for code
            system=system_prompt
        )

        if result:
            # Extract code from markdown blocks if present
            if f"```{language}" in result:
                result = result.split(f"```{language}")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "available": self._available,
            "model": self.model,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "total_tokens_generated": self.total_tokens_generated,
            "available_models": self._available_models
        }


# Convenience function for quick generation
def quick_generate(prompt: str, model: str = "deepseek-coder:6.7b") -> Optional[str]:
    """Quick generation without persistent client."""
    client = LocalLLMClient(model=model)
    if client.available:
        return client.generate(prompt)
    return None


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    print("Testing Local LLM Client...")
    client = LocalLLMClient()

    print(f"\nAvailable: {client.available}")
    print(f"Model: {client.model}")
    print(f"Available models: {client.available_models}")

    if client.available:
        print("\nTesting generation...")
        result = client.generate_code(
            "Write a Python function to check if a number is prime"
        )
        print(f"\nGenerated code:\n{result}")

        print(f"\nStats: {client.get_stats()}")
    else:
        print("\nOllama not available. Install with:")
        print("  1. Download ollama-for-amd from:")
        print("     https://github.com/likelovewant/ollama-for-amd/releases")
        print("  2. Run: ollama pull deepseek-coder:6.7b")
        print("  3. Start Ollama service")
