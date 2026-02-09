"""
HUMAN 2.0 - Cloud LLM Client
Primary provider: Anthropic Claude API for complex tasks.

Hybrid Architecture:
- Local LLM (Ollama): Simple tasks (FREE)
- Claude API: Complex tasks (paid but high quality)

Cost tracking included for budget management.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CloudProvider(Enum):
    """Supported cloud providers."""
    ANTHROPIC = "anthropic"  # Primary - Claude API
    OPENAI = "openai"        # Fallback option


@dataclass
class UsageRecord:
    """Record of API usage for cost tracking."""
    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_estimate: float  # in INR or USD depending on provider


@dataclass
class CloudConfig:
    """Configuration for cloud LLM."""
    provider: CloudProvider
    api_key: str
    model: str
    base_url: Optional[str] = None
    cost_per_1k_input: float = 0.0  # Cost per 1000 input tokens
    cost_per_1k_output: float = 0.0  # Cost per 1000 output tokens
    max_daily_cost: float = 100.0  # Daily budget limit


class CloudLLMClient:
    """
    Client for cloud-based LLM inference.

    Features:
    - Multi-provider support with automatic fallback
    - Cost tracking and budget limits
    - Request caching
    - Rate limiting
    """

    # Default models per provider
    DEFAULT_MODELS = {
        CloudProvider.ANTHROPIC: "claude-sonnet-4-20250514",  # Best for coding
        CloudProvider.OPENAI: "gpt-4o"  # Fallback
    }

    # Cost estimates (per 1000 tokens, USD)
    COST_ESTIMATES = {
        CloudProvider.ANTHROPIC: {"input": 3.0, "output": 15.0},
        CloudProvider.OPENAI: {"input": 2.5, "output": 10.0},
    }

    def __init__(
        self,
        primary_provider: CloudProvider = None,
        fallback_providers: List[CloudProvider] = None,
        max_daily_cost: float = 100.0
    ):
        """
        Initialize cloud LLM client.

        Args:
            primary_provider: Primary provider to use
            fallback_providers: Providers to try if primary fails
            max_daily_cost: Maximum daily spend limit
        """
        self.logger = logging.getLogger(__name__)

        # Determine providers based on available API keys
        self.configs: Dict[CloudProvider, CloudConfig] = {}
        self._setup_providers()

        # Set primary and fallbacks
        if primary_provider and primary_provider in self.configs:
            self.primary_provider = primary_provider
        elif CloudProvider.ANTHROPIC in self.configs:
            self.primary_provider = CloudProvider.ANTHROPIC
        elif self.configs:
            self.primary_provider = list(self.configs.keys())[0]
        else:
            self.primary_provider = None
            self.logger.warning("No cloud providers configured")

        self.fallback_providers = fallback_providers or [
            p for p in self.configs.keys() if p != self.primary_provider
        ]

        # Usage tracking
        self.usage_records: List[UsageRecord] = []
        self.max_daily_cost = max_daily_cost
        self.daily_cost = 0.0
        self.last_reset_date = datetime.now().date()

        # Stats
        self.total_requests = 0
        self.successful_requests = 0

        self.logger.info(
            f"CloudLLMClient initialized. Primary: {self.primary_provider}, "
            f"Fallbacks: {self.fallback_providers}"
        )

    def _setup_providers(self):
        """Setup available providers based on environment variables."""

        # Primary: Anthropic Claude API
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.configs[CloudProvider.ANTHROPIC] = CloudConfig(
                provider=CloudProvider.ANTHROPIC,
                api_key=anthropic_key,
                model=self.DEFAULT_MODELS[CloudProvider.ANTHROPIC],
                base_url="https://api.anthropic.com",
                cost_per_1k_input=self.COST_ESTIMATES[CloudProvider.ANTHROPIC]["input"],
                cost_per_1k_output=self.COST_ESTIMATES[CloudProvider.ANTHROPIC]["output"]
            )
            self.logger.info("Anthropic Claude configured as primary cloud provider")

        # Fallback: OpenAI (optional)
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.configs[CloudProvider.OPENAI] = CloudConfig(
                provider=CloudProvider.OPENAI,
                api_key=openai_key,
                model=self.DEFAULT_MODELS[CloudProvider.OPENAI],
                base_url="https://api.openai.com/v1",
                cost_per_1k_input=self.COST_ESTIMATES[CloudProvider.OPENAI]["input"],
                cost_per_1k_output=self.COST_ESTIMATES[CloudProvider.OPENAI]["output"]
            )
            self.logger.info("OpenAI configured as fallback provider")

    @property
    def available(self) -> bool:
        """Check if any cloud provider is available."""
        return len(self.configs) > 0

    def _check_budget(self) -> bool:
        """Check if daily budget allows another request."""
        # Reset if new day
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_cost = 0.0
            self.last_reset_date = today

        if self.daily_cost >= self.max_daily_cost:
            self.logger.warning(f"Daily budget exceeded: {self.daily_cost:.2f}")
            return False

        return True

    def _estimate_cost(
        self,
        provider: CloudProvider,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost for a request."""
        config = self.configs.get(provider)
        if not config:
            return 0.0

        input_cost = (prompt_tokens / 1000) * config.cost_per_1k_input
        output_cost = (completion_tokens / 1000) * config.cost_per_1k_output

        return input_cost + output_cost

    def _record_usage(
        self,
        provider: CloudProvider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Record usage for tracking."""
        cost = self._estimate_cost(provider, prompt_tokens, completion_tokens)
        self.daily_cost += cost

        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider.value,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_estimate=cost
        )
        self.usage_records.append(record)

        # Keep only last 1000 records
        if len(self.usage_records) > 1000:
            self.usage_records = self.usage_records[-1000:]

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system: str = None,
        provider: CloudProvider = None
    ) -> Optional[str]:
        """
        Generate text using cloud LLM.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            provider: Specific provider to use (uses primary if not specified)

        Returns:
            Generated text or None if failed
        """
        if not self._check_budget():
            return None

        # Determine providers to try
        providers_to_try = []
        if provider and provider in self.configs:
            providers_to_try = [provider]
        else:
            if self.primary_provider:
                providers_to_try.append(self.primary_provider)
            providers_to_try.extend(self.fallback_providers)

        self.total_requests += 1

        # Try each provider
        for prov in providers_to_try:
            config = self.configs.get(prov)
            if not config:
                continue

            try:
                result = await self._call_provider(
                    provider=prov,
                    config=config,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system
                )

                if result:
                    self.successful_requests += 1
                    return result

            except Exception as e:
                self.logger.warning(f"Provider {prov} failed: {e}")
                continue

        self.logger.error("All cloud providers failed")
        return None

    async def _call_provider(
        self,
        provider: CloudProvider,
        config: CloudConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system: str = None
    ) -> Optional[str]:
        """Call a specific provider's API."""

        if provider == CloudProvider.ANTHROPIC:
            return await self._call_anthropic(config, prompt, max_tokens, temperature, system)
        elif provider == CloudProvider.OPENAI:
            return await self._call_openai(config, prompt, max_tokens, temperature, system)
        else:
            return None

    async def _call_anthropic(
        self,
        config: CloudConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system: str = None
    ) -> Optional[str]:
        """Call Anthropic Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=config.api_key)

            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }

            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)

            # Record usage
            self._record_usage(
                CloudProvider.ANTHROPIC,
                config.model,
                response.usage.input_tokens,
                response.usage.output_tokens
            )

            return response.content[0].text

        except ImportError:
            self.logger.error("anthropic package not installed")
            return None
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise

    async def _call_openai(
        self,
        config: CloudConfig,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system: str = None
    ) -> Optional[str]:
        """Call OpenAI API."""
        try:
            import openai

            client = openai.OpenAI(api_key=config.api_key)

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Record usage
            self._record_usage(
                CloudProvider.OPENAI,
                config.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

            return response.choices[0].message.content

        except ImportError:
            self.logger.error("openai package not installed")
            return None
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "available": self.available,
            "primary_provider": self.primary_provider.value if self.primary_provider else None,
            "configured_providers": [p.value for p in self.configs.keys()],
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "daily_cost": self.daily_cost,
            "max_daily_cost": self.max_daily_cost,
            "budget_remaining": self.max_daily_cost - self.daily_cost
        }

    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by provider."""
        breakdown = {}
        for record in self.usage_records:
            provider = record.provider
            if provider not in breakdown:
                breakdown[provider] = 0.0
            breakdown[provider] += record.cost_estimate
        return breakdown


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("Testing Cloud LLM Client...")
        client = CloudLLMClient()

        print(f"\nAvailable: {client.available}")
        print(f"Primary: {client.primary_provider}")
        print(f"Stats: {client.get_stats()}")

        if client.available:
            print("\nTesting generation...")
            result = await client.generate(
                prompt="Write a Python function to calculate factorial",
                max_tokens=500
            )
            print(f"Result: {result[:200] if result else 'None'}...")
            print(f"\nUpdated stats: {client.get_stats()}")

    asyncio.run(test())
