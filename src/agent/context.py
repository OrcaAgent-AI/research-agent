import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Context(BaseModel):
    """The context configuration for the agent."""

    # --- LLM Configuration (Global Defaults) ---
    provider: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL_PROVIDER", "openai"),
        description="LLM provider: openai|anthropic|google-genai|deepseek|ollama"
    )
    
    model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL_NAME", "gpt-4o-mini"),
        description="Default chat model"
    )
    
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("DEFAULT_API_KEY"),
        description="Primary API key (e.g., OPENAI_API_KEY)"
    )
    
    base_url: str | None = Field(
        default_factory=lambda: os.getenv("DEFAULT_BASE_URL"),
        description="API base URL (e.g., https://api.openai.com/v1)"
    )

    # --- Query Generator Configuration ---
    query_generator_model: str = Field(
        default_factory=lambda: os.getenv("QUERY_GENERATOR_MODEL") or os.getenv("DEFAULT_MODEL_NAME", "gpt-4o-mini"),
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )
    
    query_generator_provider: str = Field(
        default_factory=lambda: os.getenv("QUERY_GENERATOR_PROVIDER") or os.getenv("DEFAULT_MODEL_PROVIDER", "openai"),
        metadata={
            "description": "The provider for query generator model."
        },
    )
    
    query_generator_api_key: str | None = Field(
        default_factory=lambda: os.getenv("QUERY_GENERATOR_API_KEY") or os.getenv("DEFAULT_API_KEY"),
        metadata={
            "description": "API key for query generator model."
        },
    )
    
    query_generator_base_url: str | None = Field(
        default_factory=lambda: os.getenv("QUERY_GENERATOR_BASE_URL") or os.getenv("DEFAULT_BASE_URL"),
        metadata={
            "description": "Base URL for query generator model."
        },
    )

    # --- Reflection Model Configuration ---
    reflection_model: str = Field(
        default_factory=lambda: os.getenv("REFLECTION_MODEL") or os.getenv("DEFAULT_MODEL_NAME", "gpt-4o-mini"),
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )
    
    reflection_provider: str = Field(
        default_factory=lambda: os.getenv("REFLECTION_PROVIDER") or os.getenv("DEFAULT_MODEL_PROVIDER", "openai"),
        metadata={
            "description": "The provider for reflection model."
        },
    )
    
    reflection_api_key: str | None = Field(
        default_factory=lambda: os.getenv("REFLECTION_API_KEY") or os.getenv("DEFAULT_API_KEY"),
        metadata={
            "description": "API key for reflection model."
        },
    )
    
    reflection_base_url: str | None = Field(
        default_factory=lambda: os.getenv("REFLECTION_BASE_URL") or os.getenv("DEFAULT_BASE_URL"),
        metadata={
            "description": "Base URL for reflection model."
        },
    )

    # --- Answer Model Configuration ---
    answer_model: str = Field(
        default_factory=lambda: os.getenv("ANSWER_MODEL") or os.getenv("DEFAULT_MODEL_NAME", "gpt-4o"),
        metadata={
            "description": "The name of the language model to use for the agent's answer generation."
        },
    )
    
    answer_provider: str = Field(
        default_factory=lambda: os.getenv("ANSWER_PROVIDER") or os.getenv("DEFAULT_MODEL_PROVIDER", "openai"),
        metadata={
            "description": "The provider for answer model."
        },
    )
    
    answer_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANSWER_API_KEY") or os.getenv("DEFAULT_API_KEY"),
        metadata={
            "description": "API key for answer model."
        },
    )
    
    answer_base_url: str | None = Field(
        default_factory=lambda: os.getenv("ANSWER_BASE_URL") or os.getenv("DEFAULT_BASE_URL"),
        metadata={
            "description": "Base URL for answer model."
        },
    )

    # --- Agent Behavior Configuration ---
    number_of_initial_queries: int = Field(
        default_factory=lambda: int(os.getenv("NUMBER_OF_INITIAL_QUERIES", "3")),
        metadata={
            "description": "The number of initial search queries to generate."
        },
    )

    max_research_loops: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_LOOPS", "2")),
        metadata={
            "description": "The maximum number of research loops to perform."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Context":
        """Create a Context instance from a RunnableConfig.

        Priority: .env file (environment variables) > runtime config > default values
        """
        from typing import Union, get_args, get_origin

        configurable = config["configurable"] if config and "configurable" in config else {}

        values: dict[str, Any] = {}

        for name, field_info in cls.model_fields.items():
            env_key = name.upper()
            env_value = os.environ.get(env_key)

            # Priority 1: Environment variable from .env file
            if env_value is not None:
                # Get the actual type, handling Optional[T] and Union types
                field_type = field_info.annotation
                origin = get_origin(field_type)

                # Handle Optional[T] which is Union[T, None]
                if origin is Union:
                    args = get_args(field_type)
                    # Filter out NoneType to get the actual type
                    actual_types = [arg for arg in args if arg is not type(None)]
                    if actual_types:
                        field_type = actual_types[0]

                # Convert based on type
                try:
                    if field_type is int or field_type == int:
                        values[name] = int(env_value)
                    elif field_type is bool or field_type == bool:
                        values[name] = env_value.lower() in ("true", "1", "yes", "on")
                    elif field_type is float or field_type == float:
                        values[name] = float(env_value)
                    else:
                        # Default: keep as string
                        values[name] = env_value
                except (ValueError, TypeError) as e:
                    # If conversion fails, fall back to string
                    print(
                        f"Warning: Failed to convert {name}={env_value} to {field_type}, using string. Error: {e}"
                    )
                    values[name] = env_value

            # Priority 2: Runtime config
            elif name in configurable:
                values[name] = configurable[name]
            # Priority 3: Use default value (handled by Pydantic)

        return cls(**values)
