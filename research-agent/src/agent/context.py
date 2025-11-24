import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Context(BaseModel):
    """The context configuration for the agent."""

    # --- LLM 相关 ---
    provider: str = Field(
        default="openai", description="llm provider: openai|anthropic|deepseek|ollama"
    )
    model: str = Field(default="gpt-4.1", description="default chat model")
    api_key: str | None = Field(
        default=None, description="primary API key if needed (e.g., OPENAI_API_KEY)"
    )

    query_generator_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent's answer generation."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
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
