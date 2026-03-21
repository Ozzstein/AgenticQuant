"""Exception hierarchy for QuantAgentLab."""


class AiQuantError(Exception):
    """Base exception for all QuantAgentLab errors."""


class ConfigError(AiQuantError):
    """Configuration loading or validation error."""


class DataPipelineError(AiQuantError):
    """Error in data ingestion or processing."""


class FactorError(AiQuantError):
    """Error in factor computation or evaluation."""


class ModelError(AiQuantError):
    """Error in model training, prediction, or loading."""


class BacktestError(AiQuantError):
    """Error in backtesting engine."""


class RegimeError(AiQuantError):
    """Error in regime detection or HMM model operations."""


class AgentError(AiQuantError):
    """Error in agent node execution."""


class LLMError(AgentError):
    """Error communicating with LLM provider."""


class ToolError(AgentError):
    """Error in LangChain tool execution."""


class ExecutionError(AiQuantError):
    """Error in order execution or portfolio management."""


class RiskBreachError(ExecutionError):
    """Pre-trade risk check violation."""


class OrderError(ExecutionError):
    """Order submission or lifecycle error."""
