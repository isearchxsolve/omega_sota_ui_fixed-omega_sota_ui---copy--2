"""Tool execution with timeout and cost tracking."""

import logging
from typing import Any, Dict, Tuple

from omega_agent.tools.registry import ToolRegistry

logger = logging.getLogger("omega_agent.tools.executor")

TOOL_COSTS = {
    "emergency_food_lookup": 0.004,
    "emergency_cash_lookup": 0.004,
    "emergency_assistance_programs": 0.004,
    "emergency_gig_income": 0.004,
    "write_files": 0.002,
    "modify_file": 0.001,
    "run_shell": 0.003,
    "archive_zip": 0.002,
    "llm_generate_files": 0.015,
    "web_search": 0.001,
    "crypto_price_api": 0.0005,
    "arxiv_search": 0.001,
    "semantic_scholar": 0.001,
    "sentiment_analysis": 0.0001,
    "text_synthesizer": 0.0001,
    "code_generator": 0.005,
    "code_validator": 0.0001,
    "code_executor": 0.002,
    "task_decomposer": 0.0001,
    "send_chat_message": 0.001
}


def handle_tool_failure(tool_name: str, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gracefully handle tool failures by returning a structured error response
    that allows the workflow to continue.
    
    This ensures that individual tool failures don't crash the entire workflow.
    """
    error_msg = str(error)
    error_type = type(error).__name__
    
    logger.warning(
        "Tool %s failed with %s: %s. Workflow will continue with fallback.",
        tool_name,
        error_type,
        error_msg[:200]
    )
    
    # Return a structured error response that indicates failure but allows continuation
    return {
        "success": False,
        "error": error_msg,
        "error_type": error_type,
        "tool_name": tool_name,
        "action_taken": f"Tool {tool_name} failed gracefully - workflow continuing",
        "fallback_triggered": True,
        "requires_alternative": True,
        "status": "failed_gracefully"
    }


class ToolExecutor:
    """Execute registered tools with cost tracking and graceful failure handling."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        domain: str = "general",
    ) -> Tuple[Any, float]:
        handler = self.registry.get_handler(tool_name)
        if not handler:
            logger.warning("Unknown tool: %s", tool_name)
            return {"error": f"Unknown tool: {tool_name}"}, 0.0

        logger.info("Executing tool %s with args %s", tool_name, list(arguments.keys()))
        
        try:
            result = await handler(**arguments)
        except Exception as e:
            # GRACEFUL FAILURE HANDLING: Catch all exceptions and return structured error
            # This prevents individual tool failures from crashing the entire workflow
            result = handle_tool_failure(tool_name, e, arguments)
        
        cost = TOOL_COSTS.get(tool_name, 0.001)

        if isinstance(result, dict) and result.get("action_required") == "ask_user":
            logger.warning(
                "Tool %s blocked — waiting for user input: %s",
                tool_name,
                (result.get("message") or "")[:120],
            )
            result = {
                **result,
                "status": "awaiting_user_input",
                "success": False,
                "action_taken": result.get("action_taken") or f"Paused: {tool_name} needs user input",
            }

        return result, cost
