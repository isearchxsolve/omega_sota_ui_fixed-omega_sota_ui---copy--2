"""Centralized failure handler for graceful tool fallback mechanisms.

This module provides utilities to handle tool failures gracefully, allowing
workflows to continue even when individual tools fail. It implements fallback
strategies for different types of tools (browser automation, Twilio, CAPTCHA, etc.).
"""

import logging
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger("omega_agent.tools.failure_handler")


class FailureHandler:
    """
    Centralized failure handler that provides fallback strategies for tool failures.
    
    When a tool fails, this handler determines the best fallback strategy:
    1. Try alternative method (e.g., OCR instead of LLM for CAPTCHA)
    2. Return partial results (e.g., link instead of browser automation)
    3. Mark as user action required (e.g., phone call instead of Twilio)
    4. Continue with degraded functionality
    """
    
    def __init__(self):
        self.failure_counts: Dict[str, int] = {}
        self.max_retries_per_tool = 2
        
    def handle_failure(
        self,
        tool_name: str,
        error: Exception,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a tool failure with appropriate fallback strategy.
        
        Args:
            tool_name: Name of the failed tool
            error: The exception that occurred
            arguments: Arguments passed to the tool
            context: Additional context about the execution
            
        Returns:
            Structured response indicating failure and fallback action
        """
        context = context or {}
        self.failure_counts[tool_name] = self.failure_counts.get(tool_name, 0) + 1
        
        error_msg = str(error)
        error_type = type(error).__name__
        
        # COMPREHENSIVE LOGGING: Log tool failure with full details
        logger.warning(
            "=== TOOL FAILURE DETECTED ===\n"
            "Tool: %s\n"
            "Attempt: %d\n"
            "Error Type: %s\n"
            "Error Message: %s\n"
            "Arguments: %s\n"
            "Determining fallback strategy...",
            tool_name,
            self.failure_counts[tool_name],
            error_type,
            error_msg[:500],
            str(arguments)[:300]
        )
        
        # Determine fallback strategy based on tool type
        fallback_strategy = self._get_fallback_strategy(tool_name, error, arguments)
        
        # COMPREHENSIVE LOGGING: Log the chosen fallback strategy
        logger.info(
            "=== FALLBACK STRATEGY SELECTED ===\n"
            "Tool: %s\n"
            "Strategy: %s\n"
            "Action Taken: %s\n"
            "Requires Alternative: %s\n"
            "Alternative Suggested: %s\n"
            "Workflow will continue gracefully.",
            tool_name,
            fallback_strategy["strategy"],
            fallback_strategy["action_taken"],
            fallback_strategy.get("requires_alternative", True),
            fallback_strategy.get("alternative_suggested", "None")
        )
        
        result = {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "tool_name": tool_name,
            "action_taken": fallback_strategy["action_taken"],
            "fallback_triggered": True,
            "fallback_strategy": fallback_strategy["strategy"],
            "requires_alternative": fallback_strategy.get("requires_alternative", True),
            "status": "failed_gracefully",
            "partial_result": fallback_strategy.get("partial_result"),
            "alternative_suggested": fallback_strategy.get("alternative_suggested")
        }
        
        # Add context if available
        if context:
            result["context"] = context
        
        return result
    
    def _get_fallback_strategy(
        self,
        tool_name: str,
        error: Exception,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine the best fallback strategy for a failed tool.
        
        Returns a dict with:
        - strategy: Name of the fallback strategy
        - action_taken: Human-readable description
        - partial_result: Any partial results that can be used
        - requires_alternative: Whether an alternative path is needed
        - alternative_suggested: Suggested alternative action
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Browser automation failures
        if "browser" in tool_name.lower() or "navigate" in tool_name.lower():
            return self._browser_automation_fallback(error, arguments)
        
        # Twilio/phone call failures
        if "phone" in tool_name.lower() or "twilio" in tool_name.lower() or "call" in tool_name.lower():
            return self._twilio_fallback(error, arguments)
        
        # CAPTCHA solving failures
        if "captcha" in tool_name.lower():
            return self._captcha_fallback(error, arguments)
        
        # Web search failures
        if "search" in tool_name.lower():
            return self._search_fallback(error, arguments)
        
        # File operation failures
        if "file" in tool_name.lower() or "write" in tool_name.lower() or "modify" in tool_name.lower():
            return self._file_operation_fallback(error, arguments)
        
        # Generic fallback for unknown tools
        return self._generic_fallback(tool_name, error, arguments)
    
    def _browser_automation_fallback(self, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy for browser automation failures."""
        url = arguments.get("url", "")
        
        if "import" in str(error).lower() or "module" in str(error).lower():
            return {
                "strategy": "link_rendered",
                "action_taken": f"Browser automation not available - rendering link for user to open: {url}",
                "partial_result": {"url": url, "method": "link_rendered"},
                "requires_alternative": False,
                "alternative_suggested": "User can manually open the link"
            }
        
        return {
            "strategy": "link_rendered",
            "action_taken": f"Browser automation failed - rendering link for user: {url}",
            "partial_result": {"url": url, "method": "link_rendered"},
            "requires_alternative": False,
            "alternative_suggested": "User can manually open the link"
        }
    
    def _twilio_fallback(self, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy for Twilio/phone call failures."""
        phone = arguments.get("phone_number", "")
        
        if "credential" in str(error).lower() or "auth" in str(error).lower():
            return {
                "strategy": "user_action_required",
                "action_taken": f"Twilio credentials not configured - phone number {phone} ready for manual call",
                "partial_result": {"phone": phone, "method": "user_action_required"},
                "requires_alternative": True,
                "alternative_suggested": f"User should manually call {phone}"
            }
        
        return {
            "strategy": "user_action_required",
            "action_taken": f"Phone call failed - phone number {phone} ready for manual call",
            "partial_result": {"phone": phone, "method": "user_action_required"},
            "requires_alternative": True,
            "alternative_suggested": f"User should manually call {phone}"
        }
    
    def _captcha_fallback(self, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy for CAPTCHA solving failures."""
        # Already has built-in fallback (LLM -> OCR), so this is final fallback
        return {
            "strategy": "manual_intervention",
            "action_taken": "CAPTCHA solving failed - manual intervention required",
            "partial_result": None,
            "requires_alternative": True,
            "alternative_suggested": "User should solve CAPTCHA manually"
        }
    
    def _search_fallback(self, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy for search failures."""
        query = arguments.get("query", "")
        
        return {
            "strategy": "continue_without_results",
            "action_taken": f"Search failed for query '{query}' - continuing with available information",
            "partial_result": {"query": query, "results": []},
            "requires_alternative": False,
            "alternative_suggested": "Proceed with available context or ask user for information"
        }
    
    def _file_operation_fallback(self, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy for file operation failures."""
        filepath = arguments.get("filepath", arguments.get("path", ""))
        
        if "permission" in str(error).lower():
            return {
                "strategy": "skip_file",
                "action_taken": f"File operation failed due to permissions - skipping: {filepath}",
                "partial_result": None,
                "requires_alternative": False,
                "alternative_suggested": "Continue with other files or ask user to check permissions"
            }
        
        return {
            "strategy": "skip_file",
            "action_taken": f"File operation failed - skipping: {filepath}",
            "partial_result": None,
            "requires_alternative": False,
            "alternative_suggested": "Continue with other operations"
        }
    
    def _generic_fallback(self, tool_name: str, error: Exception, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generic fallback strategy for unknown tool failures."""
        return {
            "strategy": "continue_without_tool",
            "action_taken": f"Tool {tool_name} failed - continuing workflow without its output",
            "partial_result": None,
            "requires_alternative": False,
            "alternative_suggested": "Proceed with available information and tools"
        }
    
    def should_retry(self, tool_name: str) -> bool:
        """Determine if a tool should be retried based on failure count."""
        return self.failure_counts.get(tool_name, 0) < self.max_retries_per_tool
    
    def reset_failure_count(self, tool_name: str) -> None:
        """Reset failure count for a tool (e.g., after successful execution)."""
        if tool_name in self.failure_counts:
            del self.failure_counts[tool_name]


# Global singleton instance
_failure_handler_instance: Optional[FailureHandler] = None


def get_failure_handler() -> FailureHandler:
    """Get or create the global failure handler instance."""
    global _failure_handler_instance
    if _failure_handler_instance is None:
        _failure_handler_instance = FailureHandler()
    return _failure_handler_instance
