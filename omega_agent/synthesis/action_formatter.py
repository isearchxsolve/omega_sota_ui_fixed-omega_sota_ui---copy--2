"""Convert tool execution results into UI-ready ActionDecision."""

import logging
from typing import Any, Dict

from omega_agent.core.types import ActionDecision

logger = logging.getLogger("omega_agent.synthesis.action_formatter")

async def format_execution_results_to_action(
    goal: str,
    domain: str,
    tool_results: Dict[str, Any]
) -> ActionDecision:
    """Formats execution results into beautiful Markdown for the UI."""
    immediate_actions = []
    action_summaries = []
    has_blocked_execution = False
    blocked_messages = []
    
    if not isinstance(tool_results, dict):
        return ActionDecision(action="AWAIT_INPUT", confidence=0.9, rationale="Invalid results.", domain=domain)
    
    for task_id, result in tool_results.items():
        if not isinstance(result, dict):
            continue
            
        if result.get("status") == "awaiting_user_input" or result.get("action_required") == "ask_user":
            has_blocked_execution = True
            if result.get("message"):
                blocked_messages.append(result["message"])
            action_summaries.append(result.get("action_taken", "Paused execution to gather data"))
            continue

        if result.get("success") and result.get("executed_actions"):
            immediate_actions.extend(result.get("executed_actions", []))
        
        if result.get("action_taken"):
            action_summaries.append(result.get("action_taken", ""))
    
    output_lines = []
    
    if has_blocked_execution:
        output_lines.append("## ⏸ Paused for Your Input\n")
        # PRINT THE EXACT MESSAGE TEXT
        for msg in list(dict.fromkeys(blocked_messages)):
            output_lines.append(msg)
            output_lines.append("")
    elif immediate_actions:
        # ACTOR MODE: Report what OMEGA actually executed, not what user should do
        output_lines.append("## ✅ ACTIONS EXECUTED 🔴\n")
        output_lines.append("### OMEGA has taken the following actions:\n")
        
        executed_count = 0
        for action in immediate_actions[:8]:
            execution = action.get("execution", {})
            method = execution.get("method", "unknown")
            
            if method == "browser_automation":
                executed_count += 1
                url = action.get("url", "#")
                title = action.get("title", "Resource")
                detail = action.get("detail", "")
                page_title = execution.get("page_title", "")
                
                output_lines.append(f"✅ **Opened**: [{title}]({url})")
                if detail:
                    output_lines.append(f"   └ {detail}")
                if page_title:
                    output_lines.append(f"   └ Page loaded: {page_title}")
                output_lines.append("")
            elif method == "user_action_required":
                # Phone calls or other actions requiring user intervention
                phone = action.get("phone", "")
                title = action.get("title", "Resource")
                reason = execution.get("reason", "")
                
                output_lines.append(f"📞 **Phone**: {title}")
                if phone:
                    output_lines.append(f"   └ Number: {phone}")
                if reason:
                    output_lines.append(f"   └ Note: {reason}")
                output_lines.append("")
            else:
                # Fallback for link_rendered (old behavior)
                url = action.get("url", "#")
                title = action.get("title", "Resource")
                detail = action.get("detail", "")
                
                output_lines.append(f"🔗 **Link**: [{title}]({url})")
                if detail:
                    output_lines.append(f"   └ {detail}")
                output_lines.append("")
        
        if executed_count > 0:
            output_lines.append(f"\n**OMEGA executed {executed_count} action(s) using browser automation.**")
        else:
            output_lines.append("\n**OMEGA has prepared these resources for you.**")
    
    if action_summaries:
        output_lines.append("\n### What OMEGA Did\n")
        for summary in list(dict.fromkeys(action_summaries)):
            output_lines.append(f"✓ {summary}")
            
    return ActionDecision(
        action="AWAIT_INPUT" if has_blocked_execution else ("REDUCE" if immediate_actions else "COMPLETE"),
        confidence=0.95,
        rationale="\n".join(output_lines) or "Execution complete",
        domain=domain,
        immediate_actions=immediate_actions,
        next_steps=["Reply with your location details above"] if has_blocked_execution else ["Click links above", "Provide missing details if prompted"],
        risk_params={"urgency": "CRITICAL" if has_blocked_execution else "NORMAL"}
    )