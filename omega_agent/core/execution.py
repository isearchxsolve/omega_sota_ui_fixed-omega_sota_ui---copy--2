"""DAG execution engine with parallel wave scheduling AND universal pre-flight validation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from omega_agent.core.config import Config
from omega_agent.core.types import ExecutionContext, ExecutionStatus, TaskNode
from omega_agent.tools.executor import ToolExecutor

logger = logging.getLogger("omega_agent.execution")

def compute_execution_waves(dag: List[TaskNode]) -> List[List[TaskNode]]:
    waves = []
    remaining = {t.id: t for t in dag}
    completed = set()
    while remaining:
        wave = []
        for task_id, task in list(remaining.items()):
            if all(dep in completed for dep in task.dependencies):
                wave.append(task)
        if not wave:
            wave = list(remaining.values())
        waves.append(wave)
        for task in wave:
            completed.add(task.id)
            del remaining[task.id]
    return waves

class DAGExecutor:
    def __init__(self, config: Config, tool_executor: ToolExecutor):
        self.config = config
        self.tool_executor = tool_executor

    def _get_missing_params(self, task: TaskNode) -> List[str]:
        """Universally catch missing or hallucinated empty arguments across ALL tools."""
        missing = []
        placeholders = {
            "", "none", "null", "undefined", "tbd", "unknown", 
            "[missing]", "<missing>", "n/a", "not provided", "pending",
            "todo", "fillme", "required", "insert_here"
        }
        
        for k, v in task.arguments.items():
            val_str = str(v).strip().lower()
            
            # Check for placeholder values
            if val_str in placeholders or val_str.startswith("<insert") or val_str.startswith("[insert"):
                missing.append(k)
                continue
            
            # Special check for location-type parameters
            location_fields = ("location", "zip_code", "zip", "address", "city", "region", "area")
            generic_locations = (
                "united states", "usa", "us", "my area", "nearby", "here", 
                "anywhere", "my location", "current location", "there"
            )
            
            if k.lower() in location_fields and val_str in generic_locations:
                missing.append(k)
                continue
            
            # Check for incomplete location strings
            if k.lower() in location_fields and len(val_str) < 3:
                missing.append(k)
                continue
            
            # Check for obviously incomplete parameters (too short)
            if k.lower() in ("zip", "zip_code", "postal_code") and len(val_str) < 4:
                missing.append(k)
                continue
            
            # Check for incomplete email/phone patterns
            if k.lower() in ("email", "phone", "contact") and len(val_str) < 8:
                if "@" not in val_str and len(val_str) < 10:
                    missing.append(k)
                    continue
        
        return missing

    async def _execute_task(self, task: TaskNode, ctx: ExecutionContext, cost_lock: asyncio.Lock, attempt: int = 1) -> Any:
        if attempt > task.max_retries:
            raise RuntimeError(f"Task {task.id} failed after {task.max_retries} retries")

        # =========================================================================
        # UNIVERSAL PAUSE 1: The LLM explicitly chose to ask the user a question
        # =========================================================================
        # ACTOR MODE: Don't halt execution for send_chat_message unless it's requesting critical missing info
        # Allow OMEGA to continue executing actions while also informing the user
        if task.tool_name == "send_chat_message":
            msg = task.arguments.get("message", "I need more information to proceed.")
            urgency = task.arguments.get("urgency", "normal")
            
            # Only halt if urgency is "high" (critical missing info)
            if urgency.lower() == "high":
                try:
                    await self.tool_executor.execute("send_chat_message", task.arguments, ctx.domain or "general")
                except Exception:
                    pass
                return {
                    "status": "awaiting_user_input",
                    "blocked_tool": "send_chat_message",
                    "action_taken": "Paused to ask the user a question.",
                    "message": msg
                }
            else:
                # Low urgency: send message but continue execution
                try:
                    await self.tool_executor.execute("send_chat_message", task.arguments, ctx.domain or "general")
                except Exception:
                    pass
                # Return success but note that message was sent
                return {
                    "success": True,
                    "action_taken": f"Sent message to user: {msg[:100]}...",
                    "chat_message_sent": True,
                    "status": "complete"
                }

        # =========================================================================
        # UNIVERSAL PAUSE 2: The tool is missing required arguments
        # =========================================================================
        missing = self._get_missing_params(task)
        if missing:
            msg = f"🔴 **I need more information to run `{task.tool_name}`.**\n\nPlease provide: **{', '.join(missing)}**"
            try:
                await self.tool_executor.execute("send_chat_message", {"message": msg, "urgency": "high"}, ctx.domain or "general")
            except Exception:
                pass
            return {
                "status": "awaiting_user_input",
                "blocked_tool": task.tool_name,
                "action_taken": f"Paused execution. Missing parameters: {', '.join(missing)}",
                "message": msg
            }

        # Normal execution
        task.status = ExecutionStatus.RUNNING
        try:
            result, cost = await asyncio.wait_for(
                self.tool_executor.execute(task.tool_name, task.arguments, ctx.domain or "general"),
                timeout=task.timeout
            )
            async with cost_lock:
                ctx.add_cost(cost)
            return result
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")

        if attempt < task.max_retries:
            await asyncio.sleep(1.0 * attempt)
            return await self._execute_task(task, ctx, cost_lock, attempt + 1)
        raise RuntimeError(f"Task {task.id} exhausted retries")

    async def _run_task_safe(self, task: TaskNode, ctx: ExecutionContext, cost_lock: asyncio.Lock) -> tuple[Any, Optional[Exception]]:
        """
        Execute a task safely with graceful failure handling.
        
        If a task fails, returns a structured error response that allows
        the workflow to continue with fallback strategies instead of crashing.
        """
        try:
            res = await self._execute_task(task, ctx, cost_lock)
            return res, None
        except Exception as e:
            # GRACEFUL FAILURE HANDLING: Return structured error instead of crashing
            logger.error(
                "Task %s (%s) failed: %s. Workflow will continue with fallback.",
                task.id,
                task.tool_name,
                str(e)[:200]
            )
            
            # Create a fallback result that indicates failure but allows continuation
            fallback_result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "tool_name": task.tool_name,
                "task_id": task.id,
                "action_taken": f"Task {task.tool_name} failed gracefully - workflow continuing",
                "fallback_triggered": True,
                "requires_alternative": True,
                "status": "failed_gracefully"
            }
            
            return fallback_result, e

    async def execute(self, dag: List[TaskNode], ctx: ExecutionContext) -> Dict[str, Any]:
        """
        Execute the DAG with graceful failure handling.
        
        GRACEFUL FAILURE HANDLING: If any task fails, the workflow continues
        with fallback results instead of crashing. Failed tasks are logged
        and their fallback results are stored so downstream tasks can proceed.
        """
        waves = compute_execution_waves(dag)
        results: Dict[str, Any] = {}
        cost_lock = asyncio.Lock()
        
        original_callback = getattr(self.tool_executor, '_ui_callback', None)
        self.tool_executor._ui_callback = ctx.ui_event_handler
        self.tool_executor._progress_queue = getattr(ctx.run_progress, '_queue', None) if ctx.run_progress else None

        try:
            for wave in waves:
                outcomes = await asyncio.gather(*[self._run_task_safe(t, ctx, cost_lock) for t in wave])
                
                wave_requires_input = False
                for task, (result, error) in zip(wave, outcomes):
                    # GRACEFUL FAILURE HANDLING: Store both successful and failed results
                    # Failed tasks return a fallback_result that allows workflow to continue
                    if result is not None:
                        results[task.id] = result
                        ctx.task_results[task.id] = result
                        
                        if error:
                            # Task failed but we have a fallback result
                            task.status = ExecutionStatus.FAILED
                            task.result = result
                            logger.warning(
                                "Task %s (%s) failed but workflow continuing with fallback",
                                task.id,
                                task.tool_name
                            )
                        else:
                            # Task succeeded
                            task.status = ExecutionStatus.SUCCESS
                            task.result = result
                            
                            if isinstance(result, dict) and result.get("status") == "awaiting_user_input":
                                wave_requires_input = True
                    else:
                        # No result at all (shouldn't happen with our fallback handling)
                        logger.error("Task %s returned no result", task.id)
                        results[task.id] = {
                            "success": False,
                            "error": "No result returned",
                            "status": "failed_gracefully"
                        }

                # =========================================================================
                # UNIVERSAL FIX: Stop executing future tasks if we are pausing for user input
                # =========================================================================
                if wave_requires_input:
                    logger.info("Halting further DAG execution to await user input.")
                    break

            try:
                from omega_agent.synthesis.action_formatter import format_execution_results_to_action
                decision = await format_execution_results_to_action(ctx.goal, ctx.domain or "general", results)
                
                if ctx.ui_event_handler:
                    try:
                        from omega_agent.api.sse_output import stream_action_decision
                        await stream_action_decision(
                            decision, 
                            event_callback=ctx.ui_event_handler,
                            progress_queue=getattr(ctx.run_progress, '_queue', None) if ctx.run_progress else None
                        )
                    except ImportError:
                        pass
                
                ctx.decision = decision
            except ImportError:
                logger.warning("Action formatter module not available")
                
            return results
        finally:
            self.tool_executor._ui_callback = original_callback