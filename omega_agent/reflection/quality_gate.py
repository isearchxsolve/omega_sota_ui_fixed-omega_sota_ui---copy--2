"""Quality gate verification and SOTA guarantee — Domain-agnostic validation."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from omega_agent.core.config import Config
from omega_agent.core.orchestrator import ModelOrchestrator
from omega_agent.core.types import AgentResult
from omega_agent.reasoning.types import DynamicDomainProfile
from omega_agent.tools.executor import ToolExecutor

logger = logging.getLogger("omega_agent.reflection.quality_gate")

class QualityGate:
    """Validate that generated code meets STRICT production standards."""

    @staticmethod
    def verify_zoho_crm_application(generated_files: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        if not generated_files:
            return False, "No files generated", {}

        file_contents = [v for v in generated_files.values() if isinstance(v, str)]
        all_content = "\n".join(file_contents)
        details = {}

        lines = [l for l in all_content.split('\n') if l.strip() and not l.strip().startswith('//')]
        line_count = len(lines)
        
        details['line_count'] = line_count
        details['file_count'] = len(generated_files)
        
        # SOTA STANDARD: Require at least 500 lines for a multi-file CRM
        if line_count < 500:
            return False, (
                f"Code volume too low for SOTA standards: {line_count} lines. "
                f"A true production CRM requires 500+ lines of dense implementation."
            ), details

        model_patterns = {
            "accounts": r"class\s+Account|model\s+Account|Account\s*\{|interface\s+Account",
            "contacts": r"class\s+Contact|model\s+Contact|Contact\s*\{|interface\s+Contact",
            "deals": r"class\s+Deal|model\s+Deal|Deal\s*\{|interface\s+Deal|Opportunity",
            "tasks": r"class\s+Task|model\s+Task|Task\s*\{|interface\s+Task",
            "users": r"class\s+User|model\s+User|User\s*\{|interface\s+User",
        }
        
        found_models = {}
        for model_name, pattern in model_patterns.items():
            if re.search(pattern, all_content, re.IGNORECASE):
                found_models[model_name] = True
        
        details['data_models'] = found_models
        
        if len(found_models) < 3:
            return False, (
                f"Missing domain logic: found only {list(found_models.keys())}. "
                f"SOTA CRM requires at least 3 core models (e.g., Accounts, Contacts, Deals)."
            ), details

        has_multi_tenant = any(re.search(p, all_content, re.IGNORECASE) for p in [r"tenant_id", r"workspace_id", r"multi[\s_]?tenant"])
        has_auth = any(re.search(p, all_content, re.IGNORECASE) for p in [r"jwt", r"passport", r"oauth", r"login"])
        has_api = any(re.search(p, all_content, re.IGNORECASE) for p in [r"\.(get|post|put|delete)\s*\(", r"endpoint", r"/api/"])
        has_ui = any(re.search(p, all_content, re.IGNORECASE) for p in [r"react", r"vue", r"jsx", r"angular"])

        all_checks = {
            'multi_tenant': has_multi_tenant,
            'authentication': has_auth,
            'api_endpoints': has_api,
            'ui_framework': has_ui,
        }
        
        if all(all_checks.values()):
            return True, (
                f"✓ SOTA Production CRM generated ({line_count} lines, {len(generated_files)} files) "
                f"with {list(found_models.keys())} data models, multi-tenant architecture, auth, API, and UI."
            ), details
        else:
            failed = [k for k, v in all_checks.items() if not v]
            return False, f"Missing critical components: {', '.join(failed)}. Rejected by SOTA quality gate.", details

    @staticmethod
    def verify_generic_build(generated_files: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        file_contents = [v for v in generated_files.values() if isinstance(v, str)]
        all_content = "\n".join(file_contents)
        lines = [l for l in all_content.split('\n') if l.strip() and not l.strip().startswith('//')]
        
        if len(lines) < 300:
            return False, f"Code volume too low ({len(lines)} lines). SOTA generic builds require 300+ lines.", {}

        has_logic = bool(re.search(r"function|class|def|const.*=>", all_content))
        if not has_logic:
            return False, "No core logic found in generated code", {}

        return True, f"✓ SOTA generic application generated ({len(lines)} lines)", {}

    @staticmethod
    def create_rejection_guidance(goal: str, reason: str) -> str:
        return (
            f"QUALITY GATE FAILED: {reason}\n\n"
            f"CRITICAL SOTA REQUIREMENTS FOR REGENERATION:\n"
            f"- You MUST generate a massive, feature-complete application.\n"
            f"- ZERO boilerplate. Implement every function, model, and route fully.\n"
            f"- Ensure dense implementation exceeding 500+ lines of code.\n"
        )

class DeliverableValidator:
    def __init__(self, config):
        self.config = config
        self.quality_gate = QualityGate()

    async def validate_and_regenerate_if_needed(self, goal: str, generated_files: Dict[str, str], profile: Any, orchestrator: Any):
        is_zoho_like = any(word in goal.lower() for word in ("zoho", "crm", "erp", "saas", "platform"))
        if is_zoho_like:
            passed, message, details = self.quality_gate.verify_zoho_crm_application(generated_files)
        else:
            passed, message, details = self.quality_gate.verify_generic_build(generated_files)
        
        logger.info(f"Quality gate result: {message}")
        if passed:
            return True, generated_files, message
        
        return False, generated_files, self.quality_gate.create_rejection_guidance(goal, message)

# ── Domain-Agnostic SOTA Quality Gate ───────────────────────────────────────────

class SOTAQualityGate:
    """
    Domain-agnostic quality gate that validates SOTA results across ANY domain.
    
    This class provides universal validation criteria that apply to all domains:
    - emergency, coding, crypto_trading, research, planning, etc.
    
    Validation includes:
    1. Result completeness and structure
    2. Actionability and execution readiness
    3. Domain-specific quality metrics (inferred from profile)
    4. SOTA standards (volume, complexity, production readiness)
    """
    
    def __init__(self, config: Config, orchestrator: ModelOrchestrator, tool_executor: ToolExecutor):
        self.config = config
        self.orchestrator = orchestrator
        self.tool_executor = tool_executor
        
        # Domain-agnostic quality thresholds
        self.min_line_count = 100  # Minimum lines for any substantial result
        self.min_action_count = 1  # At least one actionable item
        self.sota_line_count = 300  # SOTA threshold for substantial work
        self.sota_action_count = 3  # SOTA threshold for multiple actions
    
    def evaluate(self, result: AgentResult, profile: DynamicDomainProfile) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate an AgentResult against SOTA standards across ANY domain.
        
        Args:
            result: The agent result to evaluate
            profile: Dynamic domain profile providing context
        
        Returns:
            Tuple of (quality_score, evaluation_details)
        """
        evaluation_details = {
            "domain": getattr(profile, 'domain', 'unknown'),
            "checks": {},
            "passed": True,
            "score": 0.0
        }
        
        score_components = []
        
        # Check 1: Result exists and is not empty
        has_result = self._check_result_exists(result)
        evaluation_details["checks"]["result_exists"] = has_result
        score_components.append(1.0 if has_result else 0.0)
        
        # Check 2: Result has actionable content
        has_actions = self._check_actionable_content(result)
        evaluation_details["checks"]["has_actions"] = has_actions
        score_components.append(1.0 if has_actions else 0.0)
        
        # Check 3: Result completeness based on domain
        completeness = self._check_completeness(result, profile)
        evaluation_details["checks"]["completeness"] = completeness
        score_components.append(completeness)
        
        # Check 4: SOTA volume/complexity standards
        sota_volume = self._check_sota_volume(result, profile)
        evaluation_details["checks"]["sota_volume"] = sota_volume
        score_components.append(sota_volume)
        
        # Check 5: Domain-specific quality metrics
        domain_quality = self._check_domain_quality(result, profile)
        evaluation_details["checks"]["domain_quality"] = domain_quality
        score_components.append(domain_quality)
        
        # Calculate overall score
        overall_score = sum(score_components) / len(score_components)
        evaluation_details["score"] = overall_score
        evaluation_details["passed"] = overall_score >= 0.6  # 60% threshold
        
        logger.info(f"SOTA Quality Gate evaluation: {overall_score:.2f} - {evaluation_details}")
        
        return overall_score, evaluation_details
    
    def _check_result_exists(self, result: AgentResult) -> bool:
        """Check that result exists and has content."""
        if not result:
            return False
        if hasattr(result, 'content') and result.content:
            return True
        if hasattr(result, 'actions') and result.actions:
            return True
        if hasattr(result, 'output') and result.output:
            return True
        return False
    
    def _check_actionable_content(self, result: AgentResult) -> bool:
        """Check that result has actionable content."""
        if hasattr(result, 'actions') and result.actions:
            return len(result.actions) >= self.min_action_count
        if hasattr(result, 'executed_actions') and result.executed_actions:
            return len(result.executed_actions) >= self.min_action_count
        # Check for URLs, phone numbers, or other actionable items in content
        if hasattr(result, 'content'):
            content = str(result.content)
            has_url = bool(re.search(r'https?://\S+', content))
            has_phone = bool(re.search(r'\+?\d{10,}', content))
            has_command = bool(re.search(r'(execute|run|call|open|visit)', content, re.IGNORECASE))
            return has_url or has_phone or has_command
        return False
    
    def _check_completeness(self, result: AgentResult, profile: DynamicDomainProfile) -> float:
        """Check result completeness based on domain context."""
        domain = getattr(profile, 'domain', 'unknown').lower()
        
        # Domain-specific completeness checks
        if domain == 'emergency':
            return self._check_emergency_completeness(result)
        elif domain == 'coding':
            return self._check_coding_completeness(result)
        elif domain == 'crypto_trading':
            return self._check_trading_completeness(result)
        elif domain == 'research':
            return self._check_research_completeness(result)
        else:
            # Generic completeness check
            return self._check_generic_completeness(result)
    
    def _check_emergency_completeness(self, result: AgentResult) -> float:
        """Emergency domain: needs location, immediate actions, resources."""
        score = 0.0
        content = str(getattr(result, 'content', ''))
        
        # Check for location
        has_location = bool(re.search(r'(location|zip|city|state|address)', content, re.IGNORECASE))
        if has_location:
            score += 0.3
        
        # Check for immediate actions
        has_immediate = bool(re.search(r'(immediate|now|today|urgent|emergency)', content, re.IGNORECASE))
        if has_immediate:
            score += 0.3
        
        # Check for resources/links
        has_resources = bool(re.search(r'https?://\S+', content))
        if has_resources:
            score += 0.4
        
        return score
    
    def _check_coding_completeness(self, result: AgentResult) -> float:
        """Coding domain: needs code, implementation, not just suggestions."""
        score = 0.0
        content = str(getattr(result, 'content', ''))
        
        # Check for code blocks
        has_code = bool(re.search(r'```[\w]*\n[\s\S]*?```', content))
        if has_code:
            score += 0.4
        
        # Check for implementation details
        has_impl = bool(re.search(r'(function|class|def|import|require)', content, re.IGNORECASE))
        if has_impl:
            score += 0.3
        
        # Check for file structure
        has_files = bool(re.search(r'(file|path|\.py|\.js|\.ts)', content, re.IGNORECASE))
        if has_files:
            score += 0.3
        
        return score
    
    def _check_trading_completeness(self, result: AgentResult) -> float:
        """Crypto trading domain: needs strategy, signals, risk management."""
        score = 0.0
        content = str(getattr(result, 'content', ''))
        
        # Check for strategy
        has_strategy = bool(re.search(r'(strategy|approach|method)', content, re.IGNORECASE))
        if has_strategy:
            score += 0.3
        
        # Check for signals/indicators
        has_signals = bool(re.search(r'(signal|indicator|rsi|macd|ema|sma)', content, re.IGNORECASE))
        if has_signals:
            score += 0.3
        
        # Check for risk management
        has_risk = bool(re.search(r'(risk|stop.*loss|position.*size|leverage)', content, re.IGNORECASE))
        if has_risk:
            score += 0.4
        
        return score
    
    def _check_research_completeness(self, result: AgentResult) -> float:
        """Research domain: needs sources, analysis, conclusions."""
        score = 0.0
        content = str(getattr(result, 'content', ''))
        
        # Check for sources
        has_sources = bool(re.search(r'(source|reference|citation|https?://)', content, re.IGNORECASE))
        if has_sources:
            score += 0.3
        
        # Check for analysis
        has_analysis = bool(re.search(r'(analysis|analyze|data|finding)', content, re.IGNORECASE))
        if has_analysis:
            score += 0.3
        
        # Check for conclusions
        has_conclusion = bool(re.search(r'(conclusion|summary|result|finding)', content, re.IGNORECASE))
        if has_conclusion:
            score += 0.4
        
        return score
    
    def _check_generic_completeness(self, result: AgentResult) -> float:
        """Generic completeness check for unknown domains."""
        content = str(getattr(result, 'content', ''))
        
        # Check for substantial content
        if len(content) < 50:
            return 0.0
        elif len(content) < 200:
            return 0.5
        else:
            return 1.0
    
    def _check_sota_volume(self, result: AgentResult, profile: DynamicDomainProfile) -> float:
        """Check if result meets SOTA volume/complexity standards."""
        content = str(getattr(result, 'content', ''))
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Count actions if available
        action_count = 0
        if hasattr(result, 'actions') and result.actions:
            action_count = len(result.actions)
        elif hasattr(result, 'executed_actions') and result.executed_actions:
            action_count = len(result.executed_actions)
        
        # SOTA volume check
        if len(non_empty_lines) >= self.sota_line_count and action_count >= self.sota_action_count:
            return 1.0  # Meets SOTA standards
        elif len(non_empty_lines) >= self.min_line_count and action_count >= self.min_action_count:
            return 0.7  # Meets minimum standards
        elif len(non_empty_lines) >= self.min_line_count or action_count >= self.min_action_count:
            return 0.4  # Partially meets standards
        else:
            return 0.0  # Below minimum standards
    
    def _check_domain_quality(self, result: AgentResult, profile: DynamicDomainProfile) -> float:
        """Check domain-specific quality metrics."""
        domain = getattr(profile, 'domain', 'unknown').lower()
        
        # Use domain-specific validators if available
        if hasattr(result, 'generated_files') and result.generated_files:
            # For code generation results
            is_zoho_like = any(word in str(getattr(result, 'goal', '')).lower() 
                             for word in ("zoho", "crm", "erp", "saas", "platform"))
            if is_zoho_like:
                passed, _, _ = QualityGate.verify_zoho_crm_application(result.generated_files)
                return 1.0 if passed else 0.5
            else:
                passed, _, _ = QualityGate.verify_generic_build(result.generated_files)
                return 1.0 if passed else 0.5
        
        # For other domains, infer quality from content patterns
        content = str(getattr(result, 'content', ''))
        
        # Check for quality indicators
        has_details = bool(re.search(r'(detail|specific|precise|exact)', content, re.IGNORECASE))
        has_structure = bool(re.search(r'(step|phase|stage|part)', content, re.IGNORECASE))
        has_verification = bool(re.search(r'(verify|check|test|validate)', content, re.IGNORECASE))
        
        quality_score = 0.0
        if has_details:
            quality_score += 0.3
        if has_structure:
            quality_score += 0.3
        if has_verification:
            quality_score += 0.4
        
        return quality_score
    
    async def ensure_code_quality(self, goal, profile, initial_code, ctx_cost_callback):
        """
        Ensure code quality across ANY domain using domain-agnostic validation.
        
        This method validates that generated code meets SOTA standards
        regardless of the domain (coding, emergency, trading, research, etc.)
        """
        # Use domain-agnostic validation
        if isinstance(initial_code, dict):
            # Check if it's a code generation result
            is_zoho_like = any(word in str(goal).lower() 
                             for word in ("zoho", "crm", "erp", "saas", "platform"))
            if is_zoho_like:
                passed, message, details = QualityGate.verify_zoho_crm_application(initial_code)
            else:
                passed, message, details = QualityGate.verify_generic_build(initial_code)
            
            validation_result = {
                "tests_passed": passed,
                "message": message,
                "details": details
            }
            
            cost = 0.0  # No additional cost for validation
            return initial_code, validation_result, cost
        else:
            # For non-code results, return as-is
            validation_result = {"tests_passed": True, "message": "Non-code result, validation skipped"}
            return initial_code, validation_result, 0.0
    
    def should_retry(self, quality_score, attempt):
        """
        Determine if execution should be retried based on quality score.
        
        Domain-agnostic retry logic:
        - Score < 0.4: Always retry (below minimum standards)
        - Score 0.4-0.6: Retry up to 2 times (partial quality)
        - Score > 0.6: No retry (meets standards)
        - Max 3 attempts total to prevent infinite loops
        """
        if attempt >= 3:
            return False  # Max attempts reached
        
        if quality_score < 0.4:
            return True  # Below minimum, retry
        elif quality_score < 0.6 and attempt < 2:
            return True  # Partial quality, retry once
        else:
            return False  # Good enough, no retry