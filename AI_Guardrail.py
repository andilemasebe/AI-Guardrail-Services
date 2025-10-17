import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import logging
from dataclasses import dataclass

# ==================== DATA MODELS ====================

class GuardrailResult:
    def __init__(self, is_safe: bool, risk_score: float = 0.0, 
                 reasons: List[str] = None, modified_output: str = None):
        self.is_safe = is_safe
        self.risk_score = risk_score  # 0.0 (safe) to 1.0 (dangerous)
        self.reasons = reasons or []
        self.modified_output = modified_output

class ActionType(Enum):
    WEB_SEARCH = "web_search"
    SEND_EMAIL = "send_email"
    DATABASE_QUERY = "database_query"
    FILE_WRITE = "file_write"
    API_CALL = "api_call"
    CODE_EXECUTION = "code_execution"

@dataclass
class AgentAction:
    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str = ""

# ==================== BASE GUARDRAIL CLASS ====================

class BaseGuardrail(ABC):
    """Abstract base class for all guardrails"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"guardrail.{name}")
    
    @abstractmethod
    def validate(self, input_data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        pass

# ==================== CONTENT SAFETY GUARDRAILS ====================

class ContentSafetyGuardrail(BaseGuardrail):
    """Validates content for safety, toxicity, and inappropriate material"""
    
    def __init__(self):
        super().__init__("content_safety")
        # In production, these would be ML models or API calls
        self.toxic_keywords = [
            "hate", "violence", "kill", "harm", "attack", "dangerous",
            "illegal", "weapon", "exploit", "hack", "cheat"
        ]
        
        self.sensitive_topics = [
            "financial advice", "medical diagnosis", "legal advice",
            "political manipulation", "self-harm"
        ]
    
    def validate(self, text: str, context: Dict[str, Any] = None) -> GuardrailResult:
        text_lower = text.lower()
        reasons = []
        risk_score = 0.0
        
        # Check for toxic keywords
        toxic_matches = [kw for kw in self.toxic_keywords if kw in text_lower]
        if toxic_matches:
            reasons.append(f"Found potentially toxic keywords: {toxic_matches}")
            risk_score += 0.3
        
        # Check for sensitive topics
        topic_matches = [topic for topic in self.sensitive_topics if topic in text_lower]
        if topic_matches:
            reasons.append(f"Attempted discussion of sensitive topics: {topic_matches}")
            risk_score += 0.4
        
        # Simple sentiment analysis (in production, use proper NLP)
        negative_indicators = ["hate", "terrible", "awful", "disgusting"]
        if any(indicator in text_lower for indicator in negative_indicators):
            risk_score += 0.2
        
        risk_score = min(risk_score, 1.0)
        is_safe = risk_score < 0.7  # Threshold config
        
        if not is_safe:
            reasons.append("Content safety threshold exceeded")
        
        return GuardrailResult(is_safe, risk_score, reasons)

class PIIGuardrail(BaseGuardrail):
    """Detects and redacts Personally Identifiable Information"""
    
    def __init__(self):
        super().__init__("pii_detection")
        # Regex patterns for common PII (simplified for example)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
        }
    
    def validate(self, text: str, context: Dict[str, Any] = None) -> GuardrailResult:
        detected_pii = {}
        modified_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_pii[pii_type] = matches
                # Redact the PII
                for match in matches:
                    modified_text = modified_text.replace(match, f"[{pii_type.upper()}_REDACTED]")
        
        reasons = []
        if detected_pii:
            reasons.append(f"Detected PII: {detected_pii}")
        
        is_safe = len(detected_pii) == 0  # Or configure threshold
        risk_score = min(len(detected_pii) * 0.2, 1.0)
        
        return GuardrailResult(
            is_safe=is_safe,
            risk_score=risk_score,
            reasons=reasons,
            modified_output=modified_text if detected_pii else None
        )

# ==================== AGENTIC ACTION GUARDRAILS ====================

class ActionGuardrail(BaseGuardrail):
    """Validates agent actions against allow/deny lists and policies"""
    
    def __init__(self):
        super().__init__("action_validation")
        self.allowed_actions = {
            ActionType.WEB_SEARCH: {
                'domains': ['wikipedia.org', 'news.gov', 'trusted-source.com'],
                'max_results': 10
            },
            ActionType.SEND_EMAIL: {
                'allowed_recipients': ['team@company.com', 'admin@company.com'],
                'require_approval': True
            },
            ActionType.DATABASE_QUERY: {
                'allowed_tables': ['products', 'users_public'],
                'read_only': True
            }
        }
        
        self.denied_actions = [ActionType.CODE_EXECUTION]  # Explicitly denied
    
    def validate(self, action: AgentAction, context: Dict[str, Any] = None) -> GuardrailResult:
        reasons = []
        risk_score = 0.0
        
        # Check deny list
        if action.action_type in self.denied_actions:
            return GuardrailResult(
                False, 1.0, 
                [f"Action {action.action_type.value} is explicitly denied"]
            )
        
        # Check if action is allowed
        if action.action_type not in self.allowed_actions:
            return GuardrailResult(
                False, 0.8,
                [f"Action {action.action_type.value} is not in allowed list"]
            )
        
        config = self.allowed_actions[action.action_type]
        
        # Action-specific validation
        if action.action_type == ActionType.SEND_EMAIL:
            recipient = action.parameters.get('to', '')
            if recipient not in config['allowed_recipients']:
                reasons.append(f"Email recipient not in allowed list: {recipient}")
                risk_score += 0.9
            
            if config['require_approval']:
                reasons.append("Email requires manual approval")
                risk_score += 0.5
        
        elif action.action_type == ActionType.WEB_SEARCH:
            query = action.parameters.get('query', '')
            # Check for sensitive search terms
            sensitive_terms = ['confidential', 'internal', 'password']
            if any(term in query.lower() for term in sensitive_terms):
                reasons.append("Search query contains sensitive terms")
                risk_score += 0.6
        
        elif action.action_type == ActionType.DATABASE_QUERY:
            table = action.parameters.get('table', '')
            if table not in config['allowed_tables']:
                reasons.append(f"Database table not allowed: {table}")
                risk_score += 0.8
            
            if not config['read_only'] and action.parameters.get('operation') == 'write':
                reasons.append("Write operations not allowed")
                risk_score += 0.9
        
        is_safe = risk_score < 0.7
        return GuardrailResult(is_safe, risk_score, reasons)

# ==================== CONTEXT & STATE GUARDRAILS ====================

class GoalAdherenceGuardrail(BaseGuardrail):
    """Ensures agent stays aligned with original user goal"""
    
    def __init__(self):
        super().__init__("goal_adherence")
        self.max_actions_per_goal = 20
        self.allowed_deviation_score = 0.3
    
    def validate(self, current_actions: List[AgentAction], 
                 original_goal: str, context: Dict[str, Any] = None) -> GuardrailResult:
        reasons = []
        risk_score = 0.0
        
        # Check for action limit
        if len(current_actions) > self.max_actions_per_goal:
            reasons.append(f"Exceeded maximum actions per goal: {self.max_actions_per_goal}")
            risk_score += 0.8
        
        # Simple goal drift detection (in production, use embeddings similarity)
        goal_keywords = set(original_goal.lower().split())
        recent_reasoning = " ".join([action.reasoning for action in current_actions[-3:]])
        recent_keywords = set(recent_reasoning.lower().split())
        
        # Calculate overlap between goal and recent actions
        overlap = len(goal_keywords.intersection(recent_keywords)) / len(goal_keywords) if goal_keywords else 1.0
        
        if overlap < self.allowed_deviation_score:
            reasons.append(f"Agent drifting from original goal. Overlap: {overlap:.2f}")
            risk_score += 0.6
        
        # Check for loops in reasoning
        if len(current_actions) > 5:
            last_actions = [action.action_type for action in current_actions[-5:]]
            if len(set(last_actions)) < 3:  # Too repetitive
                reasons.append("Agent may be stuck in a loop")
                risk_score += 0.4
        
        is_safe = risk_score < 0.7
        return GuardrailResult(is_safe, risk_score, reasons)

class RateLimitGuardrail(BaseGuardrail):
    """Enforces rate limiting on actions"""
    
    def __init__(self):
        super().__init__("rate_limiting")
        self.action_limits = {
            ActionType.WEB_SEARCH: {'max_per_minute': 5},
            ActionType.API_CALL: {'max_per_minute': 10},
            ActionType.DATABASE_QUERY: {'max_per_minute': 20}
        }
        self.action_history = []  # In production, use Redis or similar
    
    def validate(self, action: AgentAction, context: Dict[str, Any] = None) -> GuardrailResult:
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Clean old history
        self.action_history = [(act, t) for act, t in self.action_history if t > one_minute_ago]
        
        # Count recent actions of this type
        recent_count = sum(1 for act, t in self.action_history if act == action.action_type)
        
        if action.action_type in self.action_limits:
            limit = self.action_limits[action.action_type]['max_per_minute']
            
            if recent_count >= limit:
                return GuardrailResult(
                    False, 0.9,
                    [f"Rate limit exceeded for {action.action_type.value}: {recent_count}/{limit}"]
                )
        
        # Record this action
        self.action_history.append((action.action_type, current_time))
        
        return GuardrailResult(True, 0.0)

# ==================== MAIN GUARDRAILS SERVICE ====================

class AIGuardrailsService:
    """Orchestrates all guardrails and provides unified interface"""
    
    def __init__(self):
        self.logger = logging.getLogger("guardrails.service")
        
        # Initialize all guardrails
        self.content_safety = ContentSafetyGuardrail()
        self.pii_detection = PIIGuardrail()
        self.action_validation = ActionGuardrail()
        self.goal_adherence = GoalAdherenceGuardrail()
        self.rate_limiting = RateLimitGuardrail()
        
        self.input_guardrails = [self.content_safety, self.pii_detection]
        self.action_guardrails = [self.action_validation, self.rate_limiting]
        self.context_guardrails = [self.goal_adherence]
    
    def validate_input(self, user_input: str, context: Dict[str, Any] = None) -> GuardrailResult:
        """Validate user input before processing"""
        self.logger.info(f"Validating input: {user_input[:100]}...")
        
        overall_result = GuardrailResult(True, 0.0)
        
        for guardrail in self.input_guardrails:
            result = guardrail.validate(user_input, context)
            
            if not result.is_safe:
                overall_result.is_safe = False
                overall_result.risk_score = max(overall_result.risk_score, result.risk_score)
                overall_result.reasons.extend(result.reasons)
            
            # Apply modifications (like PII redaction)
            if result.modified_output:
                overall_result.modified_output = result.modified_output
        
        return overall_result
    
    def validate_action(self, action: AgentAction, context: Dict[str, Any] = None) -> GuardrailResult:
        """Validate agent action before execution"""
        self.logger.info(f"Validating action: {action.action_type.value}")
        
        overall_result = GuardrailResult(True, 0.0)
        
        for guardrail in self.action_guardrails:
            result = guardrail.validate(action, context)
            
            if not result.is_safe:
                overall_result.is_safe = False
                overall_result.risk_score = max(overall_result.risk_score, result.risk_score)
                overall_result.reasons.extend(result.reasons)
        
        return overall_result
    
    def validate_context(self, actions_history: List[AgentAction], 
                        original_goal: str, context: Dict[str, Any] = None) -> GuardrailResult:
        """Validate agent context and state"""
        self.logger.info(f"Validating context for goal: {original_goal}")
        
        overall_result = GuardrailResult(True, 0.0)
        
        for guardrail in self.context_guardrails:
            result = guardrail.validate(actions_history, original_goal, context)
            
            if not result.is_safe:
                overall_result.is_safe = False
                overall_result.risk_score = max(overall_result.risk_score, result.risk_score)
                overall_result.reasons.extend(result.reasons)
        
        return overall_result

# ==================== USAGE EXAMPLE ====================

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize guardrails service
    guardrails = AIGuardrailsService()
    
    # Example 1: Input validation
    print("=== INPUT VALIDATION ===")
    user_input = "Research our competitor and send summary to team@company.com. Also, my SSN is 123-45-6789."
    input_result = guardrails.validate_input(user_input)
    
    print(f"Input safe: {input_result.is_safe}")
    print(f"Risk score: {input_result.risk_score:.2f}")
    print(f"Reasons: {input_result.reasons}")
    if input_result.modified_output:
        print(f"Modified output: {input_result.modified_output}")
    
    # Example 2: Action validation
    print("\n=== ACTION VALIDATION ===")
    email_action = AgentAction(
        action_type=ActionType.SEND_EMAIL,
        parameters={"to": "team@company.com", "subject": "Competitor Research", "body": "Summary..."},
        reasoning="Sending research summary to team as requested"
    )
    
    action_result = guardrails.validate_action(email_action)
    print(f"Action allowed: {action_result.is_safe}")
    print(f"Risk score: {action_result.risk_score:.2f}")
    print(f"Reasons: {action_result.reasons}")
    
    # Example 3: Context validation
    print("\n=== CONTEXT VALIDATION ===")
    actions_history = [
        AgentAction(ActionType.WEB_SEARCH, {"query": "competitor ABC"}, "Researching competitor"),
        AgentAction(ActionType.WEB_SEARCH, {"query": "competitor news"}, "Getting latest news"),
        # ... more actions
    ]
    
    context_result = guardrails.validate_context(
        actions_history, 
        "Research competitor ABC and prepare summary"
    )
    
    print(f"Context valid: {context_result.is_safe}")
    print(f"Risk score: {context_result.risk_score:.2f}")
    print(f"Reasons: {context_result.reasons}")

if __name__ == "__main__":
    main()
