"""General automation tools for OMEGA — Browser automation, Twilio, Country detection, CAPTCHA solving.

These tools provide domain-agnostic automation capabilities that can be used
across ALL domains (emergency, coding, research, crypto_trading, planning, etc.).

Capabilities:
- Browser automation using Playwright (navigate, fill forms, click elements)
- Phone call execution using Twilio API with voice capabilities (STT/TTS)
- Country detection from zip codes and location strings
- CAPTCHA solving using multimodal LLMs (OpenRouter) with Tesseract OCR fallback
- Action execution framework for making OMEGA an ACTOR, not a suggestor

Install requirements:
    playwright>=1.44.0
    twilio>=8.0.0
    pytesseract>=0.3.10
    Pillow>=10.0.0
    httpx>=0.24.0
    base64
    openai>=1.0.0  # For speech-to-text and text-to-speech
    speechrecognition>=3.10.0  # Alternative STT

After pip install, run once:
    playwright install chromium
"""

import asyncio
import base64
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from omega_agent.tools.failure_handler import get_failure_handler

logger = logging.getLogger("omega_agent.tools.automation")


# ── Country Detection ─────────────────────────────────────────────────────────

def detect_country_from_location(location: str) -> str:
    """
    Detect country from location string or zip code.
    
    Returns ISO country code (e.g., "IN" for India, "US" for United States).
    Defaults to "US" if unable to detect.
    
    Patterns:
    - 6-digit numeric = India (e.g., 788005)
    - 5-digit numeric = United States (default assumption)
    - City names with country suffixes = extract country
    """
    if not location:
        return "US"
    
    loc = location.strip()
    
    # Check for numeric zip codes
    if loc.isdigit():
        if len(loc) == 6:
            # 6-digit ZIP code pattern (e.g., 788005 = India)
            return "IN"
        elif len(loc) == 5:
            # 5-digit ZIP code (default to US, could be other countries)
            return "US"
    
    # Check for country name in location string
    loc_lower = loc.lower()
    if any(country in loc_lower for country in ["india", "indian", "assam", "delhi", "mumbai"]):
        return "IN"
    elif any(country in loc_lower for country in ["usa", "united states", "america"]):
        return "US"
    elif any(country in loc_lower for country in ["uk", "united kingdom", "britain"]):
        return "GB"
    elif any(country in loc_lower for country in ["canada", "canadian"]):
        return "CA"
    elif any(country in loc_lower for country in ["australia", "australian"]):
        return "AU"
    
    # Default to US
    return "US"


# ── Twilio Integration with Voice Capabilities ─────────────────────────────────

async def text_to_speech(text: str, voice: str = "alloy", output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert text to speech using OpenAI's TTS API.
    
    Requires OPENAI_API_KEY environment variable.
    
    Args:
        text: The text to convert to speech
        voice: Voice model to use (alloy, echo, fable, onyx, nova, shimmer)
        output_path: Optional path to save the audio file
    
    Returns:
        Dict with success status, audio data or file path, and action taken
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OPENAI_API_KEY not configured",
            "action_taken": "Text-to-speech not executed - missing API key"
        }
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        # Generate speech
        response = await client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Save to file or return bytes
        if output_path:
            response.stream_to_file(output_path)
            return {
                "success": True,
                "audio_path": output_path,
                "text": text,
                "voice": voice,
                "action_taken": f"Converted text to speech and saved to {output_path}"
            }
        else:
            audio_data = response.content
            return {
                "success": True,
                "audio_data": audio_data,
                "text": text,
                "voice": voice,
                "action_taken": "Converted text to speech (audio data in memory)"
            }
    except ImportError:
        return {
            "success": False,
            "error": "OpenAI library not installed. Install with: pip install openai",
            "action_taken": "Text-to-speech not executed - library missing"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action_taken": f"Text-to-speech failed: {str(e)}"
        }


async def speech_to_text(audio_file_path: str, language: str = "en") -> Dict[str, Any]:
    """
    Convert speech to text using OpenAI's Whisper API.
    
    Requires OPENAI_API_KEY environment variable.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        language: Language code (e.g., 'en' for English)
    
    Returns:
        Dict with success status, transcribed text, and action taken
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OPENAI_API_KEY not configured",
            "action_taken": "Speech-to-text not executed - missing API key"
        }
    
    if not os.path.exists(audio_file_path):
        return {
            "success": False,
            "error": f"Audio file not found: {audio_file_path}",
            "action_taken": "Speech-to-text not executed - file missing"
        }
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        # Transcribe audio
        with open(audio_file_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
        
        return {
            "success": True,
            "transcribed_text": response.text,
            "language": language,
            "audio_file": audio_file_path,
            "action_taken": f"Transcribed audio from {audio_file_path}"
        }
    except ImportError:
        return {
            "success": False,
            "error": "OpenAI library not installed. Install with: pip install openai",
            "action_taken": "Speech-to-text not executed - library missing"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action_taken": f"Speech-to-text failed: {str(e)}"
        }


async def make_voice_call(
    phone_number: str,
    initial_message: str = "",
    max_turns: int = 3,
    conversation_handler: Optional[callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Make an interactive voice call using Twilio with speech-to-text and text-to-speech.
    This enables OMEGA to have natural language conversations with people.
    
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, and OPENAI_API_KEY.
    
    Args:
        phone_number: The phone number to call (with country code, e.g., +1234567890)
        initial_message: Optional initial message to speak when call connects
        max_turns: Maximum number of conversation turns (back-and-forth exchanges)
        conversation_handler: Optional async function that takes user text and returns OMEGA's response
                             If not provided, uses a simple echo handler
    
    Returns:
        Dict with success status, call SID, conversation transcript, and action taken
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not all([account_sid, auth_token, from_number, openai_key]):
        return {
            "success": False,
            "error": "Missing credentials. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, and OPENAI_API_KEY environment variables.",
            "action_taken": f"Voice call to {phone_number} not executed - missing credentials"
        }
    
    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse, Gather
        
        client = Client(account_sid, auth_token)
        
        # Create a TwiML URL that handles the conversation
        # For simplicity, we'll use Twilio's built-in <Gather> for speech input
        # and <Say> for text-to-speech output
        
        twiml = VoiceResponse()
        
        if initial_message:
            twiml.say(initial_message, voice="alice")
        
        # Set up speech gathering
        gather = Gather(
            input="speech",
            action="/twilio/gather",
            speech_model="phone_call",
            speech_timeout="auto",
            language="en-US"
        )
        gather.say("Please speak your response after the beep.")
        twiml.append(gather)
        
        # If no speech detected, hang up
        twiml.say("No speech detected. Goodbye.")
        twiml.hangup()
        
        # For a full implementation, you'd need a web server to handle the TwiML callbacks
        # This is a simplified version that initiates the call
        call = client.calls.create(
            to=phone_number,
            from_=from_number,
            twiml=str(twiml),
            method="POST"
        )
        
        conversation_transcript = [{
            "turn": 0,
            "speaker": "OMEGA",
            "text": initial_message,
            "timestamp": time.time()
        }]
        
        return {
            "success": True,
            "call_sid": call.sid,
            "phone_number": phone_number,
            "conversation_transcript": conversation_transcript,
            "max_turns": max_turns,
            "action_taken": f"Initiated interactive voice call to {phone_number} (SID: {call.sid})",
            "status": call.status,
            "note": "Full conversation handling requires a web server for TwiML callbacks. This is a simplified implementation."
        }
    except ImportError:
        return {
            "success": False,
            "error": "Twilio or OpenAI library not installed. Install with: pip install twilio openai",
            "action_taken": f"Voice call to {phone_number} not executed - library missing"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action_taken": f"Voice call to {phone_number} failed: {str(e)}"
        }


async def make_phone_call(phone_number: str, message: str = "", **kwargs) -> Dict[str, Any]:
    """
    Make a simple phone call using Twilio API (legacy, non-interactive).
    
    Requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables.
    Also requires TWILIO_PHONE_NUMBER (the Twilio number to call from).
    
    Args:
        phone_number: The phone number to call (with country code, e.g., +1234567890)
        message: Optional message to speak (uses TwiML)
    
    Returns:
        Dict with success status, call SID, and action taken
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER")
    
    if not all([account_sid, auth_token, from_number]):
        return {
            "success": False,
            "error": "Twilio credentials not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables.",
            "action_taken": f"Phone call to {phone_number} not executed - missing credentials"
        }
    
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        
        # Make the call
        call = client.calls.create(
            to=phone_number,
            from_=from_number,
            url=f"http://demo.twilio.com/docs/voice.xml?msg={message}" if message else "http://demo.twilio.com/docs/voice.xml"
        )
        
        return {
            "success": True,
            "call_sid": call.sid,
            "phone_number": phone_number,
            "action_taken": f"Initiated phone call to {phone_number} (SID: {call.sid})",
            "status": call.status
        }
    except ImportError:
        return {
            "success": False,
            "error": "Twilio library not installed. Install with: pip install twilio",
            "action_taken": f"Phone call to {phone_number} not executed - library missing"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action_taken": f"Phone call to {phone_number} failed: {str(e)}"
        }


# ── Browser Automation ───────────────────────────────────────────────────────

async def execute_browser_action(url: str, action: str = "navigate", **kwargs) -> Dict[str, Any]:
    """
    Execute browser automation action using Playwright.
    
    Args:
        url: The URL to navigate to
        action: Type of action ("navigate", "fill_form", "click", etc.)
        **kwargs: Additional action-specific parameters
    
    Returns:
        Dict with success status, page title, and action taken
    """
    try:
        from omega_agent.tools.browser import browser_navigate
        result = await browser_navigate(url=url, wait_for="domcontentloaded", timeout=20)
        
        return {
            "success": result.get("success", False),
            "url": url,
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "action_taken": f"Opened {url} in browser and extracted content",
            "method": "browser_automation"
        }
    except ImportError:
        return {
            "success": False,
            "error": "Browser automation not available. Install with: pip install playwright && playwright install chromium",
            "action_taken": f"Browser action for {url} not executed - library missing",
            "method": "unavailable"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action_taken": f"Browser action for {url} failed: {str(e)}",
            "method": "error"
        }


# ── Action Execution Framework ─────────────────────────────────────────────────

async def execute_priority_actions(
    immediate_actions: List[Dict[str, Any]],
    location: str = ""
) -> List[Dict[str, Any]]:
    """
    ACTUALLY EXECUTE priority actions using browser automation and available tools.
    This makes OMEGA an ACTOR, not a suggestor, across ALL domains.
    
    GRACEFUL FAILURE HANDLING: If any tool fails, the workflow continues with
    fallback strategies (e.g., link rendering instead of browser automation).
    
    Args:
        immediate_actions: List of action dicts with 'action', 'url', 'phone', etc.
        location: Optional location string for context
    
    Returns:
        List of executed actions with execution metadata
    """
    executed = []
    failure_handler = get_failure_handler()
    
    for action in immediate_actions[:4]:
        url = action.get("url", "")
        action_type = action.get("action", "")
        
        if url and action_type in ("open_url", "visit_or_call"):
            # Actually navigate and interact with the URL using browser automation
            try:
                result = await execute_browser_action(url, action="navigate")
                executed.append({
                    **action,
                    "execution": {
                        "opened": result.get("success", False),
                        "method": result.get("method", "browser_automation"),
                        "navigated": result.get("success", False),
                        "page_title": result.get("title", ""),
                        "action_taken": result.get("action_taken", "")
                    }
                })
            except Exception as e:
                logger.warning("Browser automation failed for %s: %s", url, e)
                # Use failure handler to determine fallback
                fallback = failure_handler._browser_automation_fallback(e, {"url": url})
                executed.append({
                    **action,
                    "execution": {
                        "opened": True,
                        "method": fallback["strategy"],
                        "action_taken": fallback["action_taken"],
                        "fallback_triggered": True
                    }
                })
                
        elif action.get("phone"):
            # Actually make the phone call using Twilio
            phone = action.get("phone", "")
            try:
                call_result = await make_phone_call(phone)
                if call_result.get("success"):
                    executed.append({
                        **action,
                        "execution": {
                            "dialed": True,
                            "method": "twilio_api",
                            "call_sid": call_result.get("call_sid"),
                            "status": call_result.get("status"),
                            "action_taken": call_result.get("action_taken")
                        }
                    })
                else:
                    # Twilio failed or not configured, mark as user action required
                    fallback = failure_handler._twilio_fallback(Exception(call_result.get("error", "Unknown")), {"phone_number": phone})
                    executed.append({
                        **action,
                        "execution": {
                            "dialed": False,
                            "method": fallback["strategy"],
                            "reason": call_result.get("error", "Unknown error"),
                            "action_taken": fallback["action_taken"],
                            "fallback_triggered": True
                        }
                    })
            except Exception as e:
                logger.warning("Phone call failed for %s: %s", phone, e)
                fallback = failure_handler._twilio_fallback(e, {"phone_number": phone})
                executed.append({
                    **action,
                    "execution": {
                        "dialed": False,
                        "method": fallback["strategy"],
                        "reason": str(e),
                        "action_taken": fallback["action_taken"],
                        "fallback_triggered": True
                    }
                })
        else:
            # Fallback for other action types
            executed.append({
                **action,
                "execution": {
                    "opened": True,
                    "method": "link_rendered",
                    "action_taken": f"Action prepared: {action_type}"
                }
            })
    
    return executed


# ── CAPTCHA Solving ────────────────────────────────────────────────────────────

async def solve_captcha(
    image_path: str,
    use_llm: bool = True,
    fallback_to_ocr: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve CAPTCHA using multimodal LLM with Tesseract OCR fallback.
    
    Args:
        image_path: Path to CAPTCHA image file
        use_llm: Whether to try multimodal LLM first (default: True)
        fallback_to_ocr: Whether to fall back to Tesseract OCR if LLM fails (default: True)
    
    Returns:
        Dict with solved text, method used, and success status
    """
    if not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image file not found: {image_path}",
            "solved_text": "",
            "method": "none"
        }
    
    # Try multimodal LLM first
    if use_llm:
        try:
            llm_result = await _solve_captcha_with_llm(image_path)
            if llm_result.get("success") and llm_result.get("solved_text"):
                return llm_result
        except Exception as e:
            logger.warning("LLM CAPTCHA solving failed: %s", e)
    
    # Fallback to Tesseract OCR
    if fallback_to_ocr:
        try:
            ocr_result = _solve_captcha_with_ocr(image_path)
            if ocr_result.get("success") and ocr_result.get("solved_text"):
                return ocr_result
        except Exception as e:
            logger.warning("OCR CAPTCHA solving failed: %s", e)
    
    return {
        "success": False,
        "error": "Both LLM and OCR CAPTCHA solving failed",
        "solved_text": "",
        "method": "none"
    }


async def _solve_captcha_with_llm(image_path: str) -> Dict[str, Any]:
    """
    Solve CAPTCHA using free multimodal LLM from OpenRouter.
    
    Uses models like llava-hf/llava-1.5-7b or similar free vision models.
    """
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        return {
            "success": False,
            "error": "OPENROUTER_API_KEY not configured",
            "solved_text": "",
            "method": "llm"
        }
    
    # Read and encode image
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read image: {str(e)}",
            "solved_text": "",
            "method": "llm"
        }
    
    # Try multiple free multimodal models
    models_to_try = [
        "llava-hf/llava-1.5-7b",
        "openai/gpt-4o-mini",  # If available
        "google/gemma-3-4b-it:free",
    ]
    
    for model in models_to_try:
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Solve this CAPTCHA. Return ONLY the text shown in the image, nothing else. No explanations, no quotes, just the exact text."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 50,
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    solved_text = result["choices"][0]["message"]["content"].strip()
                    
                    # Clean up common LLM artifacts
                    solved_text = solved_text.replace('"', '').replace("'", "").strip()
                    
                    if solved_text:
                        return {
                            "success": True,
                            "solved_text": solved_text,
                            "method": f"llm_{model}",
                            "model": model
                        }
        except Exception as e:
            logger.warning("Failed to use model %s: %s", model, e)
            continue
    
    return {
        "success": False,
        "error": "All LLM models failed",
        "solved_text": "",
        "method": "llm"
    }


def _solve_captcha_with_ocr(image_path: str) -> Dict[str, Any]:
    """
    Solve CAPTCHA using Tesseract OCR as fallback.
    """
    try:
        import pytesseract
        from PIL import Image
        
        img = Image.open(image_path)
        # Convert to grayscale for better OCR accuracy
        img = img.convert('L')
        
        # Configure Tesseract for CAPTCHA
        config = '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        solved_text = pytesseract.image_to_string(img, config=config).strip()
        
        # Clean up OCR output
        solved_text = solved_text.replace('\n', '').replace(' ', '').strip()
        
        if solved_text:
            return {
                "success": True,
                "solved_text": solved_text,
                "method": "tesseract_ocr"
            }
    except ImportError:
        return {
            "success": False,
            "error": "Tesseract not installed. Install with: pip install pytesseract && apt-get install tesseract-ocr",
            "solved_text": "",
            "method": "tesseract_ocr"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Tesseract OCR failed: {str(e)}",
            "solved_text": "",
            "method": "tesseract_ocr"
        }
    
    return {
        "success": False,
        "error": "OCR produced no output",
        "solved_text": "",
        "method": "tesseract_ocr"
    }
