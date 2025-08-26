"""
Healthcare Voice (LLM) + A2A Triage + MCP Insurance
"""
import asyncio
import json
import os
import re
import base64
import uuid
import random
import string
import tempfile
from datetime import datetime
from typing import Dict

import requests
from flask import Flask, request, jsonify

# Audio imports with fallback
try:
    import speech_recognition as sr
    import pygame
    from gtts import gTTS
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Load environment
def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass

load_env()

# Session
class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.data = {}
        self.triage_complete = False
        self.triage_attempts = 0
        self.conversation_log = []
        self.start_time = datetime.now()
    
    def add_interaction(self, role, message, extra_data=None):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
            "session_data_snapshot": self.data.copy()
        }
        if extra_data:
            interaction["extra_data"] = extra_data
        self.conversation_log.append(interaction)
        print(f"SESSION-LOG: {role.upper()} - {message[:100]}...")
    
    def save_to_file(self):
        try:
            os.makedirs("sessions", exist_ok=True)
            filename = f"sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.id}.json"
            
            session_data = {
                "session_id": self.id,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "final_data": self.data,
                "triage_complete": self.triage_complete,
                "triage_attempts": self.triage_attempts,
                "conversation_log": self.conversation_log,
                "data_fields_collected": list(self.data.keys()),
                "total_interactions": len(self.conversation_log)
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            print(f"SESSION: Saved complete session to {filename}")
            return filename
        except Exception as e:
            print(f"SESSION: Save failed: {e}")
            return None

# Audio System
class AudioSystem:
    def __init__(self):
        self.enabled = AUDIO_AVAILABLE
        self.tts_enabled = False
        self.speech_enabled = False
        
        if self.enabled:
            try:
                print("Initializing audio...")
                
                try:
                    self.recognizer = sr.Recognizer()
                    self.microphone = sr.Microphone()
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                    self.recognizer.energy_threshold = 300
                    self.recognizer.dynamic_energy_threshold = True
                    self.recognizer.pause_threshold = 0.8
                    self.speech_enabled = True
                    print("Speech recognition ready")
                except Exception as e:
                    print(f"Speech recognition failed: {e}")
                    self.speech_enabled = False
                
                try:
                    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
                    pygame.mixer.init()
                    self.tts_enabled = True
                    print("TTS system ready")
                except Exception as e:
                    print(f"TTS init failed: {e}")
                    self.tts_enabled = False
                    
            except Exception as e:
                print(f"Audio init failed: {e}")
                self.enabled = False
    
    async def listen(self):
        if not self.speech_enabled:
            return input("You: ").strip()
        
        print("Listening...")
        
        def _listen():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=6)
                result = self.recognizer.recognize_google(audio, language='en-US')
                print(f"Recognized: '{result}'")
                return result.strip()
            except sr.UnknownValueError:
                return "UNCLEAR"
            except sr.WaitTimeoutError:
                return "TIMEOUT"
            except Exception:
                return "ERROR"
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _listen)
    
    async def speak(self, text):
        print(f"Agent: {text}")
        
        if not self.tts_enabled:
            return
        
        def _speak():
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    temp_file = tmp.name
                
                try:
                    tts.save(temp_file)
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    max_wait = 30
                    wait_count = 0
                    while pygame.mixer.music.get_busy() and wait_count < max_wait * 20:
                        pygame.time.wait(50)
                        wait_count += 1
                    
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                        
                finally:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                        
            except Exception as e:
                print(f"TTS error: {e}")
                self.tts_enabled = False
        
        if self.tts_enabled:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, _speak), 
                    timeout=35
                )
            except asyncio.TimeoutError:
                print("TTS timeout - disabling TTS")
                self.tts_enabled = False
            except Exception:
                self.tts_enabled = False

# A2A Message
class A2AMessage:
    def __init__(self, msg_type, agent_id, content, msg_id=None):
        self.id = msg_id or str(uuid.uuid4())
        self.type = msg_type
        self.agent_id = agent_id
        self.content = content
    
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "agent_id": self.agent_id,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["type"], data["agent_id"], data["content"], data["id"])

# A2A Triage Service
class A2ATriageService:
    def __init__(self):
        self.app = Flask(__name__)
        self.sessions = {}
        
        self.app_id = os.getenv('TRIAGE_APP_ID')
        self.app_key = os.getenv('TRIAGE_APP_KEY')
        self.instance_id = os.getenv('TRIAGE_INSTANCE_ID')
        self.token_url = os.getenv('TRIAGE_TOKEN_URL')
        self.base_url = os.getenv('TRIAGE_BASE_URL')
        
        @self.app.route('/a2a/message', methods=['POST'])
        def handle_message():
            try:
                message = A2AMessage.from_dict(request.json)
                print(f"A2A-SERVICE: Received {message.type}")
                
                if message.type == "triage_start":
                    return jsonify(self._start_triage(message).to_dict())
                elif message.type == "triage_message":
                    return jsonify(self._handle_message(message).to_dict())
                elif message.type == "triage_summary":
                    return jsonify(self._get_summary(message).to_dict())
                else:
                    return jsonify({"error": "Unknown type"}), 400
                    
            except Exception as e:
                print(f"A2A-SERVICE: Error: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _get_token(self):
        print("A2A-SERVICE: Getting token...")
        creds = base64.b64encode(f"{self.app_id}:{self.app_key}".encode()).decode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {creds}",
            "instance-id": self.instance_id
        }
        payload = {"grant_type": "client_credentials"}
        
        response = requests.post(self.token_url, headers=headers, json=payload, timeout=30)
        print(f"A2A-SERVICE: Token response: {response.status_code}")
        
        if response.status_code == 200:
            token = response.json()['access_token']
            print(f"A2A-SERVICE: Token received: {token[:20]}...")
            return token
        
        print(f"A2A-SERVICE: Token failed: {response.text}")
        raise Exception(f"Token failed: {response.status_code}")
    
    def _create_survey(self, token, age, sex):
        print(f"A2A-SERVICE: Creating survey - {age}yo {sex}")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"sex": sex.lower(), "age": {"value": age, "unit": "year"}}
        
        response = requests.post(f"{self.base_url}/surveys", headers=headers, json=payload, timeout=30)
        print(f"A2A-SERVICE: Survey response: {response.status_code}")
        
        if response.status_code == 200:
            survey_id = response.json()['survey_id']
            print(f"A2A-SERVICE: Survey created: {survey_id}")
            return survey_id
        
        print(f"A2A-SERVICE: Survey failed: {response.text}")
        raise Exception(f"Survey failed: {response.status_code}")
    
    def _send_message(self, token, survey_id, message):
        print(f"A2A-SERVICE: Sending message: '{message[:50]}...'")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"user_message": message}
        
        response = requests.post(f"{self.base_url}/surveys/{survey_id}/messages", headers=headers, json=payload, timeout=30)
        print(f"A2A-SERVICE: Message response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            result = {
                "success": True,
                "response": data.get('assistant_message', ''),
                "state": data.get('survey_state', 'active')
            }
            print(f"A2A-SERVICE: State: {result['state']}")
            return result
        
        print(f"A2A-SERVICE: Message failed: {response.text}")
        return {"success": False, "response": "Technical issue."}
    
    def _get_survey_summary(self, token, survey_id):
        print("A2A-SERVICE: Getting summary...")
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}/surveys/{survey_id}/summary", headers=headers, timeout=30)
        
        print(f"A2A-SERVICE: Summary response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"A2A-SERVICE: Summary data: {json.dumps(data, indent=2)}")
            
            urgency = "low"
            doctor = "general practitioner"
            
            for key in ['urgency', 'severity', 'priority']:
                if key in data:
                    val = str(data[key]).lower()
                    if val in ['high', 'urgent', 'emergency']:
                        urgency = "high"
                    elif val in ['medium', 'moderate']:
                        urgency = "medium"
                    break
            
            for key in ['doctor_type', 'specialist', 'recommendation']:
                if key in data:
                    doctor = str(data[key])
                    break
            
            result = {
                "success": True,
                "urgency_level": urgency,
                "doctor_type": doctor,
                "notes": str(data.get('notes', ''))
            }
            print(f"A2A-SERVICE: Final summary: {result}")
            return result
        
        print(f"A2A-SERVICE: Summary failed: {response.text}")
        return {"success": False}
    
    def _start_triage(self, message):
        content = message.content
        age = content.get("age", 30)
        sex = content.get("sex", "male")
        complaint = content.get("chief_complaint", "")
        
        try:
            token = self._get_token()
            survey_id = self._create_survey(token, age, sex)
            
            self.sessions[message.agent_id] = {
                "token": token,
                "survey_id": survey_id,
                "state": "active"
            }
            
            initial = self._send_message(token, survey_id, complaint)
            
            return A2AMessage("triage_response", "triage_service", {
                "success": True,
                "response": initial.get("response", ""),
                "state": initial.get("state", "active"),
                "survey_id": survey_id
            })
        except Exception as e:
            print(f"A2A-SERVICE: Start error: {e}")
            return A2AMessage("triage_response", "triage_service", {
                "success": False,
                "error": str(e)
            })
    
    def _handle_message(self, message):
        agent_id = message.agent_id
        user_message = message.content.get("message", "")
        
        if agent_id not in self.sessions:
            return A2AMessage("triage_response", "triage_service", {
                "success": False,
                "error": "No session"
            })
        
        session = self.sessions[agent_id]
        
        try:
            result = self._send_message(session["token"], session["survey_id"], user_message)
            session["state"] = result.get("state", "active")
            return A2AMessage("triage_response", "triage_service", result)
        except Exception as e:
            print(f"A2A-SERVICE: Message error: {e}")
            return A2AMessage("triage_response", "triage_service", {
                "success": False,
                "error": str(e)
            })
    
    def _get_summary(self, message):
        agent_id = message.agent_id
        
        if agent_id not in self.sessions:
            return A2AMessage("triage_summary_response", "triage_service", {
                "success": False,
                "error": "No session"
            })
        
        session = self.sessions[agent_id]
        
        try:
            summary = self._get_survey_summary(session["token"], session["survey_id"])
            del self.sessions[agent_id]
            return A2AMessage("triage_summary_response", "triage_service", summary)
        except Exception as e:
            print(f"A2A-SERVICE: Summary error: {e}")
            return A2AMessage("triage_summary_response", "triage_service", {
                "success": False,
                "error": str(e)
            })
    
    def run(self):
        print("A2A Triage Service starting on localhost:8887")
        self.app.run(host='localhost', port=8887, debug=False, use_reloader=False)

# A2A Client
class A2AClient:
    def __init__(self):
        self.base_url = "http://localhost:8887"
        self.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        print(f"A2A-CLIENT: Initialized")
    
    async def send_message(self, message):
        def _request():
            return requests.post(f"{self.base_url}/a2a/message", json=message.to_dict(), timeout=30)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _request)
        
        if response.status_code == 200:
            return A2AMessage.from_dict(response.json())
        raise Exception(f"A2A request failed: {response.status_code}")
    
    async def start_triage(self, age, sex, complaint):
        message = A2AMessage("triage_start", self.agent_id, {
            "age": age,
            "sex": sex,
            "chief_complaint": complaint
        })
        response = await self.send_message(message)
        return response.content
    
    async def send_triage_message(self, user_message):
        message = A2AMessage("triage_message", self.agent_id, {"message": user_message})
        response = await self.send_message(message)
        return response.content
    
    async def get_summary(self):
        message = A2AMessage("triage_summary", self.agent_id, {})
        response = await self.send_message(message)
        return response.content

# LLM Client
class LLMClient:
    def __init__(self, jwt_token, endpoint_url, project_id, connection_id):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jwt_token}'
        }
        self.endpoint_url = endpoint_url
        self.project_id = project_id
        self.connection_id = connection_id
        print("LLM: Initialized with JWT endpoint")
    
    async def process(self, user_input, session):
        print(f"LLM: Processing: '{user_input[:50]}...'")
        
        prompt = f"""You are a healthcare appointment scheduler.

Current session data: {json.dumps(session.data)}
User input: "{user_input}"

EXTRACTION RULES:
- For date of birth: Extract MM/DD/YYYY as "date_of_birth"
- For state: Extract US state as "state"
- For name: Extract full name as "name"
- For phone: Extract phone as "phone"
- For reason: Extract medical complaints as "reason"
- For provider: Extract doctor name as "provider_name"
- For date: Extract appointment date as "preferred_date"

Flow:
1. Get name, phone, reason
2. If reason is medical (pain, symptoms, illness) → set need_triage=true
3. After triage → get DOB, state → call discovery
4. Get provider → call eligibility → announce [Payer name, Policy ID, Co-pay details]
5. Schedule appointment - don't check availability, provide confirmation code, end call

JSON response:
{{
    "response": "what to say",
    "extract": {{"field": "value"}},
    "need_triage": true/false,
    "call_discovery": true/false,
    "call_eligibility": true/false,
    "done": true/false
}}"""
        
        payload = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            "project_id": self.project_id,
            "connection_id": self.connection_id,
            "max_tokens": 400,
            "temperature": 0.2
        }
        
        def _request():
            return requests.post(self.endpoint_url, headers=self.headers, json=payload, timeout=30)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _request)
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and data['choices']:
                content = data['choices'][0]['message']['content']
                
                try:
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    
                    result = json.loads(content.strip())
                    print("LLM: Response parsed")
                    return result
                except:
                    pass
        
        return {
            "response": "I understand. Please continue.",
            "extract": {},
            "need_triage": False,
            "call_discovery": False,
            "call_eligibility": False,
            "done": False
        }

# Insurance Client
class InsuranceClient:
    def __init__(self, mcp_url, api_key):
        self.mcp_url = mcp_url
        self.headers = {"Content-Type": "application/json", "X-INF-API-KEY": api_key}
        print("INSURANCE: Client initialized")
    
    def _split_name(self, name):
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0], ""
        elif len(parts) == 2:
            return parts[0], parts[1]
        else:
            return parts[0], " ".join(parts[1:])
    
    def _format_dob(self, dob):
        if not dob:
            return ""
        
        print(f"INSURANCE: Formatting DOB '{dob}'")
        
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', dob):
            month, day, year = dob.split('/')
            formatted = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            print(f"INSURANCE: Converted to '{formatted}'")
            return formatted
        
        if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', dob):
            return dob
        
        return dob
    
    async def discovery(self, name, dob, state):
        print(f"INSURANCE: Discovery - {name}, {dob}, {state}")
        first, last = self._split_name(name)
        formatted_dob = self._format_dob(dob)
        formatted_state = state.strip().title() if state else ""
        
        payload = {
            "jsonrpc": "2.0",
            "id": f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "method": "tools/call",
            "params": {
                "name": "insurance_discovery",
                "arguments": {
                    "patientDateOfBirth": formatted_dob,
                    "patientFirstName": first,
                    "patientLastName": last,
                    "patientState": formatted_state
                }
            }
        }
        
        print(f"INSURANCE: Discovery payload: {json.dumps(payload, indent=2)}")
        
        def _request():
            return requests.post(self.mcp_url, headers=self.headers, json=payload, timeout=45)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _request)
        
        print(f"INSURANCE: Discovery response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                result_text = str(data["result"])
                
                payer = ""
                member_id = ""
                
                for pattern in [r'payer[:\s]*([^\n,;]+)', r'insurance[:\s]*([^\n,;]+)']:
                    match = re.search(pattern, result_text.lower())
                    if match:
                        payer = match.group(1).strip().title()
                        break
                
                for pattern in [r'member\s*id[:\s]*([a-za-z0-9\-]+)', r'policy[:\s]*([a-za-z0-9\-]+)']:
                    match = re.search(pattern, result_text.lower())
                    if match:
                        member_id = match.group(1).strip().upper()
                        break
                
                print(f"INSURANCE: Found - Payer: {payer}, Member: {member_id}")
                return {"success": True, "payer": payer, "member_id": member_id}
        
        print("INSURANCE: Discovery failed")
        return {"success": False}
    
    async def eligibility(self, name, dob, subscriber_id, payer_name, provider_name):
        print(f"INSURANCE: Eligibility check")
        first, last = self._split_name(name)
        formatted_dob = self._format_dob(dob)
        
        provider_clean = re.sub(r'\b(Dr\.?|MD|DO)\b', '', provider_name, flags=re.IGNORECASE).strip()
        provider_first, provider_last = self._split_name(provider_clean)
        
        payload = {
            "jsonrpc": "2.0",
            "id": f"eligibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "method": "tools/call",
            "params": {
                "name": "benefits_eligibility",
                "arguments": {
                    "patientFirstName": first,
                    "patientLastName": last,
                    "patientDateOfBirth": formatted_dob,
                    "subscriberId": subscriber_id,
                    "payerName": payer_name,
                    "providerFirstName": provider_first,
                    "providerLastName": provider_last,
                    "providerNpi": "1234567890"
                }
            }
        }
        
        def _request():
            return requests.post(self.mcp_url, headers=self.headers, json=payload, timeout=45)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _request)
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                result_text = str(data["result"])
                
                copay = ""
                copay_match = re.search(r'co-?pay[:\s]*\$?([0-9,]+)', result_text.lower())
                if copay_match:
                    copay = copay_match.group(1)
                
                print(f"INSURANCE: Eligibility - Copay: ${copay}")
                return {"success": True, "copay": copay}
        
        return {"success": False}

# Main Agent
class HealthcareAgent:
    def __init__(self):
        self.session = Session()
        self.audio = AudioSystem()
        
        # JWT LLM
        jwt_token = os.getenv('JWT_TOKEN')
        endpoint_url = os.getenv('ENDPOINT_URL')
        project_id = os.getenv('PROJECT_ID')
        connection_id = os.getenv('CONNECTION_ID')
        
        if not all([jwt_token, endpoint_url, project_id, connection_id]):
            raise Exception("Missing JWT config")
            
        self.llm = LLMClient(jwt_token, endpoint_url, project_id, connection_id)
        
        # Insurance
        mcp_url = os.getenv('MCP_URL')
        insurance_key = os.getenv('X_INF_API_KEY')
        if not mcp_url or not insurance_key:
            raise Exception("Missing insurance config")
            
        self.insurance = InsuranceClient(mcp_url, insurance_key)
        
        # A2A client
        self.a2a_client = None
        try:
            self.a2a_client = A2AClient()
        except:
            print("A2A client not available")
    
    def _normalize_dob(self, dob_text):
        if not dob_text:
            return None
            
        cleaned = re.sub(r'\b(born|on|in|was|i|am|my|dob|is|birthday)\b', '', dob_text.lower()).strip()
        
        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', cleaned)
        if match:
            month, day, year = match.groups()
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
        
        match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', cleaned)
        if match:
            year, month, day = match.groups()
            return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
        
        return None
    
    def _normalize_state(self, state_text):
        if not state_text:
            return None
            
        state_map = {
            'ca': 'California', 'ny': 'New York', 'tx': 'Texas', 'fl': 'Florida',
            'il': 'Illinois', 'pa': 'Pennsylvania', 'oh': 'Ohio', 'ga': 'Georgia'
        }
        
        cleaned = state_text.lower().strip()
        
        if cleaned in state_map:
            return state_map[cleaned]
        
        if len(cleaned) > 2:
            return cleaned.title()
            
        return None
    
    def _extract_dob_from_text(self, text):
        patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'born.*?(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"
        return None
    
    def _extract_state_from_text(self, text):
        patterns = [
            r'\b(?:from|in|live in)\s+([a-zA-Z\s]+)',
            r'\b([A-Z]{2})\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, str) and match.strip():
                        normalized = self._normalize_state(match.strip())
                        if normalized:
                            return normalized
        return None
    
    async def start(self):
        print(f"Healthcare Agent starting - Session {self.session.id}")
        
        await self.audio.speak("Hello! I'm your healthcare appointment assistant. To get started, could you please tell me your full name?")
        self.session.add_interaction("assistant", "Hello! I'm your healthcare appointment assistant. To get started, could you please tell me your full name?")
        
        turn = 0
        errors = 0
        
        while turn < 50 and errors < 3:
            turn += 1
            print(f"--- Turn {turn} ---")
            
            user_input = await self.audio.listen()
            
            if user_input in ["UNCLEAR", "TIMEOUT", "ERROR"]:
                errors += 1
                await self.audio.speak("I didn't catch that. Could you please repeat?")
                continue
            
            if not user_input:
                continue
            
            errors = 0
            print(f"USER: {user_input}")
            self.session.add_interaction("user", user_input)
            
            if any(phrase in user_input.lower() for phrase in ['bye', 'goodbye', 'end', 'quit']):
                await self.audio.speak("Thank you for calling. Have a great day!")
                self.session.add_interaction("assistant", "Thank you for calling. Have a great day!")
                break
            
            # Process with LLM
            result = await self.llm.process(user_input, self.session)
            
            # Update session with better extraction
            if result.get("extract"):
                print(f"SESSION-UPDATE: Before - {self.session.data}")
                
                extractions = result["extract"]
                for key, value in extractions.items():
                    if key == "date_of_birth" and value:
                        normalized_dob = self._normalize_dob(value)
                        if normalized_dob:
                            self.session.data[key] = normalized_dob
                            print(f"SESSION-UPDATE: Normalized DOB '{value}' to '{normalized_dob}'")
                    elif key == "state" and value:
                        normalized_state = self._normalize_state(value)
                        if normalized_state:
                            self.session.data[key] = normalized_state
                            print(f"SESSION-UPDATE: Normalized state '{value}' to '{normalized_state}'")
                    elif value:
                        self.session.data[key] = value
                
                print(f"SESSION-UPDATE: After - {self.session.data}")
            
            # Additional extraction if missing
            if not self.session.data.get("date_of_birth"):
                dob_extracted = self._extract_dob_from_text(user_input)
                if dob_extracted:
                    self.session.data["date_of_birth"] = dob_extracted
                    print(f"SESSION-UPDATE: Additional DOB: '{dob_extracted}'")
            
            if not self.session.data.get("state"):
                state_extracted = self._extract_state_from_text(user_input)
                if state_extracted:
                    self.session.data["state"] = state_extracted
                    print(f"SESSION-UPDATE: Additional state: '{state_extracted}'")
            
            # Handle triage
            if (result.get("need_triage") and 
                not self.session.triage_complete and 
                self.session.triage_attempts < 1 and 
                self.a2a_client):
                print("TRIAGE: Starting session")
                await self._run_triage()
            
            # Handle discovery
            if result.get("call_discovery"):
                required = ['name', 'date_of_birth', 'state']
                print(f"INSURANCE-DISCOVERY: Required fields check")
                print(f"INSURANCE-DISCOVERY: Session data: {self.session.data}")
                
                if all(k in self.session.data and self.session.data[k] for k in required):
                    print("INSURANCE-DISCOVERY: Calling API...")
                    discovery = await self.insurance.discovery(
                        self.session.data['name'],
                        self.session.data['date_of_birth'],
                        self.session.data['state']
                    )
                    if discovery["success"]:
                        self.session.data['payer'] = discovery['payer']
                        self.session.data['member_id'] = discovery['member_id']
                        print(f"INSURANCE-DISCOVERY: Success - {discovery['payer']}, {discovery['member_id']}")
                else:
                    missing = [k for k in required if k not in self.session.data or not self.session.data[k]]
                    print(f"INSURANCE-DISCOVERY: Missing: {missing}")
            
            # Handle eligibility
            if result.get("call_eligibility"):
                required = ['name', 'date_of_birth', 'member_id', 'payer', 'provider_name']
                print(f"INSURANCE-ELIGIBILITY: Required fields check")
                print(f"INSURANCE-ELIGIBILITY: Session data: {self.session.data}")
                
                if all(k in self.session.data and self.session.data[k] for k in required):
                    print("INSURANCE-ELIGIBILITY: Calling API...")
                    eligibility = await self.insurance.eligibility(
                        self.session.data['name'],
                        self.session.data['date_of_birth'],
                        self.session.data['member_id'],
                        self.session.data['payer'],
                        self.session.data['provider_name']
                    )
                    if eligibility["success"] and eligibility['copay']:
                        copay_message = f"Great! I found your insurance. Your copay is ${eligibility['copay']}."
                        await self.audio.speak(copay_message)
                        self.session.add_interaction("assistant", copay_message)
                    else:
                        fallback_message = "I had trouble verifying your insurance, but we can proceed with scheduling."
                        await self.audio.speak(fallback_message)
                        self.session.add_interaction("assistant", fallback_message)
                else:
                    missing = [k for k in required if k not in self.session.data or not self.session.data[k]]
                    print(f"INSURANCE-ELIGIBILITY: Missing: {missing}")
            
            # Speak response
            response = result.get("response", "")
            if response:
                await self.audio.speak(response)
                self.session.add_interaction("assistant", response)
            
            # Check if done
            if result.get("done"):
                if self.session.data.get('name') and self.session.data.get('preferred_date'):
                    confirmation = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                    confirmation_message = f"Perfect! Your appointment is confirmed. Your confirmation number is {confirmation}. Thank you!"
                    await self.audio.speak(confirmation_message)
                    self.session.add_interaction("assistant", confirmation_message)
                break
        
        print(f"Conversation ended. Final data: {self.session.data}")
        
        # Save session
        saved_file = self.session.save_to_file()
        if saved_file:
            print(f"Session saved to: {saved_file}")
    
    async def _run_triage(self):
        """Run A2A triage session until complete"""
        self.session.triage_attempts += 1
        print(f"TRIAGE: Starting attempt {self.session.triage_attempts}")
        
        try:
            # Demographics
            age = 30
            sex = "male"
            
            if 'date_of_birth' in self.session.data:
                try:
                    dob = self.session.data['date_of_birth']
                    if '/' in dob:
                        parts = dob.split('/')
                        if len(parts) == 3:
                            birth_year = int(parts[2])
                            age = max(1, datetime.now().year - birth_year)
                            print(f"TRIAGE: Calculated age: {age}")
                except Exception as e:
                    print(f"TRIAGE: Age calculation failed: {e}")
            
            name = self.session.data.get('name', '').lower()
            if any(f in name for f in ['mary', 'sarah', 'jessica', 'jennifer', 'amanda']):
                sex = "female"
            
            print(f"TRIAGE: Demographics - Age: {age}, Sex: {sex}")
            
            triage_intro = "I need to ask some medical questions to assess your condition."
            await self.audio.speak(triage_intro)
            self.session.add_interaction("assistant", triage_intro)
            
            # Start triage
            complaint = self.session.data.get('reason', 'general concern')
            print(f"TRIAGE: Chief complaint: '{complaint}'")
            start_result = await self.a2a_client.start_triage(age, sex, complaint)
            
            if start_result.get("success") and start_result.get("response"):
                await self.audio.speak(start_result["response"])
                self.session.add_interaction("assistant", start_result["response"])
            
            # Triage conversation - unlimited turns until complete
            turn = 0
            print("TRIAGE: Starting conversation loop")
            
            while True:
                turn += 1
                print(f"TRIAGE: Turn {turn}")
                
                user_input = await self.audio.listen()
                
                if user_input in ["UNCLEAR", "TIMEOUT", "ERROR"]:
                    retry_message = "I didn't catch that. Please try again."
                    await self.audio.speak(retry_message)
                    continue
                
                print(f"TRIAGE USER: {user_input}")
                self.session.add_interaction("user", user_input)
                
                message_result = await self.a2a_client.send_triage_message(user_input)
                
                if message_result.get("success") and message_result.get("response"):
                    await self.audio.speak(message_result["response"])
                    self.session.add_interaction("assistant", message_result["response"])
                
                state = message_result.get("state", "").lower()
                print(f"TRIAGE: Current state: '{state}'")
                
                if state in ["completed", "finished", "done"]:
                    print(f"TRIAGE: Completed after turn {turn}")
                    break
                
                if turn >= 30:
                    print(f"TRIAGE: Safety limit reached ({turn} turns)")
                    break
            
            # Get summary
            print("TRIAGE: Getting summary...")
            summary = await self.a2a_client.get_summary()
            
            if summary.get("success"):
                self.session.triage_complete = True
                urgency = summary["urgency_level"]
                doctor = summary["doctor_type"]
                
                print(f"TRIAGE: Assessment - {urgency} priority, {doctor}")
                
                self.session.data['triage_urgency'] = urgency
                self.session.data['triage_doctor'] = doctor
                
                summary_message = f"Based on the assessment, your condition appears to be {urgency} priority. I recommend seeing a {doctor}. Now let me help schedule this appointment."
                await self.audio.speak(summary_message)
                self.session.add_interaction("assistant", summary_message)
            else:
                print("TRIAGE: Summary failed, marking complete")
                self.session.triage_complete = True
                
                fallback_message = "I've completed the medical assessment. Now let me help schedule your appointment."
                await self.audio.speak(fallback_message)
                self.session.add_interaction("assistant", fallback_message)
                
        except Exception as e:
            print(f"TRIAGE: Error: {e}")
            self.session.triage_complete = True
            
            error_message = "I'll help you schedule an appointment with a healthcare provider."
            await self.audio.speak(error_message)
            self.session.add_interaction("assistant", error_message)

def run_service():
    """Run A2A triage service"""
    print("=" * 50)
    print("A2A TRIAGE SERVICE")
    print("=" * 50)
    
    required = ['TRIAGE_APP_ID', 'TRIAGE_APP_KEY', 'TRIAGE_INSTANCE_ID', 'TRIAGE_TOKEN_URL', 'TRIAGE_BASE_URL']
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"ERROR: Missing config: {missing}")
        return
    
    service = A2ATriageService()
    service.run()

def run_agent():
    """Run healthcare agent"""
    print("=" * 50)
    print("HEALTHCARE VOICE AGENT")
    print("=" * 50)
    
    jwt_required = ['JWT_TOKEN', 'ENDPOINT_URL', 'PROJECT_ID', 'CONNECTION_ID']
    insurance_required = ['MCP_URL', 'X_INF_API_KEY']
    
    missing = []
    missing.extend([var for var in jwt_required if not os.getenv(var)])
    missing.extend([var for var in insurance_required if not os.getenv(var)])
    
    if missing:
        print(f"ERROR: Missing config: {missing}")
        return
    
    print("Configuration validated")
    if AUDIO_AVAILABLE:
        print("Audio system available")
    else:
        print("Console mode only")
    
    async def start():
        agent = HealthcareAgent()
        await agent.start()
    
    asyncio.run(start())

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "service":
        run_service()
    else:
        run_agent()

if __name__ == "__main__":
    main()
