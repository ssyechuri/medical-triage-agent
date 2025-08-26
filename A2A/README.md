# Healthcare Voice Agent

A comprehensive healthcare appointment scheduling system that uses voice interaction, AI-powered medical triage, and automated insurance verification.

## Features

- **Voice-Powered Conversation**: Complete speech-to-text and text-to-speech interaction
- **AI Medical Triage**: Uses A2A protocol to connect with third-party triage APIs
- **Insurance Integration**: Automated discovery and eligibility checking via MCP APIs
- **LLM-Guided Flow**: JWT-based language model manages conversation logic
- **Session Management**: Complete conversation logging and data persistence
- **Robust Error Handling**: Comprehensive fallback systems for audio and API failures

## Architecture

### Components

1. **HealthcareAgent**: Main conversation orchestrator
2. **AudioSystem**: Speech recognition and text-to-speech with fallbacks
3. **A2ATriageService**: HTTP-to-A2A protocol wrapper for triage APIs
4. **A2AClient**: Client for communicating with triage service
5. **LLMClient**: JWT-based language model integration
6. **InsuranceClient**: MCP JSON-RPC insurance API integration

### Conversation Flow

1. **Initial Collection**: Name, phone number, reason for visit
2. **Medical Triage**: A2A-powered symptom assessment (if medical concern)
3. **Insurance Discovery**: Automated payer and member ID lookup
4. **Benefits Verification**: Copay and coverage confirmation
5. **Appointment Scheduling**: Date/time collection and confirmation

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Microphone and speakers (for voice interaction)
- Internet connection
- Audio system permissions

### External Services

- **JWT LLM Endpoint**: For conversation management
- **MCP Insurance Server**: For discovery and eligibility APIs
- **Third-party Triage API**: For medical assessment (optional)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ssyechuri/medical-triage-agent/A2A
cd healthcare-voice-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# LLM Configuration (JWT-based)
JWT_TOKEN=your_jwt_bearer_token
ENDPOINT_URL=your_llm_endpoint_url
PROJECT_ID=your_project_id
CONNECTION_ID=your_connection_id

# Insurance API Configuration
MCP_URL=https://your-mcp-server.com/jsonrpc
X_INF_API_KEY=your_insurance_api_key

# Triage API Configuration (Optional)
TRIAGE_APP_ID=your_triage_app_id
TRIAGE_APP_KEY=your_triage_app_secret
TRIAGE_INSTANCE_ID=your_triage_instance_id
TRIAGE_TOKEN_URL=https://auth.triage-provider.com/oauth/token
TRIAGE_BASE_URL=https://api.triage-provider.com/v1
```

### 5. Audio System Setup

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

#### macOS
```bash
brew install portaudio
```

#### Windows
Audio dependencies should install automatically with pip.

## Usage

### Running the System

The system requires two processes:

#### Terminal 1: Start A2A Triage Service
```bash
python healthcare_agent.py service
```

#### Terminal 2: Start Healthcare Agent
```bash
python healthcare_agent.py
```

### Testing Without Triage

If triage APIs are not configured, the system will skip medical assessment but continue with insurance and scheduling.

### Voice Interaction

1. Speak clearly into your microphone
2. Wait for the agent to finish speaking before responding
3. If the system doesn't understand, it will ask you to repeat
4. Say "goodbye" or "end call" to terminate the session

## Configuration Details

### LLM Configuration

The system uses JWT-based authentication for the language model:

- `JWT_TOKEN`: Bearer token for API authentication
- `ENDPOINT_URL`: LLM service endpoint
- `PROJECT_ID`: Project identifier
- `CONNECTION_ID`: Connection identifier

### Insurance APIs

MCP (Model Context Protocol) server integration:

- `MCP_URL`: JSON-RPC endpoint URL
- `X_INF_API_KEY`: API key for authentication

Expected API methods:
- `insurance_discovery`: Lookup payer and member ID
- `benefits_eligibility`: Check copay and coverage

### Triage Integration

A2A protocol wrapper for third-party triage services:

- `TRIAGE_APP_ID`: Application ID
- `TRIAGE_APP_KEY`: Application secret key  
- `TRIAGE_INSTANCE_ID`: Instance identifier
- `TRIAGE_TOKEN_URL`: OAuth token endpoint
- `TRIAGE_BASE_URL`: API base URL

## API Specifications

### Insurance Discovery Request

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "insurance_discovery",
    "arguments": {
      "patientDateOfBirth": "1985-03-15",
      "patientFirstName": "John", 
      "patientLastName": "Doe",
      "patientState": "California"
    }
  }
}
```

### Insurance Eligibility Request

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call", 
  "params": {
    "name": "benefits_eligibility",
    "arguments": {
      "patientFirstName": "John",
      "patientLastName": "Doe", 
      "patientDateOfBirth": "1985-03-15",
      "subscriberId": "MEMBER123",
      "payerName": "Blue Cross",
      "providerFirstName": "Jane",
      "providerLastName": "Smith",
      "providerNpi": "1234567890"
    }
  }
}
```

## Session Data

Sessions are automatically saved to `sessions/` directory with:

- Complete conversation transcript
- Extracted patient information
- Triage results and recommendations
- Insurance verification details
- Appointment confirmation data

## Troubleshooting

### Audio Issues

**Problem**: Speech recognition not working
**Solution**: Check microphone permissions and audio drivers

**Problem**: TTS playback fails
**Solution**: Verify speaker settings and pygame installation

### API Connection Issues

**Problem**: Insurance discovery fails
**Solution**: Verify MCP server URL and API key

**Problem**: Triage session won't start  
**Solution**: Check triage API credentials and network connectivity

**Problem**: LLM requests fail
**Solution**: Validate JWT token and endpoint URL

### Common Error Messages

- `"Audio libraries not available"`: Install audio dependencies
- `"Missing JWT config"`: Set all JWT environment variables
- `"A2A client not available"`: Verify triage service is running
- `"TTS timeout"`: Audio system overloaded, will continue text-only

## Development

### Adding New Features

1. **New API Integration**: Extend `InsuranceClient` or create new client class
2. **Enhanced Triage**: Modify `A2ATriageService` message handling
3. **Conversation Logic**: Update LLM prompts in `LLMClient.process()`
4. **Audio Improvements**: Enhance `AudioSystem` reliability

### Testing

```bash
# Test individual components
python -c "from healthcare_agent import AudioSystem; AudioSystem()"
python -c "from healthcare_agent import LLMClient; print('LLM import successful')"

# Test A2A service
curl -X POST http://localhost:8887/a2a/message \
  -H "Content-Type: application/json" \
  -d '{"id":"test","type":"triage_start","agent_id":"test","content":{"age":30,"sex":"male","chief_complaint":"headache"}}'
```

### Logging

All operations are logged with prefixes:
- `AudioSystem:` - Speech and TTS operations
- `A2A-SERVICE:` - Triage API communications
- `INSURANCE:` - Discovery and eligibility calls
- `HealthcareAgent:` - Main conversation flow
- `SESSION:` - Data extraction and storage

## Security Considerations

- API keys are loaded from environment variables
- Session data may contain PHI - ensure HIPAA compliance
- Audio processing happens locally
- Network communications use HTTPS/WSS where possible

## License

Apache 2.0 LICENSE

## Support

For technical issues:
1. Check the troubleshooting section
2. Review console logs for error details
3. Verify all configuration parameters
4. Test individual components separately

## Legal disclaimer
This is for demonstration, learning purposes. Not for medical use.
