# NLP Platform on Kubernetes


# 🚀 NLP API Gateway

A modern FastAPI application that orchestrates calls between TorchServe (NER) and Mistral (Chat) to create a comprehensive NLP platform.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🔧 Installation](#-installation)
- [🚀 Getting Started](#-getting-started)
- [📖 Usage](#-usage)
- [🔗 Endpoints](#-endpoints)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

## 🎯 Overview

This API Gateway combines the power of:
- **TorchServe** for Named Entity Recognition (NER)
- **Mistral/vLLM** for intelligent conversations
- **FastAPI** for a modern and performant interface

## ✨ Features

- 🔍 **NER (Named Entity Recognition)** - Extract entities from text
- 💬 **Intelligent Chat** - Conversations powered by Mistral
- 🌐 **CORS Configured** - Ready for web applications
- 📊 **Health Check** - API status monitoring
- 📖 **Automatic Documentation** - Built-in Swagger interface
- ⚡ **Optimized Performance** - Asynchronous architecture

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │────▶│  NLP Gateway    │────▶│   TorchServe    │
│                 │    │   (FastAPI)     │    │     (NER)       │
└─────────────────┘    │                 │    └─────────────────┘
                       │                 │    
                       │                 │    ┌─────────────────┐
                       │                 │────▶│  Mistral/vLLM   │
                       └─────────────────┘    │    (Chat)       │
                                              └─────────────────┘
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- FastAPI
- TorchServe configured
- Mistral/vLLM running

### Install Dependencies

```bash
# Clone the repository
git clone <your-repo>
cd nlp-platform

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file for your environment variables:

```env
# TorchServe
TORCHSERVE_URL=http://localhost:8080
TORCHSERVE_MODEL=ner_model

# Mistral/vLLM
MISTRAL_URL=http://localhost:8001
MISTRAL_MODEL=mistral-7b-instruct

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## 🚀 Getting Started

### Development

```bash
# Start the API in development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Start with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```bash
# Build the image
docker build -t nlp-gateway .

# Run the container
docker run -p 8000:8000 nlp-gateway
```

## 📖 Usage

### Quick Start

1. **Check if the API is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test the chat**:
   ```bash
   curl -X POST http://localhost:8000/nlp/chat \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, how are you?"}'
   ```

3. **Test NER**:
   ```bash
   curl -X POST http://localhost:8000/nlp/ner \
     -H "Content-Type: application/json" \
     -d '{"text": "Emmanuel Macron lives in Paris."}'
   ```

## 🔗 Endpoints

| Method | Endpoint | Description | Example |
|---------|----------|-------------|---------|
| `GET` | `/health` | Check API status | `200 {"status": "ok"}` |
| `POST` | `/nlp/chat` | Chat with Mistral | See below |
| `POST` | `/nlp/ner` | Named Entity Recognition | See below |

### 💬 Chat Endpoint

**Request:**
```json
{
  "prompt": "Explain artificial intelligence to me",
  "max_tokens": 150,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "Artificial intelligence is...",
  "tokens_used": 142,
  "model": "mistral-7b-instruct"
}
```

### 🔍 NER Endpoint

**Request:**
```json
{
  "text": "Apple Inc. is based in Cupertino, California.",
  "language": "en"
}
```

**Response:**
```json
{
  "entities": [
    {
      "text": "Apple Inc.",
      "label": "ORG",
      "start": 0,
      "end": 10,
      "confidence": 0.99
    },
    {
      "text": "Cupertino",
      "label": "LOC",
      "start": 23,
      "end": 32,
      "confidence": 0.95
    }
  ]
}
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Tests with coverage
pytest --cov=app tests/

# Integration tests
pytest tests/integration/
```

### Test Examples

```python
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_endpoint():
    response = client.post("/nlp/chat", json={"prompt": "Hello"})
    assert response.status_code == 200
    assert "response" in response.json()
```

## 📚 Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 🔨 Development

### Project Structure

TODO

### Code Standards

- **Formatting**: Black
- **Linting**: Flake8
- **Type hints**: mypy
- **Imports**: isort

```bash
# Format code
black app/ tests/

# Check linting
flake8 app/ tests/

# Check types
mypy app/
```

## 🐛 Troubleshooting

### Common Issues

1. **"Not Found" on `/chat`**:
   - Use `/nlp/chat` instead of `/chat`
   - Check that the router is properly included with the prefix

2. **TorchServe connection error**:
   - Verify TorchServe is running on the configured port
   - Test connection: `curl http://localhost:8080/ping`

3. **Mistral timeout**:
   - Increase timeout in configuration
   - Check that vLLM is started and accessible

### Logs

```bash
# Enable detailed logs
export LOG_LEVEL=DEBUG
uvicorn main:app --log-level debug
```

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines

- Add tests for new features
- Follow code standards
- Update documentation if necessary

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 👨‍💻 Author

LPMarcoB - [@marcopalomo](https://github.com/marcopalomo)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [TorchServe](https://pytorch.org/serve/) for PyTorch model serving
- [Mistral AI](https://mistral.ai/) for language models
- The open source community for all the tools used

---

⭐ Don't forget to star this project if it helped you!
