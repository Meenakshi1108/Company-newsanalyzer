
---

# Company News Analyzer

This project is a **Company News Analyzer** that extracts, summarizes, and analyzes news articles about a specified company. It provides sentiment analysis, topic extraction, and a Hindi audio summary of the news. The backend is built with **FastAPI**, and the frontend is a **Streamlit** web application.

---

## Table of Contents

- [Project Setup](#project-setup)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (Streamlit)](#frontend-streamlit)
- [Model Details](#model-details)
  - [Summarization](#summarization)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Text-to-Speech (TTS)](#text-to-speech-tts)
- [API Development](#api-development)
- [API Usage](#api-usage)
  - [Third-Party APIs](#third-party-apis)
- [Assumptions & Limitations](#assumptions--limitations)
  - [Assumptions](#assumptions)
  - [Limitations](#limitations)

---

## Project Setup

### Backend (FastAPI)

1. **Install dependencies**:
   - Ensure you have Python 3.8+ installed.
   - Install the required packages using the provided `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the API server**:
   - Start the FastAPI server using Uvicorn:
     ```bash
     uvicorn api:app --host 0.0.0.0 --port 8000 --reload
     ```
   - The API will be available at `http://localhost:8000`.

### Frontend (Streamlit)

1. **Install Streamlit** (if not already installed via `requirements.txt`):
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit app**:
   - Start the Streamlit application:
     ```bash
     streamlit run app.py
     ```
   - The app will typically be available at `http://localhost:8501`.

---

## Model Details

### Summarization

- **Library Used**: The project employs the **`newspaper3k`** library for article extraction and summarization.
- **Functionality**: `newspaper3k` downloads articles from URLs, parses their content, and uses built-in Natural Language Processing (NLP) capabilities to generate concise summaries.

### Sentiment Analysis

- **Primary Model**: The **VADER (Valence Aware Dictionary and sEntiment Reasoner)** model from the **NLTK** library is used for sentiment analysis.
  - VADER is a lexicon and rule-based tool optimized for analyzing sentiments in social media and general text, providing positive, negative, neutral, and compound scores.
- **Optional Advanced Model**: An attempt is made to use a sentiment analysis pipeline from the **`transformers`** library (e.g., a pre-trained transformer model), but it is optional and serves as a fallback if available.

### Text-to-Speech (TTS)

- **Speech Generation**: The **`gTTS` (Google Text-to-Speech)** library generates audio from text.
- **Translation**: Before generating speech, the summary text is translated from English to Hindi using the **`googletrans`** library, enabling Hindi audio summaries.

---

## API Development

The backend is developed using **FastAPI** and exposes the following endpoints for news retrieval and analysis:

- **POST /news**:
  - **Description**: Accepts a JSON body with a `company_name` field and returns news articles with sentiment analysis and a Hindi audio summary for the specified company.
  - **Request Body Example**:
    ```json
    {
        "company_name": "Microsoft"
    }
    ```
  - **Access via curl**:
    ```bash
    curl -X POST "http://localhost:8000/news" -H "Content-Type: application/json" -d '{"company_name": "Microsoft"}'
    ```
  - **Access via Postman**:
    - Set method to `POST`.
    - URL: `http://localhost:8000/news`.
    - Headers: `Content-Type: application/json`.
    - Body: Raw JSON (as shown above).

- **GET /news/{company_name}**:
  - **Description**: Retrieves news articles, sentiment analysis, and a Hindi audio summary for the specified company using a URL parameter.
  - **Example Request**:
    ```bash
    curl -X GET "http://localhost:8000/news/Microsoft"
    ```
  - **Access via Postman**:
    - Set method to `GET`.
    - URL: `http://localhost:8000/news/Microsoft`.

- **Response Format**:
  Both endpoints return a JSON object containing:
  - `company`: The company name.
  - `status`: Success or error status.
  - `articles_count`: Number of articles retrieved.
  - `articles`: List of articles with titles, summaries, sentiments, topics, URLs, sources, and dates.
  - `comparative_analysis`: Sentiment distribution, coverage insights, and topic overlap.
  - `summary`: A final text summary.
  - `audio_file`: Path to the generated Hindi audio file (if available).

- **Testing Tools**: Use **Postman**, **curl**, or visit `http://localhost:8000/docs` for the interactive FastAPI Swagger UI to test the endpoints.

---

## API Usage

### Third-Party APIs

- **GoogleNews**:
  - **Purpose**: Fetches recent news articles related to the specified company.
  - **Integration**: The `GoogleNews` library is used to search for news within a 7-day period, providing article metadata (e.g., title, URL, date) that is then processed further.

- **googletrans**:
  - **Purpose**: Translates English summary text into Hindi for TTS.
  - **Integration**: The `googletrans` library translates text before it is passed to `gTTS`, enabling Hindi audio output.

**Note**: Neither `GoogleNews` nor `googletrans` requires API keys when used through their respective libraries, simplifying integration.

---

## Assumptions & Limitations

### Assumptions

- **News Availability**: The project assumes that sufficient news articles (at least 3) are available for the specified company within the last 7 days via GoogleNews.


### Limitations


- **Sentiment Analysis**:
  - The VADER model, while fast and effective for general text, is lexicon-based and may miss complex sentiments, sarcasm, or context-specific nuances.

- **Translation Accuracy**:
  - Translation from English to Hindi via `googletrans` may not always be accurate, potentially affecting the quality and intelligibility of the Hindi audio summary.

- **Rate Limiting**:
  - Frequent use of `GoogleNews` or `googletrans` may encounter rate limits imposed by Google, though no explicit API key-based restrictions are coded.

