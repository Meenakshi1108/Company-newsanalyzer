# Company News Analyzer

This project is a **Company News Analyzer** that extracts, summarizes, and analyzes news articles about a specified company. It provides sentiment analysis, topic extraction, and a Hindi audio summary of the news. The backend is built with **FastAPI**, and the frontend is a **Streamlit** web application.

**Live Demo**: [View the application on Hugging Face Spaces](https://huggingface.co/spaces/toph1108/Company_news_analyzer)

---

## Table of Contents

- Project Setup
  - Backend (FastAPI)
  - Frontend (Streamlit)
  - Virtual Environment
- Code Architecture
  - Modular Structure
- Model Details
  - Summarization
  - Sentiment Analysis
  - Text-to-Speech (TTS)
- API Development
- API Usage
  - Third-Party APIs
- Deployment
  - Hugging Face Spaces Deployment
- Assumptions & Limitations
  - Assumptions
  - Limitations

---

## Project Setup

### Virtual Environment

1. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Backend (FastAPI)

1. **Run the API server**:
   - Start the FastAPI server using Uvicorn:
     ```bash
     uvicorn api:app --host 0.0.0.0 --port 8000 --reload
     ```
   - The API will be available at `http://localhost:8000`.

### Frontend (Streamlit)

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   - The app will typically be available at `http://localhost:8501`.

## Code Architecture

### Modular Structure

The project uses a modular architecture for better maintainability and scalability:

```
code/
├── api.py             # FastAPI backend
├── app.py             # Streamlit frontend
├── requirements.txt
└── utils/             # Modularized utility functions
    ├── __init__.py    # Package exports and init functions
    ├── news_extractor.py     # News extraction functionality
    ├── sentiment_analyzer.py # Sentiment analysis components
    ├── text_to_speech.py     # Translation and TTS functionality
    └── analysis.py    # News analysis and processing
```

- **`news_extractor.py`**: Contains the `NewsExtractor` class for fetching and processing news articles using GoogleNews and newspaper3k.
- **`sentiment_analyzer.py`**: Contains the `SentimentAnalyzer` class implementing NLTK VADER and transformers-based sentiment analysis.
- **`text_to_speech.py`**: Contains the `TextToSpeechHindi` class for English-to-Hindi translation and audio generation.
- **`analysis.py`**: Provides functions for comprehensive news analysis including sentiment comparison and topic extraction.

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

## Deployment

### Hugging Face Spaces Deployment

This application is deployed on Hugging Face Spaces, a platform for hosting machine learning applications.

#### Deployment URL
- **Live Application**: [https://huggingface.co/spaces/toph1108/Company_news_analyzer](https://huggingface.co/spaces/toph1108/Company_news_analyzer)

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

- **Deployment Constraints**:
  - The Hugging Face Spaces deployment may have performance limitations compared to local deployment, especially with resource-intensive operations like news extraction and translation.