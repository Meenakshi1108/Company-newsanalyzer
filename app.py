import streamlit as st
import requests
import json
import pandas as pd
import time
import os
import base64
import altair as alt
import random


st.set_page_config(
    page_title="Company News Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"


def fetch_company_news(company_name):
    """Fetch company news from the API."""
    try:
        url = f"{API_URL}/news/{company_name}" if API_URL else f"/news/{company_name}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def get_audio_player(audio_file):
    """Generate HTML for audio player."""
    audio_bytes = open(audio_file, "rb").read()
    b64 = base64.b64encode(audio_bytes).decode()
    return (f'<audio controls><source src="data:audio/mp3;base64,{b64}" '
            f'type="audio/mp3"></audio>')


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f9f9f9;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin: 1rem 0;
    }
    .highlight-text {
        color: #1E88E5;
        font-weight: bold;
    }
    .footer-text {
        text-align: center;
        color: #616161;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main app UI
st.markdown(
    '<h1 class="main-header">📰 Company News Analyzer</h1>',
    unsafe_allow_html=True
)
st.markdown("""
<div style="text-align: center; padding: 0 3rem 2rem 3rem;">
    Get comprehensive news analysis, sentiment tracking, and insights about any company.
    <br>Complete with Hindi audio summaries!
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<h3 style="margin-bottom:0">🔍 Enter a Company</h3>',
    unsafe_allow_html=True
)

search_col, suggestion_col = st.columns([3, 1])

with search_col:
    company_name = st.text_input(
        "Company name",
        placeholder="Type company name here...",
        label_visibility="collapsed"
    )

with suggestion_col:
    company_options = [
        "Apple", "Google", "Microsoft", "Amazon", "Meta",
        "Netflix", "Tesla", "NVIDIA", "Intel", "SpaceX",
        "JPMorgan Chase", "Salesforce"
    ]

    if st.button("🎲 Surprise Me"):
        random_company = random.choice(company_options)
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = random_company
        company_name = random_company

# Add company categories
st.markdown(
    '<h3 style="margin-bottom:10px">🏢 Browse by Category</h3>',
    unsafe_allow_html=True
)

# Create tabs for categories
tech_tab, finance_tab= st.tabs([
    "💻 Tech", "🏦 Finance"
])

with tech_tab:
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

    if tech_col1.button("🍎 Apple", use_container_width=True):
        company_name = "Apple"
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = "Apple"

    if tech_col2.button("🔍 Google", use_container_width=True):
        company_name = "Google"
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = "Google"

    if tech_col3.button("💻 Microsoft", use_container_width=True):
        company_name = "Microsoft"
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = "Microsoft"

    if tech_col4.button("⚡ NVIDIA", use_container_width=True):
        company_name = "NVIDIA"
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = "NVIDIA"

with finance_tab:
    fin_col1, fin_col2 = st.columns(2)

    if fin_col1.button("🏦 JPMorgan", use_container_width=True):
        company_name = "JPMorgan Chase"
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = "JPMorgan Chase"

    if fin_col2.button("💳 Visa", use_container_width=True):
        company_name = "Visa"
        st.session_state.analyze_triggered = True
        st.session_state.analyze_company = "Visa"

st.markdown("### 🏢 Quick Select")

# Create multiple rows for more companies
row1 = st.columns(6)
row2 = st.columns(6)

# Create a session state to track if analysis should run
if 'analyze_triggered' not in st.session_state:
    st.session_state.analyze_triggered = False
    st.session_state.analyze_company = ""

if 'current_result' not in st.session_state:
    st.session_state.current_result = None
    st.session_state.current_company = ""

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # Default to first tab

if row1[0].button("🍎 Apple"):
    company_name = "Apple"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Apple"
if row1[1].button("🔍 Google"):
    company_name = "Google"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Google"
if row1[2].button("📦 Amazon"):
    company_name = "Amazon"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Amazon"
if row1[3].button("🚗 Tesla"):
    company_name = "Tesla"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Tesla"
if row1[4].button("👥 Meta"):
    company_name = "Meta"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Meta"
if row1[5].button("🎬 Netflix"):
    company_name = "Netflix"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Netflix"

if row2[0].button("💻 Microsoft"):
    company_name = "Microsoft"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Microsoft"
if row2[1].button("🏦 JPMorgan"):
    company_name = "JPMorgan Chase"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "JPMorgan Chase"
if row2[2].button("🚀 SpaceX"):
    company_name = "SpaceX"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "SpaceX"
if row2[3].button("⚡ NVIDIA"):
    company_name = "NVIDIA"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "NVIDIA"
if row2[4].button("☁️ Salesforce"):
    company_name = "Salesforce"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Salesforce"
if row2[5].button("🔬 Intel"):
    company_name = "Intel"
    st.session_state.analyze_triggered = True
    st.session_state.analyze_company = "Intel"

st.write("")
st.write("")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button(
        "🔍 ANALYZE COMPANY NEWS",
        use_container_width=True,
        type="primary"
    )
    st.markdown(
        '<p style="text-align: center; font-size: 12px;">'
        'Click to fetch and analyze latest news</p>',
        unsafe_allow_html=True
    )

# Enhanced loading experience
if analyze_clicked or st.session_state.analyze_triggered:
    # Reset the trigger after starting analysis
    if st.session_state.analyze_triggered:
        company_name = st.session_state.analyze_company
        st.session_state.analyze_triggered = False
        
    if company_name:
        # Set flag to show we have results that should be displayed
        st.session_state.show_results = True
        
        # Only fetch new data if company changed
        if company_name != st.session_state.current_company:
            with st.spinner():
                st.markdown(f"""
                <div style="display:flex; flex-direction:column; align-items:center; 
                      justify-content:center; padding:20px;">
                    <p style="margin-top:15px; font-size:16px; color:#1E88E5">
                        Analyzing news for <strong>{company_name}</strong>...</p>
                    <p style="font-size:12px; color:#757575; margin-top:5px">
                        This might take a few moments</p>
                </div>
                """, unsafe_allow_html=True)

                # Show a realistic progress simulation
                progress_bar = st.progress(0)
                for i in range(100):
                    # Slow down in the middle to simulate real processing
                    if 30 <= i <= 70:
                        time.sleep(0.03)
                    else:
                        time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Fetch data
                result = fetch_company_news(company_name)
                
                # Store in session state
                if result and result.get("status") == "success":
                    st.session_state.current_company = company_name
                    st.session_state.current_result = result
        else:
            # Use cached result
            result = st.session_state.current_result

        # Remove progress bar after loading
        st.empty()

        if result and result.get("status") == "success":
            # Save result in session state
            st.session_state.current_company = company_name
            st.session_state.current_result = result
            
            st.success(f"Found {result.get('articles_count')} articles for {company_name}")

            comp_analysis = result.get("comparative_analysis", {})
            tabs = ["📊 Analysis Overview", "📰 News Articles", "🔊 Hindi Summary"]
            overview_tab, articles_tab, audio_tab = st.tabs(tabs)


            with overview_tab:
                # Create two columns for sentiment analysis
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_dist = comp_analysis.get("sentiment_distribution", {})
                    sentiment_df = pd.DataFrame({
                        "Sentiment": sentiment_dist.keys(),
                        "Count": sentiment_dist.values()
                    })

                    chart = alt.Chart(sentiment_df).mark_bar().encode(
                        x=alt.X('Sentiment', sort=None),
                        y='Count',
                        color=alt.Color('Sentiment', scale=alt.Scale(
                            domain=['Positive', 'Neutral', 'Negative'],
                            range=['#4CAF50', '#2196F3', '#F44336']
                        ))
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

                with col2:
                    st.subheader("Overall Sentiment")
                    final_sentiment = comp_analysis.get("final_sentiment", "")

                    with st.container():
                        st.markdown(f"""
                        <div class="summary-box" style="background-color:#f0f7ff; padding:20px; 
                              border-radius:10px; border-left:5px solid #1E88E5;">
                            <h4 style="margin-top:0">Summary</h4>
                            <p>{final_sentiment}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Display common topics with better visualization
                common_topics = (comp_analysis.get("topic_overlap", {})
                                .get("common_topics", []))
                if common_topics:
                    st.subheader("Key Topics")

                    topic_html = ""
                    for topic in common_topics:
                        topic_html += (
                            f'<span class="topic-pill" style="background-color:#e1f5fe; margin:5px; '
                            f'padding:8px 15px; border-radius:20px; '
                            f'display:inline-block">{topic}</span>'
                        )

                    st.markdown(f"""
                    <div style="margin:15px 0">
                        {topic_html}
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("News Coverage Insights")
                for i, comparison in enumerate(
                        comp_analysis.get("coverage_differences", [])):
                    st.markdown(f"""
                    <div class="insight-card" style="background-color:#f9f9f9; margin:10px 0; padding:15px; 
                          border-radius:5px; border:1px solid #eaeaea">
                        <p><strong>📊 Observation {i+1}:</strong> 
                           {comparison.get('Comparison')}</p>
                        <p><strong>💡 Impact:</strong> {comparison.get('Impact')}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with articles_tab:
                # First row with search controls
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Add a form to contain the search input
                    with st.form(key="search_form"):
                        search_term = st.text_input(
                            "Search in articles",
                            placeholder="Filter by keyword..."
                        )
                        submitted = st.form_submit_button("Search", type="primary")
                
                with col2:
                    sentiment_filter = st.multiselect(
                        "Filter by sentiment",
                        ["Positive", "Neutral", "Negative"],
                        default=["Positive", "Neutral", "Negative"]
                    )
                # Filter and display articles
                articles = result.get("articles", [])
                filtered_articles = [
                    a for a in articles
                    if (not search_term or
                        search_term.lower() in a.get('title', '').lower() or
                        search_term.lower() in a.get('summary', '').lower())
                    and a.get('sentiment') in sentiment_filter
                ]

                if not filtered_articles:
                    st.info("No articles match your filters.")

                col1, col2 = st.columns(2)
                for i, article in enumerate(filtered_articles):
                    with (col1 if i % 2 == 0 else col2):
                        sentiment = article.get('sentiment', '')
                        sentiment_color = {
                            "Positive": "#4CAF50",
                            "Neutral": "#2196F3",
                            "Negative": "#F44336"
                        }.get(sentiment, "#9E9E9E")

                        st.markdown(f"""
                        <div style="border:1px solid #e0e0e0; border-radius:8px; 
                              padding:15px; margin-bottom:15px; 
                              border-left:5px solid {sentiment_color};">
                            <h4>{article.get('title')}</h4>
                            <p style="color:#616161; font-size:0.9em; margin-bottom:10px">
                                <span style="background-color:{sentiment_color}; 
                                      color:white; padding:3px 8px; border-radius:4px; 
                                      font-size:0.8em">
                                    {sentiment}
                                </span>
                                <span style="margin-left:10px">{article.get('date')}</span> • 
                                <span>{article.get('source')}</span>
                            </p>
                            <p>{article.get('summary')[:150]}...</p>
                            <p><a href="{article.get('url')}" target="_blank">Read more</a></p>
                        </div>
                        """, unsafe_allow_html=True)

            with audio_tab:
                audio_file = result.get("audio_file")

                st.markdown("""
                <div style="text-align:center; padding:20px 0;">
                    <h3>🎧 Hindi Audio Summary</h3>
                    <p>Listen to a synthesized summary of the news in Hindi</p>
                </div>
                """, unsafe_allow_html=True)

                if audio_file:
                    try:
                        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                audio_bytes = open(audio_file, "rb").read()
                                st.audio(audio_bytes, format="audio/mp3")

                                st.download_button(
                                    label="Download Audio File",
                                    data=audio_bytes,
                                    file_name=f"{company_name}_hindi_summary.mp3",
                                    mime="audio/mp3"
                                )
                        else:
                            st.warning(
                                "Audio file exists but may be empty or inaccessible."
                            )
                    except Exception as e:
                        st.error(f"Error displaying audio: {str(e)}")
                else:
                    st.markdown("""
                    <div style="background-color:#f5f5f5; border-radius:10px; 
                          padding:30px; text-align:center; margin:30px 0;">
                        <img src="https://img.icons8.com/fluency/96/000000/no-audio.png" 
                             style="opacity:0.6"/>
                        <p style="margin-top:15px">
                            No Hindi audio summary available for this company.</p>
                        <p style="font-size:0.9em; color:#757575">
                            Try another company or check back later.</p>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.warning("Please enter a company name.")

elif st.session_state.show_results and st.session_state.current_result is not None:
    # Retrieve stored values from session state
    company_name = st.session_state.current_company
    result = st.session_state.current_result
    
    # Display the results
    st.success(f"Found {result.get('articles_count')} articles for {company_name}")
    
    # Create tabs for better organization
    comp_analysis = result.get("comparative_analysis", {})
    overview_tab, articles_tab, audio_tab = st.tabs([
        "📊 Analysis Overview", "📰 News Articles", "🔊 Hindi Summary"
    ])
    
    with overview_tab:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_dist = comp_analysis.get("sentiment_distribution", {})
            sentiment_df = pd.DataFrame({
                "Sentiment": sentiment_dist.keys(),
                "Count": sentiment_dist.values()
            })
            
            # Chart code
            chart = alt.Chart(sentiment_df).mark_bar().encode(
                x=alt.X('Sentiment', sort=None),
                y='Count',
                color=alt.Color('Sentiment', scale=alt.Scale(
                    domain=['Positive', 'Neutral', 'Negative'],
                    range=['#4CAF50', '#2196F3', '#F44336']
                ))
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            
        with col2:
            st.subheader("Overall Sentiment")
            final_sentiment = comp_analysis.get("final_sentiment", "")
            
            with st.container():
                st.markdown(f"""
                <div class="summary-box" style="background-color:#f0f7ff; padding:20px; 
                      border-radius:10px; border-left:5px solid #1E88E5;">
                    <h4 style="margin-top:0">Summary</h4>
                    <p>{final_sentiment}</p>
                </div>
                """, unsafe_allow_html=True)
                
        # Topics display code
        common_topics = (comp_analysis.get("topic_overlap", {})
                        .get("common_topics", []))
        if common_topics:
            st.subheader("Key Topics")
            
            topic_html = ""
            for topic in common_topics:
                topic_html += (
                    f'<span class="topic-pill" style="background-color:#e1f5fe; '
                    f'margin:5px; padding:8px 15px; border-radius:20px; '
                    f'display:inline-block">{topic}</span>'
                )
                
            st.markdown(f"""
            <div style="margin:15px 0">
                {topic_html}
            </div>
            """, unsafe_allow_html=True)
            
        # Coverage differences display
        st.subheader("News Coverage Insights")
        for i, comparison in enumerate(
                comp_analysis.get("coverage_differences", [])):
            st.markdown(f"""
            <div class="insight-card" style="background-color:#f9f9f9; 
                  margin:10px 0; padding:15px; border-radius:5px; 
                  border:1px solid #eaeaea">
                <p><strong>📊 Observation {i+1}:</strong> 
                   {comparison.get('Comparison')}</p>
                <p><strong>💡 Impact:</strong> {comparison.get('Impact')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with articles_tab:
    
        search_term = st.text_input(
            "Search in articles",
            placeholder="Filter by keyword..."
        )
        
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"]
        )
        
        articles = result.get("articles", [])
        filtered_articles = [
            a for a in articles
            if (not search_term or
                search_term.lower() in a.get('title', '').lower() or
                search_term.lower() in a.get('summary', '').lower())
            and a.get('sentiment') in sentiment_filter
        ]
        
        if not filtered_articles:
            st.info("No articles match your filters.")
            
        col1, col2 = st.columns(2)
        for i, article in enumerate(filtered_articles):
            with (col1 if i % 2 == 0 else col2):
                sentiment = article.get('sentiment', '')
                sentiment_color = {
                    "Positive": "#4CAF50",
                    "Neutral": "#2196F3",
                    "Negative": "#F44336"
                }.get(sentiment, "#9E9E9E")
                
                st.markdown(f"""
                <div style="border:1px solid #e0e0e0; border-radius:8px; 
                      padding:15px; margin-bottom:15px; 
                      border-left:5px solid {sentiment_color};">
                    <h4>{article.get('title')}</h4>
                    <p style="color:#616161; font-size:0.9em; margin-bottom:10px">
                        <span style="background-color:{sentiment_color}; 
                              color:white; padding:3px 8px; border-radius:4px; 
                              font-size:0.8em">
                            {sentiment}
                        </span>
                        <span style="margin-left:10px">{article.get('date')}</span> • 
                        <span>{article.get('source')}</span>
                    </p>
                    <p>{article.get('summary')[:150]}...</p>
                    <p><a href="{article.get('url')}" target="_blank">Read more</a></p>
                </div>
                """, unsafe_allow_html=True)
    
    with audio_tab:
        audio_file = result.get("audio_file")

        st.markdown("""
        <div style="text-align:center; padding:20px 0;">
            <h3>🎧 Hindi Audio Summary</h3>
            <p>Listen to a synthesized summary of the news in Hindi</p>
        </div>
        """, unsafe_allow_html=True)

        if audio_file:
            try:
                # Check if file exists and has content
                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        audio_bytes = open(audio_file, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")

                        st.download_button(
                            label="Download Audio File",
                            data=audio_bytes,
                            file_name=f"{company_name}_hindi_summary.mp3",
                            mime="audio/mp3"
                        )
                else:
                    st.warning(
                        "Audio file exists but may be empty or inaccessible."
                    )
            except Exception as e:
                st.error(f"Error displaying audio: {str(e)}")
        else:
            st.markdown("""
            <div style="background-color:#f5f5f5; border-radius:10px; 
                    padding:30px; text-align:center; margin:30px 0;">
                <img src="https://img.icons8.com/fluency/96/000000/no-audio.png" 
                        style="opacity:0.6"/>
                <p style="margin-top:15px">
                    No Hindi audio summary available for this company.</p>
                <p style="font-size:0.9em; color:#757575">
                    Try another company or check back later.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This application extracts news from various sources, performs sentiment analysis, 
and provides a consolidated view of how a company is being portrayed in the media.
It also generates Hindi audio summaries using text-to-speech technology.
""")