# Intelligent Learning System

Functional hackathon prototype for adaptive learning with explainable analytics.

## Problem
Traditional classrooms and online platforms often deliver one-size-fits-all quizzes. Students with different mastery levels get the same sequence, reducing engagement and slowing learning.

## Target User
- Middle/high school students practicing math topics.
- Teachers who need measurable progress and concept-level insight.

## What This Prototype Demonstrates
- Adaptive question sequencing based on recent accuracy and response time.
- Measurable learning metrics (accuracy, response speed, learning gain).
- Concept-level mastery tracking.
- Structured, explainable recommendations for next learning actions.
- Basic academic integrity flagging using response-time behavior patterns.

## Tech Stack
- Python
- Streamlit
- Pandas
- Plotly

## Run Locally
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## System Flow
1. Student selects a topic and starts a session.
2. Engine picks next question difficulty from recent performance.
3. Student answers; system logs correctness + response time.
4. Dashboard updates rolling accuracy, time trends, and learning gain.
5. Engine outputs structured recommendations and integrity flags.


