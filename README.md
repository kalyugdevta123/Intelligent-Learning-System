# Intelligent Learning System

Functional prototype for adaptive learning with explainable analytics.

## Problem
Traditional classrooms and online platforms often deliver one-size-fits-all quizzes. Students with different mastery levels get the same sequence, reducing engagement and slowing learning.

## Target User
- Middle/high school students practicing math topics.
- Teachers who need measurable progress and concept-level insight.

## What This Prototype Demonstrates
- Adaptive question sequencing using a Bayesian online-learning policy (concept + difficulty), implemented with Thompson sampling.
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
2. Engine predicts success probability per concept+difficulty (Bayesian bandit) and selects the next question near a “challenge sweet spot”.
3. Student answers; system logs correctness + response time.
4. Dashboard updates rolling accuracy, time trends, and learning gain.
5. Engine outputs structured recommendations and integrity flags.

## Optional: Upload Quiz PDF (Expands Question Bank)
This prototype supports uploading a professor quiz PDF, extracting text, and using a local LLM (LM Studio or an OpenAI-compatible endpoint) to convert it into MCQs that are stored in `data/imported_topics.json`.

### Configure the local LLM
Set environment variables before running Streamlit:
- `LLM_API_BASE` (default: `http://localhost:1234/v1`)
- `LLM_MODEL` (required; your LM Studio model id)
- `LLM_API_KEY` (optional; some local servers ignore it)

### Notes
- The PDF-to-MCQ accuracy depends on PDF quality and the LLM’s extraction ability.
- No “training” is required for the adaptive engine; it learns online from student attempts using Bayesian updates.


