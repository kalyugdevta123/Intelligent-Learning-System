import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    AttemptRecord,
    explain_recommendations,
    get_adaptive_stage,
    integrity_flags,
    select_next_question,
    summarize_progress,
)
from pdf_importer import default_topic_key, import_pdf_to_topic_dict


DATA_PATH = Path(__file__).parent / "data" / "topics.json"
IMPORTED_TOPICS_PATH = Path(__file__).parent / "data" / "imported_topics.json"


def load_topics() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        base_topics = json.load(f)

    imported_topics: dict = {}
    if IMPORTED_TOPICS_PATH.exists():
        with IMPORTED_TOPICS_PATH.open("r", encoding="utf-8") as f:
            imported_topics = json.load(f)

    # Imported topics override same keys if they exist.
    base_topics.update(imported_topics)
    return base_topics


def init_state() -> None:
    defaults = {
        "topic_key": None,
        "topic_questions": [],
        "attempts": [],
        "baseline_accuracy": 0.0,
        "current_question": None,
        "question_start_time": None,
        "used_ids": set(),
        "last_feedback": None,
        "session_active": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_question() -> None:
    next_q = select_next_question(
        st.session_state.topic_questions,
        st.session_state.attempts,
        st.session_state.used_ids,
    )
    st.session_state.current_question = next_q
    if next_q is None:
        st.session_state.question_start_time = None
    else:
        st.session_state.question_start_time = time.time()


def restart_session(topic_key: str, topics: dict) -> None:
    st.session_state.topic_key = topic_key
    st.session_state.topic_questions = topics[topic_key]["questions"]
    st.session_state.attempts = []
    st.session_state.used_ids = set()
    st.session_state.current_question = None
    st.session_state.question_start_time = None
    st.session_state.session_active = True
    st.session_state.baseline_accuracy = 0.0
    set_question()


def record_answer(selected_index: int) -> None:
    q = st.session_state.current_question
    if q is None or st.session_state.question_start_time is None:
        return

    elapsed = max(0.5, time.time() - st.session_state.question_start_time)
    is_correct = selected_index == q["answer_index"]

    st.session_state.attempts.append(
        AttemptRecord(
            question_id=q["id"],
            concept=q["concept"],
            difficulty=q["difficulty"],
            correct=is_correct,
            response_time_s=elapsed,
            selected_index=selected_index,
        )
    )
    st.session_state.used_ids.add(q["id"])

    if len(st.session_state.attempts) == 3:
        first_three = summarize_progress(st.session_state.attempts)
        st.session_state.baseline_accuracy = first_three["accuracy"]

    set_question()


def render_dashboard() -> None:
    st.subheader("Performance Dashboard")
    summary = summarize_progress(st.session_state.attempts, st.session_state.baseline_accuracy)
    # Keep this lightweight during practice; show full explainability only at the end.
    include_recommendations = st.session_state.get("include_recommendations", False)
    recs = explain_recommendations(summary) if include_recommendations else []
    flags = integrity_flags(st.session_state.attempts) if include_recommendations else []

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f'{summary["accuracy"]}%')
    c2.metric("Avg Response Time", f'{summary["avg_time_s"]} s')
    c3.metric("Learning Gain", f'{summary["learning_gain_pp"]:+} pp')

    attempts_df = pd.DataFrame(
        [
            {
                "question_id": a.question_id,
                "difficulty": a.difficulty,
                "correct": 1 if a.correct else 0,
                "response_time_s": a.response_time_s,
                "concept": a.concept,
            }
            for a in st.session_state.attempts
        ]
    )

    if not attempts_df.empty:
        acc_fig = px.line(
            attempts_df.assign(
                rolling_acc=attempts_df["correct"].rolling(window=3, min_periods=1).mean() * 100
            ),
            y="rolling_acc",
            title="Rolling Accuracy (3-question window)",
            markers=True,
        )
        acc_fig.update_layout(yaxis_title="Accuracy %", xaxis_title="Question Number")
        st.plotly_chart(acc_fig, use_container_width=True)

        time_fig = px.bar(
            attempts_df,
            y="response_time_s",
            color="difficulty",
            title="Response Time per Question",
        )
        time_fig.update_layout(yaxis_title="Seconds", xaxis_title="Attempt")
        st.plotly_chart(time_fig, use_container_width=True)

    if include_recommendations:
        st.subheader("Explainable Recommendations")

        # Structured summary with readable, colored cues.
        acc = summary.get("accuracy", 0.0)
        if acc >= 80:
            st.success(f'Accuracy: {acc}% (strong performance)')
        elif acc >= 60:
            st.warning(f'Accuracy: {acc}% (improving)')
        else:
            st.error(f'Accuracy: {acc}% (needs reinforcement)')

        st.markdown(
            f"**Attempts:** {summary.get('attempt_count', 0)} | "
            f"**Avg Response Time:** {summary.get('avg_time_s', 0.0)} s | "
            f"**Learning Gain:** {summary.get('learning_gain_pp', 0.0):+.1f} pp"
        )

        st.divider()
        st.subheader("Recommended Next Steps")
        if not recs:
            st.info("No actions recommended yet.")
        else:
            for action in recs:
                priority = action.get("priority", "low")
                action_text = action.get("action", "")
                why_text = action.get("why", "")

                if priority == "high":
                    st.error(action_text)
                elif priority == "medium":
                    st.warning(action_text)
                else:
                    st.info(action_text)

                if why_text:
                    st.caption(f"Why: {why_text}")
                st.write("")  # small spacing between cards

        st.divider()
        st.subheader("Integrity Signals")
        if not flags:
            st.success("No unusual timing patterns detected.")
        else:
            for f in flags:
                st.warning(f)


def render_adaptive_stage() -> None:
    stage = get_adaptive_stage(st.session_state.attempts)
    st.subheader(f'Adaptive Stage: {stage["stage"]}')
    tone = stage.get("tone", "info")
    message = stage.get("text", "")
    if tone == "success":
        st.success(message)
    elif tone == "warning":
        st.warning(message)
    elif tone == "error":
        st.error(message)
    else:
        st.info(message)


def main() -> None:
    st.set_page_config(page_title="Intelligent Learning System", layout="wide")
    st.title("Intelligent Learning System")
    st.caption(
        "Adaptive quiz engine with measurable learning gains, data-driven personalization, and structured explainable outputs."
    )

    init_state()
    topics = load_topics()

    # Default: hide structured recommendations while the student is still answering.
    if "include_recommendations" not in st.session_state:
        st.session_state.include_recommendations = False

    with st.sidebar:
        st.header("Session Setup")
        topic_label_to_key = {v["title"]: k for k, v in topics.items()}
        selected_label = st.selectbox("Choose topic", list(topic_label_to_key.keys()))
        selected_key = topic_label_to_key[selected_label]

        if st.button("Start New Session", type="primary"):
            restart_session(selected_key, topics)

        with st.expander("Upload Quiz PDF (optional)"):
            st.markdown("Upload a professor quiz PDF. We will extract MCQs and add them to the question bank.")
            topic_title = st.text_input("Topic title for imported quiz", value="Imported Quiz")
            uploaded = st.file_uploader("Quiz PDF", type=["pdf"])
            max_questions = st.number_input("Max questions to extract", min_value=3, max_value=40, value=12, step=1)
            max_pages = st.number_input("Max PDF pages to parse", min_value=2, max_value=40, value=10, step=1)

            if st.button("Import PDF"):
                if uploaded is None:
                    st.warning("Please upload a PDF first.")
                else:
                    with st.spinner("Extracting text and generating MCQs..."):
                        # Load existing imported topics so we can assign a new unique key.
                        imported_topics: dict = {}
                        if IMPORTED_TOPICS_PATH.exists():
                            with IMPORTED_TOPICS_PATH.open("r", encoding="utf-8") as f:
                                imported_topics = json.load(f)

                        try:
                            # Ensure unique topic key.
                            new_topic_key = default_topic_key(
                                topic_title=topic_title,
                                existing_keys=set(topics.keys()) | set(imported_topics.keys()),
                            )

                            imported_topic = import_pdf_to_topic_dict(
                                pdf_bytes=uploaded.read(),
                                topic_title=topic_title,
                                topic_key=new_topic_key,
                                max_pages=int(max_pages),
                                max_questions=int(max_questions),
                            )

                            imported_topics[new_topic_key] = imported_topic
                            IMPORTED_TOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)
                            with IMPORTED_TOPICS_PATH.open("w", encoding="utf-8") as f:
                                json.dump(imported_topics, f, ensure_ascii=False, indent=2)

                            st.success("Quiz imported successfully. Reloading topics...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import failed: {e}")

        st.markdown("### Completion")
        st.write(f'Answered: `{len(st.session_state.attempts)}` / `{len(st.session_state.topic_questions)}`')

    if not st.session_state.session_active:
        st.info("Start a session from the sidebar.")
        return

    q = st.session_state.current_question

    # Show the most recent answer feedback (so it doesn't disappear after st.rerun()).
    # This must run before the `q is None` completion branch.
    if st.session_state.last_feedback is not None:
        fb = st.session_state.last_feedback
        st.subheader("Answer Explanation")
        if fb["is_correct"]:
            st.success("Correct.")
        else:
            st.error("Incorrect.")
        st.markdown(f"**Correct answer:** {fb['correct_option']}")
        st.caption(f"Explanation: {fb['explanation']}")
        st.session_state.last_feedback = None

    if q is None:
        st.success("Topic completed. Great work.")
        st.session_state.include_recommendations = True
        render_adaptive_stage()
        render_dashboard()
        return

    render_adaptive_stage()
    st.subheader(f'Question ({q["difficulty"].upper()})')
    st.write(q["text"])

    selected = st.radio("Pick one option:", q["options"], index=None, key=f'answer_{q["id"]}')
    submit = st.button("Submit Answer")
    if submit and selected is not None:
        selected_idx = q["options"].index(selected)
        is_correct = selected_idx == q["answer_index"]

        # Persist feedback across reruns so the user sees the right answer + explanation.
        correct_text = q["options"][q["answer_index"]]
        st.session_state.last_feedback = {
            "is_correct": is_correct,
            "correct_option": correct_text,
            "explanation": q["explanation"],
        }

        record_answer(selected_idx)
        st.rerun()

    st.divider()
    render_dashboard()


if __name__ == "__main__":
    main()
