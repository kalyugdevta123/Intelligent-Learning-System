import random
from dataclasses import dataclass
from statistics import mean


DIFFICULTY_SCORE = {"easy": 1, "medium": 2, "hard": 3}
ORDER = ["easy", "medium", "hard"]


@dataclass
class AttemptRecord:
    question_id: str
    concept: str
    difficulty: str
    correct: bool
    response_time_s: float


def select_next_difficulty(history: list[AttemptRecord]) -> str:
    if not history:
        return "easy"

    recent = history[-3:]
    recent_accuracy = mean(1.0 if item.correct else 0.0 for item in recent)
    recent_time = mean(item.response_time_s for item in recent)
    current_level = DIFFICULTY_SCORE[recent[-1].difficulty]

    # Increase challenge only when both correctness and speed are good.
    if recent_accuracy >= 0.8 and recent_time <= 22:
        current_level = min(3, current_level + 1)
    elif recent_accuracy <= 0.4 or recent_time > 35:
        current_level = max(1, current_level - 1)

    return ORDER[current_level - 1]


def pick_question(topic_questions: list[dict], difficulty: str, used_ids: set[str]) -> dict | None:
    same_level = [
        q for q in topic_questions
        if q["difficulty"] == difficulty and q["id"] not in used_ids
    ]
    if same_level:
        return random.choice(same_level)

    # Fallback if selected level is exhausted.
    remaining = [q for q in topic_questions if q["id"] not in used_ids]
    if not remaining:
        return None
    return random.choice(remaining)


def summarize_progress(
    attempts: list[AttemptRecord],
    baseline_accuracy: float = 0.0
) -> dict:
    if not attempts:
        return {
            "attempt_count": 0,
            "accuracy": 0.0,
            "avg_time_s": 0.0,
            "learning_gain_pp": 0.0,
            "concept_mastery": {}
        }

    accuracy = mean(1.0 if a.correct else 0.0 for a in attempts)
    avg_time = mean(a.response_time_s for a in attempts)

    concept_data: dict[str, list[bool]] = {}
    for item in attempts:
        concept_data.setdefault(item.concept, []).append(item.correct)

    concept_mastery = {
        concept: round(mean(1.0 if x else 0.0 for x in vals) * 100, 1)
        for concept, vals in concept_data.items()
    }

    return {
        "attempt_count": len(attempts),
        "accuracy": round(accuracy * 100, 1),
        "avg_time_s": round(avg_time, 1),
        "learning_gain_pp": round((accuracy * 100) - baseline_accuracy, 1),
        "concept_mastery": concept_mastery
    }


def explain_recommendations(summary: dict) -> list[dict]:
    actions: list[dict] = []

    if summary["accuracy"] < 60:
        actions.append({
            "priority": "high",
            "action": "Reinforce foundations using easier scaffolded questions.",
            "why": "Overall accuracy is below 60%."
        })

    if summary["avg_time_s"] > 30:
        actions.append({
            "priority": "medium",
            "action": "Add timed micro-practice to improve retrieval speed.",
            "why": "Average response time is above 30 seconds."
        })

    weak_concepts = [
        concept for concept, mastery in summary["concept_mastery"].items()
        if mastery < 50
    ]
    if weak_concepts:
        actions.append({
            "priority": "high",
            "action": f"Targeted revision for concepts: {', '.join(weak_concepts)}.",
            "why": "Concept mastery under 50% indicates persistent misconceptions."
        })

    if not actions:
        actions.append({
            "priority": "low",
            "action": "Increase challenge with more medium/hard mixed practice.",
            "why": "Current indicators show stable performance."
        })

    return actions


def integrity_flags(attempts: list[AttemptRecord]) -> list[str]:
    if len(attempts) < 4:
        return []

    times = [a.response_time_s for a in attempts]
    acc = [1 if a.correct else 0 for a in attempts]

    flags: list[str] = []
    if mean(times) < 3 and mean(acc) > 0.85:
        flags.append("Unusually fast and highly accurate pattern detected.")

    near_zero_spread = max(times) - min(times) < 0.8
    if near_zero_spread and len(times) >= 6:
        flags.append("Response times show very low variance; review attempt validity.")

    return flags
