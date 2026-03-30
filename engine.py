import random
from dataclasses import dataclass
from statistics import mean, median


DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_SCORE = {"easy": 1, "medium": 2, "hard": 3}
ORDER = DIFFICULTIES

# Zone of proximal development target: avoid frustration (too many wrongs) and avoid boredom (too many corrects).
P_TARGET_SUCCESS = 0.55

EASY_STREAK_TO_UNLOCK_MEDIUM = 5
MEDIUM_STREAK_TO_UNLOCK_HARD = 5
MEDIUM_WINDOW = 5
MEDIUM_WRONG_MAJORITY = 3
REMEDIATION_EASY_COUNT = 3


@dataclass
class AttemptRecord:
    question_id: str
    concept: str
    difficulty: str
    correct: bool
    response_time_s: float
    selected_index: int | None = None


def _bandit_params_for(attempts: list[AttemptRecord]) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Build Beta(alpha,beta) parameters per (concept,difficulty) from attempts.
    This is an online-learning model: each wrong increments beta, each correct increments alpha.
    """
    state: dict[str, dict[str, tuple[float, float]]] = {}
    for a in attempts:
        concept = a.concept
        difficulty = a.difficulty
        state.setdefault(concept, {})
        alpha, beta = state[concept].get(difficulty, (1.0, 1.0))
        if a.correct:
            alpha += 1.0
        else:
            beta += 1.0
        state[concept][difficulty] = (alpha, beta)

    # Ensure every difficulty exists for each concept we saw.
    for concept in list(state.keys()):
        for d in DIFFICULTIES:
            state[concept].setdefault(d, (1.0, 1.0))

    return state


def _expected_success(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _correct_streak_for_difficulty(attempts: list[AttemptRecord], difficulty: str) -> int:
    streak = 0
    for a in reversed(attempts):
        if a.difficulty != difficulty:
            break
        if not a.correct:
            break
        streak += 1
    return streak


def _latest_medium_remediation_state(
    attempts: list[AttemptRecord],
) -> tuple[bool, set[str]]:
    
    medium_indices = [i for i, a in enumerate(attempts) if a.difficulty == "medium"]
    if len(medium_indices) < MEDIUM_WINDOW:
        return False, set()

    last_window_indices = medium_indices[-MEDIUM_WINDOW:]
    last_medium_window = [attempts[i] for i in last_window_indices]
    wrong_medium = [a for a in last_medium_window if not a.correct]
    if len(wrong_medium) < MEDIUM_WRONG_MAJORITY:
        return False, set()

    weak_concepts = {a.concept for a in wrong_medium}
    window_end = last_window_indices[-1]
    easy_after_window = [
        a for a in attempts[window_end + 1:]
        if a.difficulty == "easy"
    ]

    still_in_remediation = len(easy_after_window) < REMEDIATION_EASY_COUNT
    return still_in_remediation, weak_concepts


def _pick_from_constraints(
    remaining: list[dict],
    bandit: dict[str, dict[str, tuple[float, float]]],
    allowed_difficulties: set[str],
    preferred_concepts: set[str] | None = None,
) -> dict | None:
    filtered = [q for q in remaining if q["difficulty"] in allowed_difficulties]
    if preferred_concepts:
        focused = [q for q in filtered if q["concept"] in preferred_concepts]
        if focused:
            filtered = focused

    if not filtered:
        return None

    concepts = sorted({q["concept"] for q in filtered})
    concept_to_focus = None
    lowest_expected = 1.1
    for c in concepts:
        exp_vals = []
        for d in allowed_difficulties:
            alpha, beta = bandit.get(c, {}).get(d, (1.0, 1.0))
            exp_vals.append(_expected_success(alpha, beta))
        avg_expected = mean(exp_vals) if exp_vals else 0.5
        if avg_expected < lowest_expected:
            lowest_expected = avg_expected
            concept_to_focus = c

    assert concept_to_focus is not None

    chosen_difficulty = None
    best_distance = 999.0
    for d in sorted(allowed_difficulties):
        alpha, beta = bandit.get(concept_to_focus, {}).get(d, (1.0, 1.0))
        sampled_p = random.betavariate(alpha, beta)
        distance = abs(sampled_p - P_TARGET_SUCCESS)
        if distance < best_distance:
            best_distance = distance
            chosen_difficulty = d

    assert chosen_difficulty is not None

    exact = [
        q for q in filtered
        if q["concept"] == concept_to_focus and q["difficulty"] == chosen_difficulty
    ]
    if exact:
        return random.choice(exact)

    same_concept = [q for q in filtered if q["concept"] == concept_to_focus]
    if same_concept:
        return random.choice(same_concept)

    return random.choice(filtered)


def get_adaptive_stage(attempts: list[AttemptRecord]) -> dict:
    easy_streak = _correct_streak_for_difficulty(attempts, "easy")
    medium_streak = _correct_streak_for_difficulty(attempts, "medium")
    remediation_active, weak_concepts = _latest_medium_remediation_state(attempts)

    if easy_streak < EASY_STREAK_TO_UNLOCK_MEDIUM:
        remaining = EASY_STREAK_TO_UNLOCK_MEDIUM - easy_streak
        return {
            "stage": "Foundation",
            "tone": "warning",
            "text": f"Building easy-level mastery. {remaining} more consecutive easy correct answers to unlock medium.",
        }

    if remediation_active:
        concept_text = ", ".join(sorted(weak_concepts)) if weak_concepts else "recent weak concepts"
        return {
            "stage": "Remediation",
            "tone": "error",
            "text": (
                f"Medium performance dipped (majority wrong in recent window). "
                f"Serving easy reinforcement for: {concept_text}."
            ),
        }

    if medium_streak < MEDIUM_STREAK_TO_UNLOCK_HARD:
        remaining = MEDIUM_STREAK_TO_UNLOCK_HARD - medium_streak
        return {
            "stage": "Medium Practice",
            "tone": "info",
            "text": f"Medium is active. {remaining} more consecutive medium correct answers to unlock hard.",
        }

    return {
        "stage": "Hard Challenge",
        "tone": "success",
        "text": "Hard is unlocked. Adaptive policy now mixes easy/medium/hard based on concept mastery.",
    }


def select_next_question(
    topic_questions: list[dict],
    attempts: list[AttemptRecord],
    used_ids: set[str],
) -> dict | None:
    remaining = [q for q in topic_questions if q["id"] not in used_ids]
    if not remaining:
        return None

    bandit = _bandit_params_for(attempts)

    medium_unlocked = _correct_streak_for_difficulty(attempts, "easy") >= EASY_STREAK_TO_UNLOCK_MEDIUM
    hard_unlocked = _correct_streak_for_difficulty(attempts, "medium") >= MEDIUM_STREAK_TO_UNLOCK_HARD
    remediation_active, weak_concepts = _latest_medium_remediation_state(attempts)

    if not medium_unlocked:
        pick = _pick_from_constraints(
            remaining=remaining,
            bandit=bandit,
            allowed_difficulties={"easy"},
        )
        if pick:
            return pick
        return random.choice(remaining)

    if remediation_active:
        pick = _pick_from_constraints(
            remaining=remaining,
            bandit=bandit,
            allowed_difficulties={"easy"},
            preferred_concepts=weak_concepts,
        )
        if pick:
            return pick

    if not hard_unlocked:
        # After easy mastery, prioritize medium (with fallback to easy).
        pick = _pick_from_constraints(
            remaining=remaining,
            bandit=bandit,
            allowed_difficulties={"medium", "easy"},
        )
        if pick:
            return pick
        return random.choice(remaining)

    # After medium mastery, allow full easy/medium/hard progression.
    pick = _pick_from_constraints(
        remaining=remaining,
        bandit=bandit,
        allowed_difficulties={"easy", "medium", "hard"},
    )
    if pick:
        return pick

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
    if len(attempts) < 6:
        return []

    times = [a.response_time_s for a in attempts]
    acc = [1 if a.correct else 0 for a in attempts]
    selected = [a.selected_index for a in attempts if a.selected_index is not None]

    flags: list[str] = []

    # Robust speed anomaly detection using median + MAD (median absolute deviation).
    med_t = median(times)
    mad = median([abs(t - med_t) for t in times])  # >= 0
    mad = max(mad, 1e-6)  # avoid divide-by-zero for near-constant times
    speed_threshold = med_t - 2.5 * mad

    speed_outliers = sum(1 for t in times if t <= speed_threshold)
    accuracy_rate = mean(acc)

    if accuracy_rate >= 0.85 and speed_outliers >= 3:
        flags.append("Speed-accuracy anomaly: unusually fast correct answers detected.")

    # Very low variance combined with high accuracy can indicate automation/memorization.
    near_zero_spread = (max(times) - min(times)) <= 1.0
    if accuracy_rate >= 0.8 and near_zero_spread:
        flags.append("Timing consistency anomaly: very low variance response times.")

    # Low-entropy option selection (same option repeatedly).
    if selected:
        freq: dict[int, int] = {}
        for s in selected:
            freq[s] = freq.get(s, 0) + 1
        best_fraction = max(freq.values()) / len(selected)
        if best_fraction >= 0.8 and accuracy_rate >= 0.5:
            flags.append("Low-entropy answering pattern: repeated option selection.")

    return flags
