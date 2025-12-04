import json
import requests
from difflib import SequenceMatcher
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# ========= Helper: Talk to local LLM via Ollama =========
def call_llm(prompt: str) -> str:
    """
    Calls a local LLM using Ollama and returns the generated text.
    Make sure Ollama is running:
        ollama pull llama3.2
        ollama run llama3.2
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def try_parse_json(text: str):
    """
    Robust JSON parser for LLM output.
    Handles code fences and extra text.
    """
    text = text.strip()

    # Strip ```json fences if present
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Could not parse JSON from LLM output")


def is_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Returns True if two strings are very similar (to avoid near-duplicate questions).
    """
    a_norm = " ".join(a.lower().split())
    b_norm = " ".join(b.lower().split())
    if not a_norm or not b_norm:
        return False
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold


# ================== Routes ==================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/practice")
def practice():
    role = request.args.get("role", "Software Quality Analyst")
    difficulty = request.args.get("difficulty", "easy")
    subject = request.args.get("subject", "Software Testing - Manual Basics")
    mode = request.args.get("mode", "both")  # descriptive / mcq / both

    # sanity check
    if mode not in ("descriptive", "mcq", "both"):
        mode = "both"

    return render_template(
        "practice.html",
        role=role,
        difficulty=difficulty,
        subject=subject,
        mode=mode
    )


# ================== Descriptive Question ==================

@app.route("/api/generate-question", methods=["POST"])
def generate_question():
    data = request.get_json() or {}
    role = data.get("role", "Software Quality Analyst")
    difficulty = data.get("difficulty", "easy")
    subject = data.get("subject", "Software Testing - Manual Basics")
    previous_questions = data.get("previous_questions", [])  # from frontend

    prev_block = ""
    if previous_questions:
        joined = "\n".join(f"- {q}" for q in previous_questions)
        prev_block = f"""
Previously asked questions in this session (do NOT repeat or trivially rephrase):

{joined}

""".strip()

    prompt = f"""
You are an interviewer hiring for a {role} position.

Generate ONE {difficulty} level descriptive interview question
focused strictly on the subject: "{subject}".

{prev_block}

Rules:
- One clear, unique question.
- It must NOT be the same as or a trivial rephrasing of any previous question listed above.
- No numbering, bullets, or explanations.
- Suitable for spoken or written answers.
Return ONLY the question text.
""".strip()

    question = call_llm(prompt)
    return jsonify({"question": question})


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json() or {}
    role = data.get("role", "")
    subject = data.get("subject", "")
    question = data.get("question", "")
    answer = data.get("answer", "")

    feedback_prompt = f"""
You are an expert interviewer for a {role} role.

Interview Subject: {subject}

Evaluate the following answer.

Question: {question}
Candidate Answer: {answer}

Respond in STRICT JSON with exactly these fields:
- score (1 to 10 integer)
- strengths (1–3 sentences)
- improvements (1–3 sentences)
- model_answer (3–8 sentences)

Return ONLY JSON. No extra words.
""".strip()

    raw = call_llm(feedback_prompt)

    try:
        data = try_parse_json(raw)
    except Exception:
        data = {
            "score": 6,
            "strengths": "Answer shows partial understanding of the topic.",
            "improvements": "Improve structure, depth, and include examples.",
            "model_answer": raw[:800]
        }

    return jsonify(data)


# ================== MCQ ==================

@app.route("/api/generate-mcq", methods=["POST"])
def generate_mcq():
    data = request.get_json() or {}
    role = data.get("role", "Software Quality Analyst")
    difficulty = data.get("difficulty", "easy")
    subject = data.get("subject", "Software Testing - Manual Basics")
    previous_questions = data.get("previous_questions", [])  # from frontend

    # Hard-ban super common textbook MCQs you don't want
    banned_questions = [
        "what is the primary goal of software testing",
        "what is the main goal of software testing",
        "purpose of software testing",
        "main aim of software testing",
    ]

    def is_banned(q: str) -> bool:
        ql = q.lower()
        return any(b in ql for b in banned_questions)

    # We will try multiple times to get a good, non-repetitive MCQ
    mcq = None

    for attempt in range(6):
        prev_block = ""
        if previous_questions:
            joined = "\n".join(f"- {q}" for q in previous_questions)
            prev_block = f"""
Previously asked MCQ questions in this session (do NOT repeat or trivially rephrase):

{joined}

""".strip()

        prompt = f"""
You are preparing INTERVIEW-LEVEL MCQ questions for a {role} role.

Subject: "{subject}"
Difficulty: {difficulty}

{prev_block}

IMPORTANT MCQ DESIGN RULES:
- The question must be NEW and clearly different from the previous ones above.
- Do NOT ask simple definition-only questions.
- Do NOT ask about the "primary goal" / "purpose" / "main aim" of software testing.
- Prefer scenario-based or application-based questions.
- Use real-world situations, decisions, mistakes, or edge cases.
- Exactly 4 options: A, B, C, D.
- Only ONE option must be correct.
- Add a short explanation for why the correct option is right (and optionally why others are wrong).

Return STRICT JSON only:

{{
  "question": "question text",
  "options": {{
    "A": "option A",
    "B": "option B",
    "C": "option C",
    "D": "option D"
  }},
  "correct_answer": "A",
  "explanation": "short explanation"
}}
""".strip()

        raw = call_llm(prompt)

        try:
            candidate = try_parse_json(raw)
        except Exception:
            continue

        q_text = (candidate.get("question") or "").strip()
        if not q_text:
            continue

        # Hard-ban certain patterns
        if is_banned(q_text):
            continue

        # Avoid near-duplicates vs previous questions
        if any(is_similar(q_text, prev) for prev in previous_questions):
            continue

        mcq = candidate
        break

    # Fallback if everything failed – but use a DIFFERENT, safe MCQ (not goal-of-testing)
    if mcq is None:
        mcq = {
            "question": "Which of the following is an example of non-functional testing?",
            "options": {
                "A": "Unit testing",
                "B": "Integration testing",
                "C": "Performance testing",
                "D": "Smoke testing"
            },
            "correct_answer": "C",
            "explanation": "Performance testing checks how the system behaves under load, which is non-functional behaviour."
        }

    return jsonify(mcq)


# ================== Run ==================

if __name__ == "__main__":
    app.run(debug=True)
