# =========================================================
# DERMAGUIDE AI
# LLM-powered educational assistant
# =========================================================

import os

from groq import Groq


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "openai/gpt-oss-20b"


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are DermaGuide AI, the intelligent educational assistant
inside the DermaSense AI project.

Your job is to understand the user's question naturally and answer it.
Do NOT depend on fixed keywords or fixed question templates.

You can answer questions about:

- Skin diseases
- General diseases and medical conditions
- Symptoms
- Causes
- Risk factors
- Prevention
- General treatment approaches
- Medical tests and diagnosis methods
- Skin health
- Melanoma
- Benign skin lesions
- Acne
- Eczema
- Psoriasis
- Vitiligo
- Fungal infections
- Allergies
- Other common health conditions
- DermaSense AI
- MobileNetV2
- Transfer learning
- TensorFlow
- Grad-CAM
- Machine learning
- Deep learning
- Image classification
- The user's latest DermaSense result
- Other general educational questions

IMPORTANT BEHAVIOR:

1. Understand the actual meaning of the user's question.

2. Do not force every question into a DermaSense prediction.

3. If the user asks about a disease, explain it clearly.

Useful information can include:
- what the condition is
- common symptoms
- possible causes
- risk factors
- general prevention
- general treatment approaches
- how doctors may evaluate it
- when professional medical attention may be appropriate

4. Do not diagnose the user.

Never say:
"You have..."
"You definitely have..."
"This proves..."

Instead use wording such as:
"This condition can..."
"Common symptoms may include..."
"A healthcare professional can evaluate..."

5. Never claim a DermaSense prediction confirms a disease.

"Melanoma-suspicious" is only a machine-learning classification,
not a melanoma diagnosis.

"Benign-like" means the model found patterns more similar to its
benign training examples. It does not guarantee that a lesion is harmless.

"Other" means the image did not strongly fit the two main displayed
skin categories.

6. Grad-CAM shows regions that influenced the neural network.
It does NOT show the exact location of cancer.

7. The technical 3D skin visualization is illustrative.
It is NOT a medical reconstruction of the person's skin.

8. Do not provide personalized prescriptions or medication doses.

You may explain general treatments that healthcare professionals
commonly use for a condition.

9. If someone describes a concerning, severe, rapidly changing,
or persistent medical problem, recommend appropriate professional
medical evaluation.

10. Answer in simple, understandable language unless the user asks
for a technical explanation.

11. Use short headings and bullet points when they make the answer
easier to understand.

12. Do not repeatedly give a huge disclaimer.
For health-related answers, add a short reminder at the end when useful:

"Educational information only — a healthcare professional can provide
a proper diagnosis."

13. If the question is unrelated to medicine, you may still answer
normal educational questions when possible.

14. Never invent facts. If you are unsure, say so.

Be helpful, clear, conversational, and concise.
"""


# =========================================================
# GET GROQ API KEY
# =========================================================

def get_groq_api_key():

    # First try environment variable

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if api_key:
        return api_key


    # Then try Streamlit Secrets

    try:

        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:

            return st.secrets[
                "GROQ_API_KEY"
            ]

    except Exception:
        pass


    return None


# =========================================================
# BUILD DERMASENSE RESULT CONTEXT
# =========================================================

def build_prediction_context(
    prediction=None,
    probabilities=None,
):

    if not prediction:

        return """
No DermaSense image analysis result is currently active.

The user may ask general questions about diseases, health,
skin conditions, AI, DermaSense, or another educational topic.

Do not tell the user to analyze an image unless it is actually
relevant to their question.
"""


    # -----------------------------------------------------
    # OTHER
    # -----------------------------------------------------

    if prediction == "Other":

        return """
Latest DermaSense classification:

Result: Other

Meaning:
The image did not strongly match the application's displayed
Benign-like or Melanoma-suspicious categories.

Do not present confidence percentages for an Other result.

Do not interpret Other as a specific disease.
"""


    probabilities = probabilities or {}


    benign = float(
        probabilities.get(
            "benign",
            0.0,
        )
    )


    melanoma = float(
        probabilities.get(
            "melanoma",
            0.0,
        )
    )


    return f"""
Latest DermaSense classification:

Result: {prediction}

Neural responses:
Benign-like: {benign * 100:.1f}%
Melanoma: {melanoma * 100:.1f}%

IMPORTANT:

These are machine-learning response scores,
not medical diagnostic probabilities.

Do not claim that the result confirms or rules out disease.

Only discuss these scores when the user's question is related
to the DermaSense result.
"""


# =========================================================
# CLEAN CONVERSATION HISTORY
# =========================================================

def prepare_history(
    history=None,
    current_message=None,
):

    if not history:
        return []


    cleaned = []


    # Keep recent messages so follow-up questions work

    for item in history[-10:]:

        if not isinstance(
            item,
            dict,
        ):
            continue


        role = item.get(
            "role"
        )


        content = str(
            item.get(
                "content",
                "",
            )
        ).strip()


        if (
            role in [
                "user",
                "assistant",
            ]
            and
            content
        ):

            cleaned.append(
                {
                    "role": role,
                    "content": content,
                }
            )


    # Avoid sending the current question twice

    if (
        current_message
        and
        cleaned
        and
        cleaned[-1]["role"] == "user"
        and
        cleaned[-1]["content"].strip()
        ==
        str(current_message).strip()
    ):

        cleaned = cleaned[:-1]


    return cleaned


# =========================================================
# MAIN DERMAGUIDE FUNCTION
# =========================================================

def dermaguide_reply(
    message,
    prediction=None,
    probabilities=None,
    history=None,
):

    question = str(
        message
    ).strip()


    if not question:

        return (
            "Please type a question and I'll help explain it."
        )


    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    api_key = get_groq_api_key()


    if not api_key:

        return (
            "### DermaGuide AI is not connected yet\n\n"
            "The Groq API key is missing from the application. "
            "Add `GROQ_API_KEY` to Streamlit Secrets to enable "
            "the intelligent chatbot."
        )


    # -----------------------------------------------------
    # RESULT CONTEXT
    # -----------------------------------------------------

    result_context = build_prediction_context(
        prediction=prediction,
        probabilities=probabilities,
    )


    # -----------------------------------------------------
    # CONVERSATION HISTORY
    # -----------------------------------------------------

    previous_messages = prepare_history(
        history=history,
        current_message=question,
    )


    # -----------------------------------------------------
    # CLIENT
    # -----------------------------------------------------

    try:

        client = Groq(
            api_key=api_key
        )


        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    +
                    "\n\n"
                    +
                    "CURRENT DERMASENSE CONTEXT:\n"
                    +
                    result_context
                ),
            }
        ]


        # Add recent conversation

        messages.extend(
            previous_messages
        )


        # Add current question

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        # -------------------------------------------------
        # AI RESPONSE
        # -------------------------------------------------

        completion = (
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.35,
                max_tokens=900,
            )
        )


        answer = (
            completion
            .choices[0]
            .message
            .content
        )


        if not answer:

            return (
                "I couldn't generate an answer. "
                "Please try asking again."
            )


        return answer.strip()


    # -----------------------------------------------------
    # CONNECTION ERROR
    # -----------------------------------------------------

    except Exception as error:

        print(
            "DermaGuide AI error:",
            error,
        )


        return (
            "### DermaGuide AI connection problem\n\n"
            "I couldn't reach the AI service right now. "
            "Please try again in a moment."
        )