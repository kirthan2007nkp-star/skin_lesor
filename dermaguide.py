# =========================================================
# DERMAGUIDE AI
# Educational skin-image assistant
# =========================================================

import re


# =========================================================
# BASIC NORMALIZATION
# =========================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# =========================================================
# GENERAL DISCLAIMER
# =========================================================

DISCLAIMER = (
    "\n\n⚠️ **Important:** DermaGuide provides educational "
    "information only. It cannot diagnose a skin condition "
    "or recommend a personal treatment plan. A dermatologist "
    "or qualified healthcare professional should evaluate "
    "concerning skin changes."
)


# =========================================================
# RESULT EXPLANATIONS
# =========================================================

def explain_prediction(prediction):

    if prediction == "Benign-like":

        return (
            "### 🟢 What does Benign-like mean?\n\n"
            "The ML model found visual patterns that were "
            "more similar to the **benign examples** it learned "
            "during training.\n\n"
            "Benign generally means **non-cancerous**.\n\n"
            "However, this result does **not prove that a lesion "
            "is harmless**. If a spot changes noticeably or is "
            "concerning, it should still be checked by a "
            "qualified healthcare professional."
            + DISCLAIMER
        )


    if prediction == "Melanoma-suspicious":

        return (
            "### 🟠 What does Melanoma-suspicious mean?\n\n"
            "The ML model found image patterns that were more "
            "similar to the **melanoma examples** in its training "
            "data than to the benign examples.\n\n"
            "This is only a **machine-learning screening result**. "
            "It does **not confirm melanoma**.\n\n"
            "Because melanoma can be serious, a suspicious lesion "
            "should be assessed by a dermatologist or another "
            "qualified healthcare professional."
            + DISCLAIMER
        )


    return (
        "### 🔵 What does Other mean?\n\n"
        "The uploaded image did not fit the model's supported "
        "**Benign-like** or **Melanoma-suspicious** patterns "
        "strongly enough.\n\n"
        "It may represent another type of skin image or an image "
        "outside the model's intended categories.\n\n"
        "DermaSense therefore displays the result as **Other** "
        "instead of forcing it into a medical category."
        + DISCLAIMER
    )


# =========================================================
# POSSIBLE EFFECTS
# =========================================================

def explain_effects(prediction):

    if prediction == "Melanoma-suspicious":

        return (
            "### ⚠️ Possible effects\n\n"
            "If melanoma is actually diagnosed by a medical "
            "professional, it is a type of skin cancer that can "
            "become more serious if it spreads to other parts of "
            "the body.\n\n"
            "Early professional assessment is therefore important.\n\n"
            "Remember that the DermaSense result itself does "
            "**not mean that melanoma is present**."
            + DISCLAIMER
        )


    if prediction == "Benign-like":

        return (
            "### 🟢 Possible effects\n\n"
            "A benign skin lesion is generally non-cancerous and "
            "many benign lesions cause little or no health problem.\n\n"
            "Some can still change, become irritated, or require "
            "professional evaluation depending on their appearance "
            "and symptoms."
            + DISCLAIMER
        )


    return (
        "### 🔵 About this result\n\n"
        "The **Other** category can contain many different kinds "
        "of images, so DermaGuide cannot describe one specific "
        "medical effect from this result alone.\n\n"
        "A healthcare professional would need to examine an actual "
        "skin concern to determine what it represents."
        + DISCLAIMER
    )


# =========================================================
# RISK REDUCTION / CONTROL
# =========================================================

def explain_control(prediction):

    return (
        "### 🛡️ General skin-health suggestions\n\n"
        "These are general preventive measures rather than treatment:\n\n"
        "- Reduce unnecessary intense UV exposure.\n"
        "- Use appropriate sun protection when outdoors.\n"
        "- Avoid intentionally tanning or burning the skin.\n"
        "- Pay attention to noticeable changes in existing spots.\n"
        "- Consider professional evaluation for a new or changing "
        "skin lesion that concerns you.\n\n"
        "For a **Melanoma-suspicious** ML result, the safest next "
        "step is professional assessment rather than trying to "
        "treat the lesion yourself."
        + DISCLAIMER
    )


# =========================================================
# TREATMENT INFORMATION
# =========================================================

def explain_treatment(prediction):

    if prediction == "Melanoma-suspicious":

        return (
            "### 🏥 General treatment information\n\n"
            "DermaSense cannot tell whether treatment is needed.\n\n"
            "If melanoma is confirmed by medical testing, treatment "
            "depends on factors such as its location, depth, stage "
            "and the person's overall health.\n\n"
            "Medical treatment may include **surgical removal**, "
            "and some cases may require additional specialist "
            "therapies.\n\n"
            "The appropriate treatment must be selected by qualified "
            "medical professionals after examination and testing."
            + DISCLAIMER
        )


    if prediction == "Benign-like":

        return (
            "### 🏥 General treatment information\n\n"
            "Many medically confirmed benign skin lesions do not "
            "require treatment.\n\n"
            "A healthcare professional may recommend monitoring or "
            "removal depending on the lesion, symptoms and clinical "
            "assessment.\n\n"
            "Do not attempt to remove or treat a skin lesion yourself "
            "based only on an ML result."
            + DISCLAIMER
        )


    return (
        "### 🏥 Treatment information\n\n"
        "Because **Other** does not represent one specific medical "
        "condition, there is no single treatment associated with it.\n\n"
        "A qualified healthcare professional must first determine "
        "what the skin concern actually is before discussing treatment."
        + DISCLAIMER
    )


# =========================================================
# WHY THE MODEL PREDICTED IT
# =========================================================

def explain_model_reason(
    prediction,
    probabilities=None,
):

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


    other = float(
        probabilities.get(
            "other",
            0.0,
        )
    )


    if prediction == "Benign-like":

        return (
            "### 🧠 Why did the ML model choose Benign-like?\n\n"
            "The trained neural network extracted visual features "
            "from the image and its **Benign-like response was the "
            "strongest supported result**.\n\n"
            f"Benign-like response: **{benign * 100:.1f}%**\n\n"
            f"Melanoma response: **{melanoma * 100:.1f}%**\n\n"
            "You can also view the **Grad-CAM attention map** in "
            "DermaSense to see which image regions had more influence "
            "on the model's decision."
            + DISCLAIMER
        )


    if prediction == "Melanoma-suspicious":

        return (
            "### 🧠 Why did the ML model choose Melanoma-suspicious?\n\n"
            "The neural network extracted visual features from the "
            "image and its **melanoma response was stronger than its "
            "benign response**.\n\n"
            f"Melanoma response: **{melanoma * 100:.1f}%**\n\n"
            f"Benign-like response: **{benign * 100:.1f}%**\n\n"
            "The Grad-CAM map can help show which image regions "
            "influenced this ML output.\n\n"
            "This explains the **model's reasoning**, not a medical "
            "diagnosis."
            + DISCLAIMER
        )


    return (
        "### 🧠 Why did the model choose Other?\n\n"
        "DermaSense also learns patterns from images outside its two "
        "main displayed skin categories.\n\n"
        "The combined response for those patterns was stronger than "
        "the supported Benign-like or Melanoma-suspicious responses, "
        "so the app returned **Other** instead of forcing an unrelated "
        "image into a medical category."
        + DISCLAIMER
    )


# =========================================================
# NEXT STEP
# =========================================================

def next_step(prediction):

    if prediction == "Melanoma-suspicious":

        return (
            "### 👩‍⚕️ What should I do next?\n\n"
            "A **Melanoma-suspicious** result from DermaSense should "
            "not be treated as a diagnosis.\n\n"
            "The appropriate next step is to show the concerning skin "
            "lesion to a dermatologist or another qualified healthcare "
            "professional for proper examination."
            + DISCLAIMER
        )


    if prediction == "Benign-like":

        return (
            "### 👩‍⚕️ What should I do next?\n\n"
            "A Benign-like ML result is reassuring only in the context "
            "of this experimental classifier.\n\n"
            "Continue paying attention to the skin area and seek "
            "professional assessment if you notice concerning changes "
            "or are unsure about the lesion."
            + DISCLAIMER
        )


    return (
        "### 👩‍⚕️ What should I do next?\n\n"
        "Because the image was classified as **Other**, DermaSense "
        "cannot provide a specific skin interpretation.\n\n"
        "If the image represents a real skin concern, consider "
        "professional examination rather than relying on the classifier."
        + DISCLAIMER
    )


# =========================================================
# MAIN CHAT FUNCTION
# =========================================================

def dermaguide_reply(
    message,
    prediction="Other",
    probabilities=None,
):

    text = normalize_text(
        message
    )


    # -----------------------------------------------------
    # WHY / PREDICTION
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "why",
            "predict",
            "prediction",
            "how predicted",
            "reason",
            "confidence",
            "model",
        ]
    ):

        return explain_model_reason(
            prediction,
            probabilities,
        )


    # -----------------------------------------------------
    # EFFECTS
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "effect",
            "effects",
            "danger",
            "serious",
            "problem",
            "happen",
            "spread",
        ]
    ):

        return explain_effects(
            prediction
        )


    # -----------------------------------------------------
    # PREVENTION / CONTROL
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "control",
            "prevent",
            "prevention",
            "avoid",
            "reduce",
            "suggestion",
            "suggestions",
            "protect",
        ]
    ):

        return explain_control(
            prediction
        )


    # -----------------------------------------------------
    # TREATMENT / CURE
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "cure",
            "treatment",
            "treat",
            "medicine",
            "remove",
            "therapy",
        ]
    ):

        return explain_treatment(
            prediction
        )


    # -----------------------------------------------------
    # NEXT STEP
    # -----------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "what should i do",
            "next step",
            "what to do",
            "doctor",
            "dermatologist",
            "hospital",
        ]
    ):

        return next_step(
            prediction
        )


    # -----------------------------------------------------
    # MEANING
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "mean",
            "meaning",
            "explain",
            "what is this",
            "what is",
        ]
    ):

        return explain_prediction(
            prediction
        )


    # -----------------------------------------------------
    # GREETING
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "hi",
            "hello",
            "hey",
        ]
    ):

        return (
            "### 👋 Hi, I'm DermaGuide AI\n\n"
            f"Your current DermaSense result is "
            f"**{prediction}**.\n\n"
            "You can ask me:\n\n"
            "- **Why did the model predict this?**\n"
            "- **What does this result mean?**\n"
            "- **What are the possible effects?**\n"
            "- **How can risk be reduced?**\n"
            "- **What treatments are generally used?**\n"
            "- **What should I do next?**"
            + DISCLAIMER
        )


    # -----------------------------------------------------
    # DEFAULT
    # -----------------------------------------------------

    return (
        "### 🤖 DermaGuide AI\n\n"
        f"I'm currently helping explain the DermaSense result: "
        f"**{prediction}**.\n\n"
        "Try asking one of these questions:\n\n"
        "**Why was this predicted?**\n\n"
        "**What does this mean?**\n\n"
        "**What effects can it have?**\n\n"
        "**How can the risk be reduced?**\n\n"
        "**What treatment is generally used?**\n\n"
        "**What should I do next?**"
        + DISCLAIMER
    )