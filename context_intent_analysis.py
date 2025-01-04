import openai


def analyze_context_and_intent(
        recent_messages: list[dict[str, str]],
        model_name: str = "gpt-3.5-turbo"
) -> str:
    """
    Use a multi-shot or few-shot prompting strategy to analyze the user's
    intent or the context of the conversation.

    Parameters
    ----------
    recent_messages : list of dict
        The last few messages (user or assistant).
    model_name : str
        The LLM to call.

    Returns
    -------
    str
        A textual analysis or classification result.
    """
    # Define few-shot examples
    examples = [
        # דוגמה על רגשות ותמיכה
        {"role": "user", "content": "I feel overwhelmed with everything going on at work and home."},
        {"role": "assistant",
         "content": "Analysis: User is expressing feelings of overwhelm and is seeking emotional support."},

        # דוגמה על שינוי קריירה
        {"role": "user", "content": "I want to switch careers but don’t know where to start."},
        {"role": "assistant", "content": "Analysis: User is seeking guidance on starting a career transition."},

        # דוגמה על שיפור מערכות יחסים
        {"role": "user", "content": "Lisa and I argue a lot about household responsibilities."},
        {"role": "assistant",
         "content": "Analysis: User is seeking support for improving communication and managing household responsibilities."},

        # דוגמה על התפתחות אישית
        {"role": "user", "content": "I want to set a better example for my kids and be more present at home."},
        {"role": "assistant", "content": "Analysis: User is seeking personal growth and family-oriented improvements."},

        # דוגמה על מיקוד בתוכנית פעולה
        {"role": "user", "content": "How can I create a proposal for a cybersecurity project at work?"},
        {"role": "assistant",
         "content": "Analysis: User is seeking practical advice for preparing a professional project proposal."},

        # דוגמה על ניהול זמן ואיזון
        {"role": "user", "content": "I don’t have time for hobbies or relaxing after work."},
        {"role": "assistant",
         "content": "Analysis: User is seeking strategies for better time management and stress relief."},

        # דוגמה על שיפור תקשורת
        {"role": "user", "content": "I want to improve how I communicate with my team and family."},
        {"role": "assistant",
         "content": "Analysis: User is seeking to enhance interpersonal and communication skills."},
    ]

    # Create the prompt context
    prompt = examples + recent_messages
    prompt.append({"role": "assistant", "content": "Analysis:"})

    # Call the LLM
    response = openai.chat.completions.create(
        model=model_name,
        messages=prompt
    )

    # Extract the analysis from the LLM response
    analysis = response.choices[0].message.content.strip()
    return analysis
