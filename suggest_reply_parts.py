import openai
from AdvancedChatMemory import AdvancedChatMemory
from LifeCoachingBookRetriever import LifeCoachingBookRetriever
from context_intent_analysis import analyze_context_and_intent


def suggest_reply_part1(
        conversation_history,
        model_name: str = "gpt-3.5-turbo"
):
    response = openai.chat.completions.create(
        model=model_name,
        messages=conversation_history
    )
    return response


def suggest_reply_part2(
        memory: AdvancedChatMemory,
        model_name: str = "gpt-3.5-turbo"
) -> str:
    """
    Suggest the next reply using advanced memory.

    Parameters
    ----------
    memory : AdvancedChatMemory
        Manages storing/summarizing conversation.
    model_name : str
        The LLM to call.

    Returns
    -------
    str
        The suggested assistant reply.
    """
    # 1. Retrieve context from memory
    context = memory.get_prompt_context()

    # 2. Make the LLM API call
    response = openai.chat.completions.create(
        model=model_name,
        messages=context
    )

    # 3. Get the LLM's reply
    reply = response.choices[0].message.content

    # 4. Add the reply to memory
    memory.add_message(role="assistant", content=reply)

    # 5. Return the reply
    return reply


def suggest_reply_part4(
        memory: AdvancedChatMemory,
        book_retriever: LifeCoachingBookRetriever,
        model_name: str = "gpt-3.5-turbo"
) -> str:
    """
    Suggest a reply that combines:
      - Advanced memory from Part 2
      - (Optional) analysis from Part 3 (if you want)
      - External knowledge from a life-coaching book stored in Chroma

    Steps:
      1. Get conversation context from memory.
      2. (Optional) call analyze_context_and_intent(...) to shape the tone/content.
      3. Retrieve relevant chunks from the life-coaching book.
      4. Combine memory context + relevant chunks (and possibly the analysis).
      5. Call the LLM, store the result in memory, return it.
    """
    # 1. Get the conversation context
    context = memory.get_prompt_context()

    # 2. Optionally analyze last user message
    user_message = context[-1]["content"] if context else ""
    intent_analysis = analyze_context_and_intent(context[-5:]) if user_message else ""

    # 3. Retrieve relevant chunks from the book
    relevant_chunks = book_retriever.retrieve_relevant_chunks(user_message)

    # 4. Construct final LLM prompt with memory + relevant chunks
    prompt = context + [
        {"role": "system", "content": "Relevant knowledge from the book:"},
        {"role": "system", "content": "\n\n".join(relevant_chunks)}
    ]

    # 5. Make LLM call
    response = openai.chat.completions.create(
        model=model_name,
        messages=prompt
    )

    # 6. Add the reply to memory
    reply = response.choices[0].message.content
    memory.add_message(role="assistant", content=reply)

    return reply
