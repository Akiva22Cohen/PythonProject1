class AdvancedChatMemory:
    """
    Memory class to store, summarize, and manage conversation context.
    """

    def __init__(self, recent_messages=None, max_recent_messages=5):
        """
        Initialize the memory with optional recent messages and a limit.
        """
        self.summary = "Conversation summary: "  # Store older context as a summary
        self.recent_messages = recent_messages if recent_messages else []
        self.max_recent_messages = max_recent_messages  # Limit for recent messages

    def add_message(self, role: str, content: str):
        """
        Add a message to memory. Update the summary if recent messages exceed the limit.
        """
        self.recent_messages.append({"role": role, "content": content})

        # If the number of recent messages exceeds the limit, summarize older ones
        if len(self.recent_messages) > self.max_recent_messages:
            # Example of simple summarization
            summarized_content = " ".join(
                [msg["content"] for msg in self.recent_messages[:-self.max_recent_messages]]
            )
            self.summary += summarized_content + " "
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]  # Keep only recent ones

    def get_prompt_context(self) -> list[dict[str, str]]:
        """
        Combine the summary with recent messages to create the LLM context.
        """
        context = []
        if self.summary.strip():
            context.append({"role": "system", "content": self.summary.strip()})
        context.extend(self.recent_messages)
        return context
