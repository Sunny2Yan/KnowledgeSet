SINGLE_MESSAGE_TEMPLATE = "name: {name}\nrole: {role}\nspeak content: {content}\n"

SUMMARY_PROMPT_TEMPLATE = """Summarize the lines of conversation provided.

EXAMPLE
Conversation:
role: Human
speak content: What do you think of artificial intelligence?

role: AI
speak content: I think artificial intelligence is a force for good.

role: Human
speak content: Why do you think artificial intelligence is a force for good?

role: AI
speak content: Because artificial intelligence will help humans reach their full potential.

Summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Conversation:
{conversation}

Summary:"""

