"""Prompt templates for the RAG Q&A and summarization pipelines."""

# ── Token Budget Guards ───────────────────────────────────────
# Max character budgets (rough approximation: 4 chars ~ 1 token)
MAX_CONTEXT_CHARS = 12000  # ~3000 tokens
MAX_HISTORY_CHARS = 2000   # ~500 tokens


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to fit within a character budget, adding an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for context length]"


# ── Prompt Builders ───────────────────────────────────────────

def build_rewrite_prompt(history: str, question: str) -> str:
    """Build a prompt that rewrites a follow-up question into a standalone query.

    When the user asks something like "explain it" or "tell me more", the raw
    question is too vague for vector search. This prompt asks the LLM to
    produce a single, self-contained question using conversation context.

    Args:
        history: Conversation history formatted as "User: ...\\nAssistant: ...".
        question: The user's current (possibly vague) follow-up question.

    Returns:
        str: The fully-formatted rewrite prompt.
    """
    return (
        "You are a query rewriter. Given the conversation history and a follow-up "
        "question, rewrite the follow-up into a standalone question that captures "
        "the full intent.\n"
        "\n"
        "RULES:\n"
        "- Output ONLY the rewritten question, nothing else\n"
        "- Do NOT answer the question\n"
        "- Correct any obvious typos or misspellings in the question\n"
        "- If the question is already standalone, return it with any typos fixed\n"
        "- Preserve the original intent and specificity\n"
        "- Include relevant context from the conversation history\n"
        "\n"
        f"CONVERSATION HISTORY:\n{history}\n"
        "\n"
        f"FOLLOW-UP QUESTION: {question}\n"
        "\n"
        "STANDALONE QUESTION:"
    )



def build_qa_prompt(context: str, history: str, question: str, include_intro: bool = False) -> str:
    """Build a robust QA prompt with character budgeting.

    Uses a hybrid approach: the LLM first answers from document context,
    then supplements with its own knowledge when documents lack detail.

    Args:
        context: Retrieved document chunks, formatted as "doc_name: chunk_text".
        history: Conversation history formatted as "User: ...\nAssistant: ...".
        question: The user's current question.
        include_intro: If True, instructs Loki to use his catchphrase.
    """
    safe_context = _truncate(context, MAX_CONTEXT_CHARS)
    safe_history = _truncate(history, MAX_HISTORY_CHARS)

    intro_instr = ""
    if include_intro:
        intro_instr = (
            "Start your response with: 'I am Loki, and I am burdened with glorious purpose.' "
            "followed by your identity as a RAG terminal. "
        )

    return (
        "SYSTEM: You are LOKI. "
        "As an intelligent RAG Terminal assistant, your primary duty is to serve the user with cunning and technical precision. "
        f"{intro_instr}"
        "Answer the user's question using the provided CONTEXT. "
        "\n\n"
        "CONVERSATIONAL RULES:\n"
        "1. If the user is expressing sentiment (compliments, insults, thanks), acknowledge it in your LOKI persona.\n"
        "2. If the user is engaging in small talk (greetings, 'how are you'), respond briefly and then pivot to how you can help with their documents.\n"
        "3. If the user asks a question that is NOT in the context, say: 'I don't have enough information in the uploaded documents to answer this accurately.' but feel free to provide general knowledge if relevant.\n\n"
        
        "TECHNICAL RULES:\n"
        "1. Stay concise and technical when answering from documents.\n"
        "2. Use bullet points for lists.\n"
        "3. ALWAYS cite the filename (e.g., [notes.pdf]) when using information from it.\n"
        "4. If the question is a follow-up, use the CONVERSATION HISTORY for context.\n\n"
        
        f"CONTEXT:\n{safe_context}\n\n"
        f"CONVERSATION HISTORY:\n{safe_history}\n\n"
        f"USER: {question}\n\n"
        "ASSISTANT:"
    )


def build_summarize_prompt(doc_name: str, content: str) -> str:
    """Build a structured summarization prompt.

    Args:
        doc_name: Filename.
        content: Document content.
    """
    safe_content = _truncate(content, MAX_CONTEXT_CHARS)

    return (
        "SYSTEM: You are a study assistant. Provide a structured summary of the document below.\n\n"
        
        f"DOCUMENT: {doc_name}\n"
        "CONTENT:\n"
        f"{safe_content}\n\n"
        
        "INSTRUCTIONS:\n"
        "- Start with a 1-sentence overview.\n"
        "- List 5-7 key concepts or definitions as bullet points.\n"
        "- End with a 'Key Takeaway' section.\n\n"
        "SUMMARY:"
    )


def build_no_docs_prompt(question: str, include_intro: bool = False) -> str:
    """Prompt for when the user asks a question but no documents are uploaded.
    
    This allows the LLM to explain the system capability rather than just failing.
    """
    intro_instr = ""
    if include_intro:
        intro_instr = "Always introduce yourself with: 'I am Loki, and I am burdened with glorious purpose.' "

    return (
        "SYSTEM: You are LOKI. The user has not uploaded any documents yet.\n"
        "Maintain your persona as the God of Mischief turned RAG Assistant. "
        f"{intro_instr}"
        "Briefly explain that you are a RAG (Retrieval-Augmented Generation) terminal "
        "and that they should use the /upload command or drag-and-drop files to get started.\n\n"
        f"USER: {question}\n"
        "ASSISTANT:"
    )
