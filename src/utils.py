PROMPT_TEMPLATE = """
    "<s> You are a helpful assistant.[INST] {instruction}  [/INST]"
"""

def extract_anthropic_prompt(text):
    # extracting prompt and response from HH RLHF
    search_term = "\n\nAssistant: "
    search_term_idx = text.rfind(search_term)
    assert search_term_idx != -1, "Search term not found"
    
    # return (prompt, response)
    return (text[:search_term_idx + len(search_term)],
            text[search_term_idx + len(search_term):])
    