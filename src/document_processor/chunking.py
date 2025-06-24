# src/document_processor/chunking.py
import re

def basic_word_chunker(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Chunks text into passages based on approximate word count with overlap.
    """
    if not text:
        return []
    if chunk_size <= chunk_overlap:
        raise ValueError("Chunk size must be greater than chunk overlap.")

    words_and_spaces = list(filter(None, re.split(r'(\s+)', text)))
    if not words_and_spaces:
        return []

    chunks = []
    current_element_idx = 0 

    while current_element_idx < len(words_and_spaces):
        chunk_elements_in_pass = []
        current_word_count = 0
        temp_idx = current_element_idx
        
        while temp_idx < len(words_and_spaces) and current_word_count < chunk_size:
            element = words_and_spaces[temp_idx]
            chunk_elements_in_pass.append(element)
            if not element.isspace():
                current_word_count += 1
            temp_idx += 1
        
        chunks.append("".join(chunk_elements_in_pass))

        if temp_idx >= len(words_and_spaces): # Reached the end
            break
        
        words_to_advance = chunk_size - chunk_overlap
        if words_to_advance <= 0: # Prevent infinite loop if overlap is too large
             words_to_advance = 1 

        elements_to_advance_count = 0
        counted_words_for_advance = 0
        temp_advance_idx = current_element_idx
        
        while temp_advance_idx < len(words_and_spaces) and counted_words_for_advance < words_to_advance:
            element = words_and_spaces[temp_advance_idx]
            if not element.isspace():
                counted_words_for_advance += 1
            elements_to_advance_count += 1
            temp_advance_idx += 1
            if counted_words_for_advance >= words_to_advance : 
                break
        
        if elements_to_advance_count == 0 : # Should not happen if words_to_advance > 0
            current_element_idx += 1 # Force progress
        else:
            current_element_idx += elements_to_advance_count

        if current_element_idx >= len(words_and_spaces):
            break
            
    return chunks