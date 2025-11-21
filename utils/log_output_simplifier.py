"""
Log Output Simplifier Module

This module provides functions to clean and format LLM log output.
Originally adapted from log-output-simplifier/main.py, now integrated into this repository.
"""

import json
import re


def clean_html(text_block: str) -> str:
    """
    Replaces HTML links with a 'Text (URL)' format and removes other HTML tags.

    Args:
        text_block: Input text containing HTML tags and links.

    Returns:
        Cleaned text with HTML links converted to 'Text (URL)' format and
        all other HTML tags removed. Whitespace is normalized.
    """
    if not text_block:
        return ""

    # Pattern to find <a href="...">...</a> and capture the URL and the link text.
    link_pattern = re.compile(r'<a.*?href="([^"]*)".*?>(.*?)</a>', re.DOTALL)
    cleaned_text = link_pattern.sub(r"\2 (\1)", text_block)

    # Pattern to find and remove any other leftover HTML tags.
    tag_pattern = re.compile(r"<[^>]+>")
    cleaned_text = tag_pattern.sub("", cleaned_text)

    # Normalize whitespace for better readability.
    cleaned_text = re.sub(r"[\r\n]{2,}", "\n\n", cleaned_text).strip()

    return cleaned_text


def clean_and_format_llm_log(messy_text: str) -> str:
    """
    Parses, cleans, and formats a messy LLM log string, ensuring no sections are lost.

    This function processes LLM log output and formats it into a structured markdown format
    with clear section markers. It extracts various sections like Task, Thought Process,
    Action, Observation, Final Answer, etc.

    Args:
        messy_text: The raw JSON string from the LLM log file or the answer field content.

    Returns:
        A clean, formatted string with all available sections included.
    """
    try:
        # Try to parse as JSON first
        log_data = json.loads(messy_text)
        # The main content is usually inside the "answer" key.
        answer_content = log_data.get("answer", messy_text)
        # Normalize escaped newlines to actual newlines.
        answer_content = answer_content.replace("\\n", "\n")
    except json.JSONDecodeError:
        # If not JSON, treat as plain text
        answer_content = messy_text.replace("\\n", "\n")
    except (ValueError, TypeError) as e:
        # Handle value/type errors during JSON parsing
        return f"Error parsing JSON: {e}"
    except Exception as e:
        # 予期しないエラーの場合は詳細な情報を記録
        return (
            f"An unexpected error occurred during JSON parsing: {type(e).__name__}: {e}"
        )

    # Define all sections we want to extract from the log.
    sections_to_find = [
        {"title": "## 📝 Task タスク", "marker": "タスク："},
        {"title": "## 💬 Reaction 反応", "marker": "反応："},
        {"title": "## 📂 Classification 分類", "marker": "分類："},
        {"title": "## 📊 Status 状態", "marker": "状態："},
        {"title": "## 🤖 LLM Thought Process 思考", "marker": "思考："},
        {"title": "## ⚡ Action 行動", "marker": "行動："},
        {"title": "## ⌨️ Action Input 行動入力", "marker": "行動入力："},
        {"title": "## 📚 Raw Search Results (Cleaned) 観察", "marker": "観察："},
        {"title": "## ✅ Final Answer 回答", "marker": "回答："},
        {"title": "## 🔗 URLs URL", "marker": "URL："},
    ]

    # Find the starting position of each section marker in the text.
    found_sections = []
    for section in sections_to_find:
        # Use a loop to find all occurrences of a marker (like '思考：')
        start_index = -1
        while True:
            start_index = answer_content.find(section["marker"], start_index + 1)
            if start_index == -1:
                break
            found_sections.append(
                {
                    "start": start_index,
                    "title": section["title"],
                    "marker_len": len(section["marker"]),
                }
            )

    # If no markers are found, clean the entire text as a fallback.
    if not found_sections:
        return (
            "No known section markers found. Performing a full clean:\n\n"
            + clean_html(answer_content)
        )

    # Sort the found sections by their starting position to process them in order.
    found_sections.sort(key=lambda x: x["start"])

    output_parts = []
    # Extract the content for each section.
    for i, section in enumerate(found_sections):
        content_start = section["start"] + section["marker_len"]

        # Determine the end of the current section's content.
        # It's either the start of the next section or the end of the string.
        if i + 1 < len(found_sections):
            content_end = found_sections[i + 1]["start"]
        else:
            # For the last section, go all the way to the end of the content.
            content_end = len(answer_content)

        content = answer_content[content_start:content_end]

        # Clean the extracted content block.
        cleaned_content = clean_html(content)

        # Avoid adding empty sections
        if not cleaned_content.strip():
            continue

        output_parts.append(section["title"])
        output_parts.append("---")

        # Special formatting for the "Observation" (Raw Search Results) section.
        if "Raw Search Results" in section["title"]:
            results = cleaned_content.split(
                "################################################"
            )
            for j, result in enumerate(results, 1):
                if result.strip():
                    output_parts.append(f"### Result {j}\n{result.strip()}")
        else:
            output_parts.append(cleaned_content.strip())

        output_parts.append("\n")

    return "\n".join(output_parts)

