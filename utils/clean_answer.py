def clean_answer(answer: str) -> str:
    """
    Clean and standardize an answer string.
    
    Args:
        answer: The answer string to clean
        
    Returns:
        Cleaned answer string
    """
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove trailing period
    if answer.endswith("."):
        answer = answer[:-1]
    
    # Handle yes/no answers
    if any(prefix in answer for prefix in ["yes,", "yes. "]):
        return "yes"
    elif any(prefix in answer for prefix in ["no,", "no. "]):
        return "no"
    
    # Clean up parentheses content
    if "(" in answer and ")" in answer and answer.split("(")[0].strip() != "":
        answer = answer.split("(")[0].strip()
    
    # Handle alternatives
    if " or " in answer:
        answer = answer.split(" or ")[0].strip()
    
    # Handle equality statements
    if " is " in answer:
        answer = answer.split(" is ")[-1].strip()
    if " = " in answer:
        answer = answer.split(" = ")[-1].strip()
    if " ≈ " in answer:
        answer = answer.split(" ≈ ")[-1].strip()

    for unit in [
            "degree",
            "cm",
            "centimeter",
            "meter",
            "mile",
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "year",
            "foot",
            "feet",
            "inch",
            "yard",
            "liter",
            "gram",
            "kilogram",
            "ounce",
            "pound",
        ]:
        if unit in answer:
            answer = answer.split(unit)[0].strip()
    return answer