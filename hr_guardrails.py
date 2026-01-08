HR_KEYWORDS = [
    "leave", "vacation", "benefits", "pay",
    "termination", "harassment", "policy",
    "performance", "manager", "employee"
]

def is_hr_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in HR_KEYWORDS)