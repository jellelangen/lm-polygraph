import re

NUMBER_REGEX = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_number_gsm8k(s: str) -> str:
    s = s.replace(",", "")
    nums = NUMBER_REGEX.findall(s)
    if not nums:
        return ""
    f = float(nums[-1])
    return str(int(f)) if f == int(f) else str(f)
