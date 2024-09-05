import re


def get_num_after_keyword(string: str, keyword: str) -> float:
    """Extract a number in the string located after the specific keyword.

    Parameters
    ----------
    string : str
        The string to be searched.
    keyword : str
        The keyword to be searched.

    Returns
    -------
    number : float
        The extracted number.

    Examples
    --------
    >>> string = "00162 tl015 pl060 tv000 pv000.jpg"
    >>> get_num_after_keyword(string, "tl")
        15.0
    >>> get_num_after_keyword(string, "pl")
        60.0
    """
    pattern = rf"{keyword}([-+]?\d+\.?\d?)"
    string = str(string)
    matches = re.findall(pattern, string)
    num_matches = len(matches)

    if num_matches == 0:
        raise RuntimeError(f"No match found in '{string}'. key='{keyword}'")
    elif num_matches > 1:
        raise RuntimeError(f"Multiple matches found {matches} in '{string}'. key='{keyword}'")

    number = float(matches[0])
    return number
