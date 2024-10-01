import re
from pathlib import Path
from typing import Union


def get_num_after_keyword(string: Union[str, Path], keyword: str) -> Union[int, float]:
    """Extract a number in the string located after the specific keyword.

    Parameters
    ----------
    string : Union[str, Path]
        The string to be searched.
    keyword : str
        The keyword to be searched.

    Returns
    -------
    number : Union[int, float]
        The extracted number.

    Examples
    --------
    >>> string = "00162 tl015 pl060 tv000 pv000.jpg"
    >>> get_num_after_keyword(string, "tl")
        15
    >>> get_num_after_keyword(string, "pl")
        60
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

    if number.is_integer():
        number = int(number)

    return number
