
import re
from typing import Tuple


class TimeSignatureUtil:
    """
    Conversion between time signatures and REMI-z time-signature tokens ``ts-(num,den)``,
    e.g. ``ts-(4,4)``, ``ts-(6,8)`` (no space, so the token survives whitespace splitting).

    Supported: denominator a power of 2 from 1 to 64, numerator from 1 to 2 * denominator
    (a bar is at most two whole notes long).
    """

    MAX_DENOMINATOR = 64
    TOKEN_PATTERN = re.compile(r"ts-\((\d+),(\d+)\)")

    @classmethod
    def is_supported(cls, numerator: int, denominator: int) -> bool:
        """
        Check whether a time signature has a token.

        Args:
            numerator (int): Time-signature numerator.
            denominator (int): Time-signature denominator.

        Returns:
            bool: True if the denominator is a power of 2 in [1, 64] and the numerator is in
                [1, 2 * denominator].
        """
        return (
            1 <= denominator <= cls.MAX_DENOMINATOR
            and denominator & (denominator - 1) == 0
            and 1 <= numerator <= 2 * denominator
        )

    @classmethod
    def convert_time_signature_to_ts_token(cls, numerator: int, denominator: int) -> str:
        """
        Convert a time signature to its token.

        Args:
            numerator (int): Time-signature numerator.
            denominator (int): Time-signature denominator.

        Returns:
            str: The token ``ts-(numerator,denominator)``, e.g. ``ts-(6,8)``.

        Raises:
            ValueError: If the time signature is not supported (see ``is_supported``).
        """
        if not cls.is_supported(numerator, denominator):
            raise ValueError(f"Invalid time signature: {numerator}/{denominator}")
        return f"ts-({numerator},{denominator})"

    @classmethod
    def convert_time_signature_token_to_tuple(cls, token: str) -> Tuple[int, int]:
        """
        Convert a time-signature token to a time signature.

        Args:
            token (str): A token ``ts-(num,den)``, e.g. ``ts-(6,8)``.

        Returns:
            Tuple[int, int]: ``(numerator, denominator)``.

        Raises:
            ValueError: If the token is malformed or the time signature is not supported.
        """
        m = cls.TOKEN_PATTERN.fullmatch(token)
        if m is None or not cls.is_supported(int(m.group(1)), int(m.group(2))):
            raise ValueError(f"Invalid time signature token: {token}")
        return int(m.group(1)), int(m.group(2))
