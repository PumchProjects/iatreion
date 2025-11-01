import re
from collections.abc import Callable

rule_symbols = ' _~&|()'
code_pattern = re.compile(r'<u(?P<code>\d+)>')
range_pattern = re.compile(r'(?P<start>.)-(?P<end>.)')


def encode_string(s: str, /, to_replace: str | None = None) -> str:
    if to_replace is None:
        to_replace = rule_symbols
    for symbol in to_replace:
        s = s.replace(symbol, f'<u{ord(symbol)}>')
    return s


def decode_string(s: str, /) -> str:
    def replace(match: re.Match) -> str:
        code = int(match.group('code'))
        return chr(code)

    return code_pattern.sub(replace, s)


def name_to_stem(pattern: str, /) -> Callable[[str], str]:
    compiled = re.compile(pattern)

    def callback(name: str) -> str:
        if (match := compiled.search(name)) is not None:
            return match.group()
        return name

    return callback


def stem_to_name(pattern: str, mapping: dict[str, str], /) -> Callable[[str], str]:
    compiled = re.compile(pattern)

    def callback(stem: str) -> str:
        return compiled.sub(lambda m: mapping[m.group()], stem)

    return callback


def expand_range(s: str, /) -> str:
    def replace(match: re.Match) -> str:
        start = match.group('start')
        end = match.group('end')
        return ''.join(chr(c) for c in range(ord(start), ord(end) + 1))

    return range_pattern.sub(replace, s)
