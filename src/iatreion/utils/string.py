import re

rule_symbols = ' _~&|()'
code_pattern = re.compile(r'<u(?P<code>\d+)>')


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
