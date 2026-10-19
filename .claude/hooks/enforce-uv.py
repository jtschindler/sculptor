#!/usr/bin/env python3
import json
import os
import re
import shlex
import sys

command = json.load(sys.stdin).get("tool_input", {}).get("command", "")

redirects = {
    "pip": "uv add / uv remove",
    "pytest": "uv run pytest",
    "ruff": "uv run ruff",
    "mypy": "uv run mypy",
    "pyrefly": "uv run pyrefly",
    "ty": "uv run ty",
    "prek": "uv run prek",
    "pre-commit": "uv run prek",
    "uvx": "uv run (add the tool with `uv add --dev`, or `uv run --with <pkg>`)",
}
# Wrappers that run their argument as a program (env python, time pytest, ...).
wrappers = {"env", "exec", "time", "nohup", "command", "sudo", "xargs", "nice"}


def suggestion(name):
    if re.fullmatch(r"python(\d+(\.\d+)?)?", name):
        return "uv run python"
    if re.fullmatch(r"pip(\d+(\.\d+)?)?", name):
        return redirects["pip"]
    return redirects.get(name)


def strip_heredocs(text):
    # Drop heredoc bodies: they are data, not commands.
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in re.finditer(r"<<(-?)\s*(['\"]?)(\w+)\2", line):
            strip_tabs, delim = m.group(1), m.group(3)
            while i < len(lines):
                body = lines[i].lstrip("\t") if strip_tabs else lines[i]
                i += 1
                if body == delim:
                    break
    return "\n".join(out)


def block(name):
    print(f"Blocked: bare '{name}'. Use '{suggestion(name)}' instead.", file=sys.stderr)
    sys.exit(2)


try:
    text = strip_heredocs(command).replace("\n", ";")
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
except ValueError:
    sys.exit(0)  # unparseable (e.g. unbalanced quotes): let it through

# Only inspect the program name at the start of each sub-command,
# so quoted text and arguments (grep 'pip install', commit messages) are ignored.
at_start = True
in_wrapper = False
for i, tok in enumerate(tokens):
    if set(tok) <= set("&|;()"):
        at_start, in_wrapper = True, False
        continue
    if not at_start:
        continue
    if "=" in tok and not tok.startswith("="):
        continue  # leading VAR=value assignment
    if in_wrapper and tok.startswith("-"):
        continue  # wrapper flags, e.g. env -i, time -p
    name = os.path.basename(tok)
    if name in wrappers:
        in_wrapper = True
        continue
    if suggestion(name):
        block(name)
    if name == "uv" and tokens[i + 1 : i + 3] == ["tool", "run"]:
        block("uvx")
    at_start = in_wrapper = False
