"""Discord bot starter: thin transport, thick testable logic, clean shutdown."""

from __future__ import annotations

from ..addons import env_example, gitignore
from ..spec import ProjectOptions, TemplateSpec

INIT = '''"""__DISPLAY_NAME__ - Discord bot."""

__version__ = "0.1.0"
'''

CONFIG = '''"""Environment configuration, validated once at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    token: str
    prefix: str = "!"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        values = os.environ if env is None else env
        token = (values.get("DISCORD_TOKEN") or "").strip()
        if not token:
            raise ConfigError(
                "DISCORD_TOKEN is not set. Create a bot at https://discord.com/developers/applications, "
                "copy the token into .env, and never commit it."
            )
        if len(token) < 20:
            raise ConfigError("DISCORD_TOKEN looks too short to be valid")
        return cls(token=token, prefix=values.get("BOT_PREFIX") or "!")
'''

LOGIC = '''"""Pure bot logic: no discord import, no network, fully unit-testable."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

DICE = re.compile(r"^(\\d{0,2})d(\\d{1,3})([+-]\\d{1,3})?$", re.IGNORECASE)
MAX_MESSAGE = 2000


class CommandError(ValueError):
    """A user-facing error: the message is safe to send back to Discord."""


@dataclass
class Roll:
    count: int
    sides: int
    modifier: int
    rolls: list[int]

    @property
    def total(self) -> int:
        return sum(self.rolls) + self.modifier

    def render(self) -> str:
        detail = " + ".join(str(roll) for roll in self.rolls)
        if self.modifier:
            detail += f" {'+' if self.modifier > 0 else '-'} {abs(self.modifier)}"
        return f"🎲 {detail} = **{self.total}**"


def parse_roll(expression: str, rng: random.Random | None = None) -> Roll:
    """Parse real dice notation like ``2d6+3`` / ``d20`` and roll it."""

    text = (expression or "").strip().lower().replace(" ", "")
    if not text:
        text = "1d6"
    match = DICE.match(text)
    if not match:
        raise CommandError(f"`{expression}` is not dice notation - try `2d6+3`")
    count = int(match.group(1) or 1)
    sides = int(match.group(2))
    modifier = int(match.group(3) or 0)
    if not 1 <= count <= 20:
        raise CommandError("roll between 1 and 20 dice")
    if not 2 <= sides <= 1000:
        raise CommandError("dice need between 2 and 1000 sides")
    generator = rng or random.SystemRandom()
    return Roll(count=count, sides=sides, modifier=modifier, rolls=[generator.randint(1, sides) for _ in range(count)])


def chunk_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Split long text on line boundaries so Discord never rejects it."""

    if limit < 50:
        raise ValueError("limit is too small to be useful")
    chunks: list[str] = []
    current = ""
    for line in (text or "").splitlines():
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current or not chunks:
        chunks.append(current)
    return chunks


def format_uptime(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def is_command(content: str, prefix: str = "!") -> bool:
    text = (content or "").strip()
    return bool(prefix) and text.startswith(prefix) and len(text) > len(prefix)
'''

BOT = '''"""Discord transport: commands, error handling and graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import time

import discord
from discord.ext import commands

from .config import Config
from .logic import CommandError, chunk_message, format_uptime, parse_roll

log = logging.getLogger("bot")
STARTED_AT = time.monotonic()


class Bot(commands.Bot):
    def __init__(self, config: Config):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.prefix, intents=intents, help_command=None)
        self.config = config

    async def on_ready(self) -> None:
        log.info("logged in as %s (%s guilds)", self.user, len(self.guilds))

    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        if isinstance(error, CommandError):
            await ctx.reply(str(error))
        elif isinstance(error, commands.CommandNotFound):
            return
        else:
            log.exception("command failed", exc_info=error)
            await ctx.reply("Something went wrong - the details are in the bot logs.")

    async def close(self) -> None:
        log.info("shutting down")
        await super().close()


def create_bot(config: Config) -> Bot:
    bot = Bot(config)

    @bot.command(name="ping")
    async def ping(ctx: commands.Context) -> None:
        """Round-trip latency."""
        await ctx.reply(f"Pong! {round(bot.latency * 1000)}ms")

    @bot.command(name="roll")
    async def roll(ctx: commands.Context, *, expression: str = "1d6") -> None:
        """Roll dice, e.g. `!roll 2d6+3`."""
        await ctx.reply(parse_roll(expression).render())

    @bot.command(name="uptime")
    async def uptime(ctx: commands.Context) -> None:
        await ctx.reply(f"Up for {format_uptime(time.monotonic() - STARTED_AT)}")

    @bot.command(name="echo")
    async def echo(ctx: commands.Context, *, text: str = "") -> None:
        """Repeat text (long text is split safely)."""
        for chunk in chunk_message(text or "(nothing to echo)"):
            await ctx.send(chunk)

    @bot.command(name="about")
    async def about(ctx: commands.Context) -> None:
        await ctx.reply("__DISPLAY_NAME__ - built from the discord-bot starter.")

    return bot


async def _run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bot = create_bot(Config.from_env())
    async with bot:
        await bot.start(bot.config.token)


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        log.info("interrupted - bye")


if __name__ == "__main__":
    main()
'''

TESTS = '''"""Pure logic tests - they never connect to Discord."""

from __future__ import annotations

import random

import pytest

from __MODULE__.logic import (
    CommandError,
    chunk_message,
    format_uptime,
    is_command,
    parse_roll,
)


def test_dice_notation():
    roll = parse_roll("2d6+3", rng=random.Random(7))
    assert roll.count == 2 and roll.sides == 6 and roll.modifier == 3
    assert 5 <= roll.total <= 15
    assert roll.render().startswith("🎲")


def test_default_roll():
    assert parse_roll("").count == 1


@pytest.mark.parametrize("expression", ["abc", "0d6", "d1", "99d6", "2d6+", "d"])
def test_bad_dice_notation(expression):
    with pytest.raises(CommandError):
        parse_roll(expression)


def test_chunking_respects_limit():
    text = "\\n".join(["x" * 60 for _ in range(10)])
    chunks = chunk_message(text, limit=100)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).replace("\\n", "") == text.replace("\\n", "")


def test_chunking_keeps_short_message_whole():
    assert chunk_message("hello", limit=100) == ["hello"]


def test_uptime_formatting():
    assert format_uptime(0) == "0s"
    assert format_uptime(65) == "1m 5s"
    assert format_uptime(90061) == "1d 1h 1m 1s"


def test_command_detection():
    assert is_command("!ping")
    assert not is_command("ping")
    assert not is_command("!")
'''

REQUIREMENTS = """discord.py==2.3.2
# dev / test
pytest==8.2.2
"""


CONFTEST = """import sys
from pathlib import Path

# Make the package importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).parent))
"""


def _tokens(text: str, options: ProjectOptions) -> str:
    """Replace __TOKEN__ markers; generated code keeps its own braces."""

    return (
        text.replace("__MODULE__", options.module)
        .replace("__DISPLAY_NAME__", options.display_name)
        .replace("__DESCRIPTION__", options.description or f"{options.display_name} Discord bot")
        .replace("__NAME__", options.name)
        .replace("__PORT__", str(options.port))
    )


def build(options: ProjectOptions) -> dict[str, str]:
    return {
        f"{options.module}/__init__.py": _tokens(INIT, options),
        f"{options.module}/config.py": CONFIG,
        f"{options.module}/logic.py": LOGIC,
        f"{options.module}/bot.py": _tokens(BOT, options),
        "tests/test_logic.py": _tokens(TESTS, options),
        "conftest.py": CONFTEST,
        "requirements.txt": REQUIREMENTS,
        ".env.example": env_example(options, include_secret=False) + "DISCORD_TOKEN=\\n",
        ".gitignore": gitignore(),
    }


SPEC = TemplateSpec(
    id="discord-bot",
    name="Discord bot",
    tagline="Commands, dice/utility logic, graceful shutdown - no port to expose",
    description=(
        "A Discord bot with slash-style prefix commands, error handling that answers the "
        "user instead of crashing, and all the interesting logic in a testable module. "
        "Runs as a worker: no web server, no public URL."
    ),
    stack="Python 3.11 · discord.py 2 · asyncio",
    kind="worker",
    tags=["bot", "discord", "worker", "python", "community"],
    supports_auth=False,
    supports_database=False,
    default_port=0,
    build=build,
    good_for=["community servers", "moderation helpers", "fun utilities"],
    run_command="python -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt\ncp .env.example .env  # add DISCORD_TOKEN\npython -m __MODULE__.bot",
    test_command="python -m pytest -q",
    preview="worker",
    next_steps=[
        "Create a bot + token: https://discord.com/developers/applications",
        "pip install -r requirements.txt, put DISCORD_TOKEN in .env",
        "python -m __MODULE__.bot (tests run without a token)",
    ],
)
