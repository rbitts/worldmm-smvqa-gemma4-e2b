from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, override

RuntimeLocation = Literal["local", "remote"]


@dataclass(frozen=True, slots=True)
class ConfigNotFoundError(Exception):
    path: Path

    @override
    def __str__(self) -> str:
        return f"ConfigNotFound: {self.path}"


@dataclass(frozen=True, slots=True)
class MalformedConfigError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"MalformedConfig: {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class RemoteOnlyError(Exception):
    action: str

    @override
    def __str__(self) -> str:
        return f"remote-only: {self.action}"


@dataclass(frozen=True, slots=True)
class MissingRemoteConfigError(Exception):
    name: str

    @override
    def __str__(self) -> str:
        return f"MissingRemoteConfig: {self.name}"


@dataclass(frozen=True, slots=True)
class AppConfig:
    path: Path
    runtime_location: RuntimeLocation
    values: dict[str, dict[str, str]]


REMOTE_ENV_FLAG: Final = "WORLDMM_SMVQA_ALLOW_REMOTE_ON_THIS_HOST"


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ConfigNotFoundError(path=path)

    values: dict[str, dict[str, str]] = {}
    section = ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MalformedConfigError(path=path, detail=str(exc)) from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise MalformedConfigError(
                    path=path,
                    detail=f"line {line_number}: expected section",
                )
            section = line[:-1].strip()
            values[section] = {}
            continue
        if not section:
            raise MalformedConfigError(
                path=path,
                detail=f"line {line_number}: key before section",
            )
        key, separator, value = line.strip().partition(":")
        if separator == "":
            raise MalformedConfigError(
                path=path,
                detail=f"line {line_number}: expected key: value",
            )
        values[section][key.strip()] = value.strip().strip('"').strip("'")

    runtime_location = values.get("runtime", {}).get("location")
    match runtime_location:
        case "local" | "remote":
            return AppConfig(
                path=path,
                runtime_location=runtime_location,
                values=values,
            )
        case None:
            raise MalformedConfigError(path=path, detail="missing runtime.location")
        case other:
            raise MalformedConfigError(
                path=path,
                detail=f"invalid runtime.location={other}",
            )


def require_remote(config: AppConfig, action: str, env: Mapping[str, str]) -> None:
    if config.runtime_location == "remote" and env.get(REMOTE_ENV_FLAG) == "1":
        return
    raise RemoteOnlyError(action=action)


def require_env(name: str, env: Mapping[str, str]) -> str:
    value = env.get(name)
    if value:
        return value
    raise MissingRemoteConfigError(name=name)
