"""Standard link functions and helpers."""

from __future__ import annotations

from typing import Mapping, Type

from ..base import LinkFunction
from .common import CLogLogLink, IdentityLink, InverseLink, LogLink, LogitLink

_LINK_REGISTRY: Mapping[str, Type[LinkFunction]] = {
	"identity": IdentityLink,
	"log": LogLink,
	"logit": LogitLink,
	"inverse": InverseLink,
	"cloglog": CLogLogLink,
	"comploglog": CLogLogLink,
}


def get_link(link: str | LinkFunction) -> LinkFunction:
	"""Resolve a link identifier or instance into a usable link object."""

	if isinstance(link, LinkFunction):
		return link
	if isinstance(link, str):
		key = link.lower()
		factory = _LINK_REGISTRY.get(key)
		if factory is None:
			raise ValueError(f"Unknown link identifier: {link}")
		return factory()
	raise TypeError("Link must be a string name or LinkFunction instance")


__all__ = [
	"IdentityLink",
	"LogLink",
	"LogitLink",
	"InverseLink",
	"CLogLogLink",
	"get_link",
]