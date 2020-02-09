"""Interfaces for BEC codes.
"""

from zope.interface import Attribute, Interface, implementer


class IBasis(Interface):
    """Interface for a DVR basis."""


class IHFB(Interface):
    """Interface for HFB-like codes."""
    basis = Attribute("IBasis instance.")
    functional = Attribute("IFunctional")
