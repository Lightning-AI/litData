# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from functools import lru_cache
from importlib.util import find_spec
from typing import TypeVar

import pkg_resources
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


@lru_cache
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False

    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@lru_cache
def module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> module_available('os')
    True
    >>> module_available('os.bla')
    False
    >>> module_available('bla.bla')
    False

    """
    module_names = module_path.split(".")
    if not package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


class RequirementCache:
    """Boolean-like class to check for requirement and module availability.

    Args:
        requirement: The requirement to check, version specifiers are allowed.
        module: The optional module to try to import if the requirement check fails.

    >>> RequirementCache("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(RequirementCache("torch>=0.1"))
    True
    >>> bool(RequirementCache("torch>100.0"))
    False
    >>> RequirementCache("torch")
    Requirement 'torch' met
    >>> bool(RequirementCache("torch"))
    True
    >>> bool(RequirementCache("unknown_package"))
    False

    """

    def __init__(self, requirement: str, module: str | None = None) -> None:
        self.requirement = requirement
        self.module = module

    def _check_requirement(self) -> None:
        if hasattr(self, "available"):
            return
        try:
            # first try the pkg_resources requirement
            pkg_resources.require(self.requirement)
            self.available = True
            self.message = f"Requirement {self.requirement!r} met"
        except Exception as ex:
            self.available = False
            self.message = f"{ex.__class__.__name__}: {ex}.\n HINT: Try running `pip install -U {self.requirement!r}`"
            requirement_contains_version_specifier = any(c in self.requirement for c in "=<>")
            if not requirement_contains_version_specifier or self.module is not None:
                module = self.requirement if self.module is None else self.module
                # sometimes `pkg_resources.require()` fails but the module is importable
                self.available = module_available(module)
                if self.available:
                    self.message = f"Module {module!r} available"

    def __bool__(self) -> bool:
        """Format as bool."""
        self._check_requirement()
        return self.available

    def __str__(self) -> str:
        """Format as string."""
        self._check_requirement()
        return self.message

    def __repr__(self) -> str:
        """Format as string."""
        return self.__str__()
