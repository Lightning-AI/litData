import base64
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args

from litdata.constants import _CRYPTOGRAPHY_AVAILABLE

if _CRYPTOGRAPHY_AVAILABLE:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class EncryptionLevel:
    SAMPLE = "sample"
    CHUNK = "chunk"


EncryptionLevelType = Literal["sample", "chunk"]


class Encryption(ABC):
    """Base class for encryption algorithm."""

    @property
    @abstractmethod
    def algorithm(self) -> str:
        pass

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str, password: str) -> Any:
        pass


class FernetEncryption(Encryption):
    """Encryption for the Fernet package.

    Adapted from: https://cryptography.io/en/latest/fernet/

    """

    def __init__(
        self,
        password: str,
        level: EncryptionLevelType = "sample",
    ) -> None:
        super().__init__()
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise ModuleNotFoundError(str(_CRYPTOGRAPHY_AVAILABLE))

        if level not in get_args(EncryptionLevelType):
            raise ValueError("The provided `level` should be either `sample` or `chunk`")

        self.password = password
        self.level = level
        self.salt = os.urandom(16)
        self.key = self._derive_key(password, self.salt)
        self.fernet = Fernet(self.key)

    @property
    def algorithm(self) -> str:
        return "fernet"

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "level": self.level,
        }

    def save(self, file_path: str) -> None:
        state = self.state_dict()
        state["salt"] = base64.urlsafe_b64encode(self.salt).decode("utf-8")
        with open(file_path, "wb") as file:
            file.write(json.dumps(state).encode("utf-8"))

    @classmethod
    def load(cls, file_path: str, password: str) -> "FernetEncryption":
        with open(file_path, "rb") as file:
            state = json.load(file)

        salt = base64.urlsafe_b64decode(state["salt"])
        instance = cls(password=password, level=state["level"])
        instance.salt = salt
        instance.key = instance._derive_key(password, salt)
        instance.fernet = Fernet(instance.key)
        return instance


class RSAEncryption(Encryption):
    """Encryption for the RSA package.

    Adapted from: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/rsa/

    """

    def __init__(
        self,
        password: str,
        level: EncryptionLevelType = "sample",
    ) -> None:
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise ModuleNotFoundError(str(_CRYPTOGRAPHY_AVAILABLE))
        if level not in get_args(EncryptionLevelType):
            raise ValueError("The provided `level` should be either `sample` or `chunk`")

        self.password = password
        self.level = level
        self.private_key, self.public_key = self._generate_keys()

    @property
    def algorithm(self) -> str:
        return "rsa"

    def _generate_keys(self) -> tuple[Any, Any]:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def encrypt(self, data: bytes) -> bytes:
        if not self.public_key:
            raise AttributeError("Public key not found.")
        return self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt(self, data: bytes) -> bytes:
        if not self.private_key:
            raise AttributeError("Private key not found.")
        return self.private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def state_dict(self) -> dict[str, Union[str, None]]:
        return {
            "algorithm": self.algorithm,
            "level": self.level,
        }

    def __getstate__(self) -> dict[str, Union[str, None]]:
        encryption_algorithm = (
            serialization.BestAvailableEncryption(self.password.encode())
            if self.password
            else serialization.NoEncryption()
        )
        return {
            "private_key": self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=encryption_algorithm,
            ).decode("utf-8")
            if self.private_key
            else None,
            "public_key": self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")
            if self.public_key
            else None,
            "password": self.password,
            "level": self.level,
        }

    def __setstate__(self, state: dict[str, Union[str, None]]) -> None:
        # Restore the state from the serialized data
        self.password = state["password"] if state["password"] else ""
        self.level = state["level"]  # type: ignore

        if state["private_key"]:
            self.private_key = serialization.load_pem_private_key(
                state["private_key"].encode("utf-8"),
                password=self.password.encode() if self.password else None,
            )
        else:
            self.private_key = None

        if state["public_key"]:
            self.public_key = serialization.load_pem_public_key(
                state["public_key"].encode("utf-8"),
            )
        else:
            self.public_key = None

    def _load_private_key(self, key_path: str, password: str) -> Any:
        with open(key_path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(),
                password=password.encode(),
            )

    def _load_public_key(self, key_path: str) -> Any:
        with open(key_path, "rb") as key_file:
            return serialization.load_pem_public_key(key_file.read())

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as file:
            file.write(json.dumps(self.__getstate__()).encode("utf-8"))

    @classmethod
    def load(cls, file_path: str, password: str) -> "RSAEncryption":
        with open(file_path, "rb") as file:
            state = json.load(file)

        instance = cls(password=password, level=state["level"])
        instance.__setstate__(state)
        return instance

    def save_keys(self, private_key_path: str, public_key_path: str) -> None:
        state = self.__getstate__()
        if not state["private_key"] or not state["public_key"]:
            raise AttributeError("Keys not found.")
        with open(private_key_path, "wb") as key_file:
            key_file.write(state["private_key"].encode("utf-8"))

        with open(public_key_path, "wb") as key_file:
            key_file.write(state["public_key"].encode("utf-8"))

    def load_keys(self, private_key_path: str, public_key_path: str, password: str) -> None:
        self.private_key = self._load_private_key(private_key_path, password)
        self.public_key = self._load_public_key(public_key_path)
