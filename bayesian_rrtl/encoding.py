"""Categorical encoding of the actual Fact objects, without Atari imports.

Codes are labels, never ordered cut points. Unobserved entries are -1 and
observed=False. Explicit 'False' facts remain observed.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

CATEGORIES = {"comparative": ("less", "same", "more"), "logical": ("False", "True")}
ENCODER_VERSION = 1


@dataclass(frozen=True)
class RelationKey:
    type: str
    name: str
    obj1: str
    obj2: str | None = None
    obj3: str | None = None

    def __post_init__(self):
        if self.type not in CATEGORIES:
            raise ValueError(f"Unsupported relation type: {self.type}")
        if not self.name or not self.obj1:
            raise ValueError("Relations require a name and first object")

    @classmethod
    def from_fact(cls, fact):
        return cls(fact.type, fact.name, fact.obj1, fact.obj2, fact.obj3)

    def sort_key(self):
        return tuple((value is not None, value or "") for value in
                     (self.type, self.name, self.obj1, self.obj2, self.obj3))


def catalog_hash(catalog):
    payload = {"version": ENCODER_VERSION, "catalog": [asdict(key) for key in catalog]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EncodedBatch:
    values: np.ndarray
    observed: np.ndarray
    catalog: tuple[RelationKey, ...]
    catalog_hash: str

    def __post_init__(self):
        object.__setattr__(self, "catalog", tuple(self.catalog))
        if len(set(self.catalog)) != len(self.catalog):
            raise ValueError("Duplicate relations in catalog")
        values = np.array(self.values, dtype=np.int8, copy=True)
        if not np.array_equal(values, self.values):
            raise ValueError("Categorical codes must be exact small integers")
        observed = np.array(self.observed, dtype=bool, copy=True)
        if values.ndim != 2 or values.shape != observed.shape or values.shape[1] != len(self.catalog):
            raise ValueError("Invalid encoded batch shape")
        if self.catalog_hash != catalog_hash(self.catalog):
            raise ValueError("Catalog hash mismatch")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing entries must use code -1")
        for j, key in enumerate(self.catalog):
            codes = values[observed[:, j], j]
            if np.any((codes < 0) | (codes >= len(CATEGORIES[key.type]))):
                raise ValueError("Invalid categorical value")
        values.flags.writeable = False
        observed.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed", observed)

    def __len__(self):
        return len(self.values)

    def take(self, indices):
        return EncodedBatch(self.values[indices], self.observed[indices], self.catalog, self.catalog_hash)


class RelationalEncoder:
    def __init__(self, catalog):
        self.catalog = tuple(sorted(set(catalog), key=RelationKey.sort_key))
        self.catalog_hash = catalog_hash(self.catalog)
        self._indices = {key: i for i, key in enumerate(self.catalog)}

    @classmethod
    def from_states(cls, states):
        """Infer once on a fixed dataset; later unknown relations raise an error."""
        return cls(RelationKey.from_fact(fact) for state in states for fact in state)

    def transform(self, states):
        states = list(states)
        values = np.full((len(states), len(self.catalog)), -1, dtype=np.int8)
        observed = np.zeros(values.shape, dtype=bool)
        for i, state in enumerate(states):
            for fact in state:
                key = RelationKey.from_fact(fact)
                if key not in self._indices:
                    raise ValueError(f"Unknown relation: {key}")
                j = self._indices[key]
                value = str(fact.value) if isinstance(fact.value, (bool, np.bool_)) else fact.value
                if value not in CATEGORIES[key.type]:
                    raise ValueError(f"Invalid value {value!r} for {key}")
                code = CATEGORIES[key.type].index(value)
                if observed[i, j] and values[i, j] != code:
                    raise ValueError(f"Conflicting facts for {key}")
                values[i, j], observed[i, j] = code, True
        return EncodedBatch(values, observed, self.catalog, self.catalog_hash)

    def to_dict(self):
        return {"version": ENCODER_VERSION, "catalog": [asdict(k) for k in self.catalog],
                "catalog_hash": self.catalog_hash}

    @classmethod
    def from_dict(cls, data):
        if data["version"] != ENCODER_VERSION:
            raise ValueError("Unsupported encoder version")
        encoder = cls(RelationKey(**key) for key in data["catalog"])
        if encoder.catalog_hash != data["catalog_hash"]:
            raise ValueError("Catalog hash mismatch")
        return encoder
