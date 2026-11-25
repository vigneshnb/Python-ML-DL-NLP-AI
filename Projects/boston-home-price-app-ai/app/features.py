"""Central feature schema - the single source of truth for model inputs.

The ORDER of ``FEATURES`` is the order of the model's input vector. It matches the
column order of ``keras.datasets.boston_housing`` used to train
``boston_home_price_prediction.h5`` and is cross-checked at start-up against
``models/preprocessing.json`` (``feature_order``). Display groups are UI-only and
never change the order in which the input vector is assembled.

Validation bounds
-----------------
``min`` / ``max`` are hard limits: values outside them are physically or
definitionally impossible (negative rates, percentages above 100, B above its
mathematical maximum of 396.9, ...) and are rejected.

Values inside the hard limits but outside the range seen in the training data are
accepted, but the API returns a warning: the model is extrapolating there.

``default`` is the median of the training split (verified by tests against
``preprocessing.json``). Defaults are only used when the user explicitly clicks
"Use training medians"; missing inputs are never filled in silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    description: str
    dtype: str  # "float" | "int" | "binary"
    default: float
    min: float
    max: float
    step: float
    unit: str
    group: str
    control: str = "number"  # "number" | "toggle"
    validation: str = ""
    options: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["options"] = [{"value": v, "label": lbl} for v, lbl in self.options]
        return d


GROUPS: tuple[dict, ...] = (
    {
        "id": "neighborhood",
        "title": "Property & Neighborhood",
        "description": "Land use and crime characteristics of the census tract.",
    },
    {
        "id": "structure",
        "title": "Environmental & Structural",
        "description": "Air quality, dwelling size and housing age.",
    },
    {
        "id": "access",
        "title": "Accessibility & Tax",
        "description": "Location access, property tax and schooling.",
    },
    {
        "id": "historical",
        "title": "Historical Dataset Variables",
        "description": (
            "Socio-economic variables from the original 1970s dataset. They are kept "
            "because the model was trained on them and cannot run without them; they "
            "carry documented historical bias."
        ),
    },
)


FEATURES: tuple[Feature, ...] = (
    Feature(
        name="CRIM", label="Crime rate",
        description="Per-capita crime rate by town.",
        dtype="float", default=0.26888, min=0.0, max=100.0, step=0.00001, unit="per capita",
        group="neighborhood", validation="Number from 0 to 100.",
    ),
    Feature(
        name="ZN", label="Residential zoning",
        description="Share of residential land zoned for lots over 25,000 sq. ft.",
        dtype="float", default=0.0, min=0.0, max=100.0, step=0.1, unit="%",
        group="neighborhood", validation="Percentage from 0 to 100.",
    ),
    Feature(
        name="INDUS", label="Industrial land",
        description="Share of non-retail business acres per town.",
        dtype="float", default=9.69, min=0.0, max=100.0, step=0.01, unit="%",
        group="neighborhood", validation="Percentage from 0 to 100.",
    ),
    Feature(
        name="CHAS", label="Charles River",
        description="Whether the tract bounds the Charles River.",
        dtype="binary", default=0, min=0, max=1, step=1, unit="",
        group="neighborhood", control="toggle", validation="0 (no) or 1 (yes).",
        options=((0, "Not river-adjacent"), (1, "River-adjacent")),
    ),
    Feature(
        name="NOX", label="Nitric oxide level",
        description="Nitric oxides concentration.",
        dtype="float", default=0.538, min=0.0, max=1.0, step=0.001, unit="parts per 10M",
        group="structure", validation="Number from 0 to 1.",
    ),
    Feature(
        name="RM", label="Rooms per dwelling",
        description="Average number of rooms per dwelling.",
        dtype="float", default=6.1985, min=1.0, max=15.0, step=0.001, unit="rooms",
        group="structure", validation="Number from 1 to 15.",
    ),
    Feature(
        name="AGE", label="Older homes",
        description="Share of owner-occupied units built before 1940.",
        dtype="float", default=78.5, min=0.0, max=100.0, step=0.1, unit="%",
        group="structure", validation="Percentage from 0 to 100.",
    ),
    Feature(
        name="DIS", label="Distance to employment",
        description="Weighted distance to five Boston employment centres.",
        dtype="float", default=3.1423, min=0.0, max=20.0, step=0.0001, unit="weighted distance",
        group="structure", validation="Number from 0 to 20.",
    ),
    Feature(
        name="RAD", label="Highway access",
        description="Index of accessibility to radial highways.",
        dtype="int", default=5, min=1, max=24, step=1, unit="index",
        group="access", validation="Whole number from 1 to 24.",
    ),
    Feature(
        name="TAX", label="Property tax rate",
        description="Full-value property-tax rate per $10,000.",
        dtype="float", default=330.0, min=100.0, max=1000.0, step=1, unit="per $10k",
        group="access", validation="Number from 100 to 1000.",
    ),
    Feature(
        name="PTRATIO", label="Pupil–teacher ratio",
        description="Pupil–teacher ratio by town.",
        dtype="float", default=19.1, min=5.0, max=40.0, step=0.1, unit="pupils per teacher",
        group="access", validation="Number from 5 to 40.",
    ),
    Feature(
        name="B", label="B (historical)",
        description="1000(Bk − 0.63)², where Bk is the 1970 proportion of Black residents by town. "
                    "A racially derived variable with documented bias.",
        dtype="float", default=391.25, min=0.0, max=396.9, step=0.01, unit="index",
        group="historical", validation="Number from 0 to 396.9 (its mathematical maximum).",
    ),
    Feature(
        name="LSTAT", label="Lower-status population",
        description="Share of the population classed as 'lower status' in the 1970 census.",
        dtype="float", default=11.395, min=0.0, max=100.0, step=0.01, unit="%",
        group="historical", validation="Percentage from 0 to 100.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def schema_for_client(training_ranges: dict[str, tuple[float, float]] | None = None) -> dict:
    """Serialisable schema consumed by the template and front-end JS."""
    feats = []
    for f in FEATURES:
        d = f.to_dict()
        if training_ranges and f.name in training_ranges:
            lo, hi = training_ranges[f.name]
            d["train_min"], d["train_max"] = lo, hi
        feats.append(d)
    return {"groups": list(GROUPS), "features": feats}
