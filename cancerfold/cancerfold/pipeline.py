"""End-to-end unified runner for CancerFold ensembles (Target: v0.2.0)."""


def describe_pipeline() -> str:
    """Return a concise description of the planned CancerFold pipeline."""
    return (
        "CancerFold pipeline: input parsing -> AF2 evolutionary trunk -> "
        "stochastic SDE StructureModule -> ensemble thermodynamics analysis."
    )
