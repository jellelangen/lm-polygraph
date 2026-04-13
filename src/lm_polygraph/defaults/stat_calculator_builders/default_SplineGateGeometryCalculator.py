from lm_polygraph.stat_calculators import SplineGateGeometryCalculator
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentBase


def load_stat_calculator(cfg, env: BuilderEnvironmentBase):
    return SplineGateGeometryCalculator(
        include_eos=cfg.get("include_eos", False),
        eps=cfg.get("eps", 1e-12),
    )