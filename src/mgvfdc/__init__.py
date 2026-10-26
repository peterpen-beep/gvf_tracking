"""MGVFDC：改进 GVF + GPIO 抗扰的四足机器人无碰撞目标跟踪制导控制复现。

论文：S. Kong, J. Sun, A. Luo, et al.,
"A Collision-Free Target Tracking Controller With Uncertain Disturbance Rejection
 for Quadruped Robots," IEEE Trans. Intell. Veh., vol. 9, no. 1, pp. 670-680, 2024.
DOI: 10.1109/TIV.2023.3296669
"""

from .config import (GVFConfig, GuidanceConfig, GPIOConfig, SimConfig,
                     PAPER_DISTURBANCE, paper_gvf_config, paper_guidance_config)
from .controllers import ControlOut, MGVFDCController, make_controller
from .gvf import (FieldDiagnostics, attractive_field, beta_triplet, blend_weight,
                  blending_function, circulation_sense, guidance_angle,
                  integrate_field_line, repulsive_field, resultant_field)
from .metrics import format_table, summarize
from .observer import GPIO
from .plant import (CircularTarget, Obstacle, StaticTarget, Unicycle, Target,
                    obstacle_arrays, safety_clearance, wrap_angle)
from .scenarios import (ALL_CASES, Scenario, case1, case2, case3, fig3_scenario,
                        fig3_longrange, fig5_scenario, stress_dynamic_obstacle)
from .sim import SimResult, run_comparison, run_simulation

__all__ = [
    "GVFConfig", "GuidanceConfig", "GPIOConfig", "SimConfig", "PAPER_DISTURBANCE",
    "paper_gvf_config", "paper_guidance_config",
    "ControlOut", "MGVFDCController", "make_controller",
    "FieldDiagnostics", "attractive_field", "beta_triplet", "blend_weight",
    "blending_function", "circulation_sense", "guidance_angle",
    "integrate_field_line", "repulsive_field", "resultant_field",
    "format_table", "summarize",
    "GPIO",
    "CircularTarget", "Obstacle", "StaticTarget", "Unicycle", "Target",
    "obstacle_arrays", "safety_clearance", "wrap_angle",
    "ALL_CASES", "Scenario", "case1", "case2", "case3", "fig3_scenario",
    "fig3_longrange", "fig5_scenario", "stress_dynamic_obstacle",
    "SimResult", "run_comparison", "run_simulation",
]

__version__ = "1.0.0"
