"""Example evaluation: a second-order IIR recurrence.

The recurrence is::

    y[n] = x[n] + 1.8 * y[n-1] - 0.9 * y[n-2]

which in vecrec is ``Recurse(TIKernel([0, 1.8, -0.9]), Var("x"))`` (the kernel
taps are the feedback coefficients on ``y[n-0], y[n-1], y[n-2]``; the leading
``0`` means ``y[n]`` does not feed back into itself).

This case runs out-of-the-box with only numpy installed. It demonstrates the two
halves of the harness:

* ``run_vecrec`` generates / compiles / runs the vecrec C++ benchmark across a
  few transformed variants and reports their throughput (plus vecrec's own
  internal correctness validation across variants).
* ``baselines`` time external-library implementations of the same recurrence and
  check them against a numpy reference. A scipy baseline is added automatically
  only if scipy is importable, illustrating how to wire in optional libraries.
"""

from __future__ import annotations

import numpy as np

from vecrec import CodeGen, generate_and_run_benchmark
from vecrec.expr import Recurse, TIKernel, Type, Var, with_types
from vecrec.transform import (
    AnnotateLanes,
    Any,
    ConstantFold,
    Delay,
    Dilate,
    Optional,
    PushDownConvertLanes,
    SaturatePreorder,
    Seq,
)
from vecrec.util import ElementType

from evals_common.harness import Baseline, EvalCase, platform_compiler_flags

# Feedback coefficients on y[n-1] and y[n-2].
A1 = 1.8
A2 = -0.9


def _numpy_loop(x: np.ndarray) -> np.ndarray:
    """Reference IIR via an explicit numpy loop."""
    y = np.zeros_like(x)
    for n in range(len(x)):
        prev1 = y[n - 1] if n >= 1 else 0.0
        prev2 = y[n - 2] if n >= 2 else 0.0
        y[n] = x[n] + A1 * prev1 + A2 * prev2
    return y


class FirstOrderIIR(EvalCase):
    name = "second-order-iir"
    vecrec_bytes_per_cell = 8  # float32 input + float32 output
    vecrec_work_mode = "full_vector"
    baseline_work_mode = "full_vector"
    boundary_convention = "1D IIR recurrence (no alignment boundary)"

    def __init__(self, n_elements: int = 1 << 14) -> None:
        # kept small so the Python baselines stay fast
        self.n_elements = n_elements

    def run_vecrec(self) -> dict:
        with with_types(Type.Arith, ElementType.Float):
            expr = Recurse(TIKernel([0, A1, A2]), Var("x"))

        schedule = Seq(
            Optional(
                Seq(
                    Dilate(),
                    Dilate(),
                    Any(Dilate(), Delay()),
                    SaturatePreorder(ConstantFold),
                )
            ),
            AnnotateLanes(512),
            PushDownConvertLanes(),
        )
        results = schedule.apply_signal(expr)

        codegen = CodeGen()
        return generate_and_run_benchmark(
            codegen=codegen,
            exprs=results[:3],
            include_correctness_check=True,
            input_size=1 << 16,
            warmup_iterations=5,
            benchmark_iterations=10,
            compiler_flags=platform_compiler_flags(),
        )

    def make_inputs(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1.0, 1.0, size=self.n_elements).astype(np.float32)
        return {"x": x}

    def reference(self, **inputs: np.ndarray) -> np.ndarray:
        return _numpy_loop(inputs["x"])

    def baselines(self) -> list[Baseline]:
        baselines = [Baseline("numpy_loop", lambda **kw: _numpy_loop(kw["x"]))]

        try:
            from scipy.signal import lfilter
        except ImportError:
            pass
        else:
            # y[n] - A1*y[n-1] - A2*y[n-2] = x[n]  =>  a = [1, -A1, -A2], b = [1]
            baselines.append(
                Baseline(
                    "scipy_lfilter",
                    lambda **kw: lfilter([1.0], [1.0, -A1, -A2], kw["x"]).astype(np.float32),
                )
            )
        return baselines


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([FirstOrderIIR()]))
