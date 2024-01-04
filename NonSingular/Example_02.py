from NumericalSolver import *


# ============================Example 02 System===================================
def system(t, u):
    x, y = u
    dxdt = y - x*(x**2+y**2-1)
    dydt = -x-y*(x**2+y**2-1)
    return np.array([dxdt, dydt])

# ============================Example 02 Euler===================================

example_02_euler = NumericalSolver(system=system,  solver="euler",)

example_02_euler.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=4000,
          seed=8,)

example_02_euler.plot()

# ===========================Example 02 Adams-Moulton=============================
example_02_moulton = NumericalSolver(system=system,  solver="adams-moulton",)

example_02_moulton.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=4000,
          seed=11,)

example_02_moulton.plot()

# ===========================Example 02 Predictor-Corrector=============================

example_02_predcorr = NumericalSolver(system=system,  solver="predictor-corrector",)

example_02_predcorr.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=4000,
          seed=11,)

example_02_predcorr.plot()

# ==============Example 02 Runge-Kutta - 2nd order========================

example_02_runge2 = NumericalSolver(system=system,  solver="runge-kutta-2",)

example_02_runge2.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=4000,
          seed=11,)

example_02_runge2.plot()

# ==============Example 02 Runge-Kutta - 4th order========================
example_02_runge4 = NumericalSolver(system=system,  solver="runge-kutta-4",)

example_02_runge4.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=4000,
          seed=11,)

example_02_runge4.plot()

# ==============Example 02 Adams-Bashforth - 2nd order========================
example_02_bash2 = NumericalSolver(system=system,  solver="adams-bashforth-2",)

example_02_bash2.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=4000,
          seed=11,)

example_02_bash2.plot()
