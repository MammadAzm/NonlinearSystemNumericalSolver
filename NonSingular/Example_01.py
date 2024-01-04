from NumericalSolver import *


# ============================Example 01 System===================================
def system(t, u):
    x, y = u
    dxdt = y
    dydt = (1-x**2)*y-x
    return np.array([dxdt, dydt])

# ============================Example 01 Euler===================================

example_01_euler = NumericalSolver(system=system,  solver="euler",)

example_01_euler.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=600,
          seed=8,)

example_01_euler.plot()

# ===========================Example 01 Adams-Moulton=============================
exmple_01_moulton = NumericalSolver(system=system,  solver="adams-moulton",)

exmple_01_moulton.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=600,
          seed=45,)

exmple_01_moulton.plot()

# ===========================Example 01 Predictor-Corrector=============================

exmple_01_predcorr = NumericalSolver(system=system,  solver="predictor-corrector",)

exmple_01_predcorr.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=600,
          seed=45,)

exmple_01_predcorr.plot()

# ==============Example 01 Runge-Kutta - 2nd order========================

exmple_01_runge2 = NumericalSolver(system=system,  solver="runge-kutta-2",)

exmple_01_runge2.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=600,
          seed=45,)

exmple_01_runge2.plot()

# ==============Example 01 Runge-Kutta - 4th order========================
exmple_01_runge4 = NumericalSolver(system=system,  solver="runge-kutta-4",)

exmple_01_runge4.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=600,
          seed=45,)

exmple_01_runge4.plot()

# ==============Example 01 Adams-Bashforth - 2nd order========================
exmple_01_bash2 = NumericalSolver(system=system,  solver="adams-bashforth-2",)

exmple_01_bash2.solve(initialPointsCount=128,
          individualSolutionsCount=2,
          x0Range=(-8, +8),
          y0Range=(-8, +8),
          timeRange=(0, 20),
          timesCount=600,
          seed=45,)

exmple_01_bash2.plot()
