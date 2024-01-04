import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
from typing import Callable
import time

import warnings
warnings.filterwarnings("ignore")


class NumericalSolver:
    def __init__(self, system: Callable, solver: str):
        """
        # Constructor Arguments:
            - `system` : Callable function of 2nd-order differential equation system
            - `solver` : Of string type. Options are:
                * `euler`
                * `adams-moulton`
                * `predictor-corrector`
                * `runge-kutta-2`
                * `runge-kutta-4`
                * `adams-bashforth-2`

        See Examples...
        """
        self.system = system
        self.solutions = {}

        if solver == "euler":
            self.solver = self._euler_solver
        elif solver == "adams-moulton":
            self.solver = self._adams_moulton_solver
        elif solver == "predictor-corrector":
            self.solver = self._predictor_corrector_solver
        elif solver == "runge-kutta-2":
            self.solver = self._runge_kutta_second_order_solver
        elif solver == "runge-kutta-4":
            self.solver = self._runge_kutta_forth_order_solver
        elif solver == "adams-bashforth-2":
            self.solver = self._adams_bashforth_2nd_order_solver


    def solve(self,
              initialPointsCount: int = 32,
              individualSolutionsCount: int = 4,
              x0Range: tuple = (-8, +8),
              y0Range: tuple = (-8, +8),
              timeRange: tuple = (0, 100),
              timesCount: int = 10000,
              seed: int = 45,
              ):
        """
        Inputs:
            * `initialPointsCount` : Of `int` type. Count of initial points you want to solve the system for.
            * `individualSolutionsCount` : Of `int` type by which `initialPointsCount` must be divisivle. Count of states you want to see their solution with respect to time.
            * `x0Range` : Of `tuple - (number, number)` type. The range of initial conditions for state x1.
            * `y0Range` : Of `tuple - (number, number)` type. The range of initial conditions for state x2.
            * `timeRange` : Of `tuple - (number, number)` type. The timespan in which the system is going to be analyzed.
            * `timesCount` : Of type `int`. Count of sample times you want to devide your timespan by.
            * `seed` : Of type `int`. Random Seed.
            
        What to do Next !?
            Run `.plot` method on your instance to get your solutions.
        """
        random.seed(seed)
        size = initialPointsCount
        eachStateSize = individualSolutionsCount

        x0 = [random.uniform(x0Range[0], x0Range[1]) for _ in range(size)]
        y0 = [random.uniform(y0Range[0], y0Range[1]) for _ in range(size)]

        time_points = np.linspace(timeRange[0], timeRange[1], timesCount)

        self.solutions[self.getSolver(self.solver)] = {
            "solutions": [],
            "initialPointsCount": size,
            "individualSolutionsCount": eachStateSize,
            "initialValues": [(x0[i], y0[i]) for i in range(len(x0))],
            "x0s": x0,
            "y0s": y0,
            "timeSpan": time_points,
            "x0Range": x0Range,
            "y0Range": y0Range,
        }

        print("Processing...")
        start = time.time()
        for i in tqdm(range(len(x0))):

            initial_conditions = np.array([x0[i], y0[i]])

            solution = self.solver(self.system, initial_conditions, time_points)

            self.solutions[self.getSolver(self.solver)]["solutions"].append(solution)
        end = time.time()

        self.solutions[self.getSolver(self.solver)]["runtime"] = end - start

    def plot(self,):
        """
        No inputs required. Just call this method on your function so that you could visualize the solutions for your given system.
        """
        print("Plotting...")

        plt.figure(figsize=(12, 8))
        plt.title(f'{self.getSolver(self.solver)} Method Solution for Nonlinear System')

        for solution in tqdm(self.solutions[self.getSolver(self.solver)]["solutions"]):
            plt.plot(solution[:, 0], solution[:, 1], color="blue", linewidth=0.75)

        plt.scatter(self.solutions[self.getSolver(self.solver)]["x0s"], self.solutions[self.getSolver(self.solver)]["y0s"], s=10, c="black")

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(self.solutions[self.getSolver(self.solver)]["x0Range"])
        plt.ylim(self.solutions[self.getSolver(self.solver)]["y0Range"])
        plt.grid(True)
        plt.show()

        indices = []
        for _ in range(self.solutions[self.getSolver(self.solver)]["individualSolutionsCount"]):
            indices.append(random.randint(0,self.solutions[self.getSolver(self.solver)]["initialPointsCount"]))

        print("\n--------------------------------------------------------------------\n")
        print("Solution for each state of the system with respect to some of the given initial conditions\n")

        plt.figure(figsize=(12, 10))
        for i in list(range(0,self.solutions[self.getSolver(self.solver)]["initialPointsCount"],int(self.solutions[self.getSolver(self.solver)]["initialPointsCount"]/self.solutions[self.getSolver(self.solver)]["individualSolutionsCount"]))):

            plt.subplot(int(f"{self.solutions[self.getSolver(self.solver)]['individualSolutionsCount']}2{list(range(0, self.solutions[self.getSolver(self.solver)]['initialPointsCount'], int(self.solutions[self.getSolver(self.solver)]['initialPointsCount'] / self.solutions[self.getSolver(self.solver)]['individualSolutionsCount']))).index(i) + 1}"))
            plt.subplots_adjust(top=1.5)
            plt.plot(self.solutions[self.getSolver(self.solver)]['timeSpan'], self.solutions[self.getSolver(self.solver)]['solutions'][i][:, 0], label='x(t)', color="blue", linewidth=0.5)
            plt.title(f'solution for x1 with respect to the initial condition: x1={round(self.solutions[self.getSolver(self.solver)]["x0s"][i],2)} , x2={round(self.solutions[self.getSolver(self.solver)]["y0s"][i],2)}', fontdict={"fontsize":10})

        for i in list(range(0,self.solutions[self.getSolver(self.solver)]["initialPointsCount"],int(self.solutions[self.getSolver(self.solver)]["initialPointsCount"]/self.solutions[self.getSolver(self.solver)]["individualSolutionsCount"]))):
            plt.subplot(int(f'{self.solutions[self.getSolver(self.solver)]["individualSolutionsCount"]}2{list(range(0,self.solutions[self.getSolver(self.solver)]["initialPointsCount"],int(self.solutions[self.getSolver(self.solver)]["initialPointsCount"]/self.solutions[self.getSolver(self.solver)]["individualSolutionsCount"]))).index(i) + 1 + len(list(range(0,self.solutions[self.getSolver(self.solver)]["initialPointsCount"],int(self.solutions[self.getSolver(self.solver)]["initialPointsCount"]/self.solutions[self.getSolver(self.solver)]["individualSolutionsCount"]))))}'))

            plt.subplots_adjust(top=1.5)
            plt.plot(self.solutions[self.getSolver(self.solver)]['timeSpan'], self.solutions[self.getSolver(self.solver)]['solutions'][i][:, 1], label='x(t)', color="blue", linewidth=0.5)
            plt.title(f'solution for x2 with respect to the initial condition: x1={round(self.solutions[self.getSolver(self.solver)]["x0s"][i],2)} , x2={round(self.solutions[self.getSolver(self.solver)]["y0s"][i],2)}', fontdict={"fontsize":10})

        plt.show()


    def getSolver(self, solver):
        solvers = {
            self._euler_solver: "euler",
            self._adams_moulton_solver: "adams-moulton",
            self._predictor_corrector_solver: "predictor-corrector",
            self._runge_kutta_second_order_solver: "runge-kutta-2",
            self._runge_kutta_forth_order_solver: "runge-kutta-4",
            self._adams_bashforth_2nd_order_solver: "adams-bashforth-2",
        }

        return solvers[solver]


    def _euler_solver(self, f, y0, t):
        num_eqs = len(y0)

        y = np.zeros((len(t), num_eqs))
        y[0, :] = y0

        h = t[1] - t[0]

        for i in range(1, len(t)):
            y[i, :] = y[i-1, :] + h * f(t[i-1], y[i-1, :])

        return y


    def _adams_moulton_solver(self, f, y0, t):
        num_eqs = len(y0)

        y = np.zeros((len(t), num_eqs))
        y[0, :] = y0

        h = t[1] - t[0]

        alpha = [1/2, 1/2]

        for i in range(1, len(t)):
            def system_to_solve(u):
                return u - y[i-1, :] - 0.5 * h * (f(t[i], u) + f(t[i-1], y[i-1, :]))

            initial_guess = y[i-1, :]

            y[i, :] = fsolve(system_to_solve, initial_guess)

        return y


    def _predictor_corrector_solver(self, f, y0, t):
        num_eqs = len(y0)

        y = np.zeros((len(t), num_eqs))
        y[0, :] = y0

        h = t[1] - t[0]

        for i in range(1, len(t)):
            y_pred = y[i-1, :] + h * f(t[i-1], y[i-1, :])

            def system_to_solve(u):
                return u - y[i-1, :] - h * (f(t[i], u) + f(t[i-1], y[i-1, :])) / 2

            initial_guess = y_pred
            y[i, :] = fsolve(system_to_solve, initial_guess)

        return y

    def _runge_kutta_forth_order_solver(self, f, y0, t):
        num_eqs = len(y0)

        y = np.zeros((len(t), num_eqs))
        y[0, :] = y0

        h = t[1] - t[0]

        a = np.array([1/6, 1/3, 1/3, 1/6])
        b = np.array([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]])
        c = np.array([0, 1/2, 1/2, 1])

        for i in range(1, len(t)):
            k = np.zeros((4, num_eqs))

            for j in range(4):
                k[j, :] = h * f(t[i-1] + c[j] * h, y[i-1, :] + np.dot(b[j, :], k))

            y[i, :] = y[i-1, :] + np.dot(a, k)

        return y


    def _runge_kutta_second_order_solver(self, f, y0, t):
        num_eqs = len(y0)

        y = np.zeros((len(t), num_eqs))
        y[0, :] = y0

        h = t[1] - t[0]

        a = np.array([1/2, 1/2])
        b = np.array([0, 1])
        for i in range(1, len(t)):
            k1 = h * f(t[i-1], y[i-1, :])
            k2 = h * f(t[i-1] + a[1] * h, y[i-1, :] + b[0] * k1)
            y[i, :] = y[i-1, :] + a[0] * k1 + a[1] * k2

        return y

    def _adams_bashforth_2nd_order_solver(self, f, y0, t):
        num_eqs = len(y0)

        y = np.zeros((len(t), num_eqs))
        y[0, :] = y0

        h = t[1] - t[0]

        alpha = np.array([-1, 3])

        for i in range(1, min(2, len(t))):
            k1 = h * f(t[i-1], y[i-1, :])
            k2 = h * f(t[i-1] + 0.5 * h, y[i-1, :] + 0.5 * k1)
            k3 = h * f(t[i-1] + 0.5 * h, y[i-1, :] + 0.5 * k2)
            k4 = h * f(t[i-1] + h, y[i-1, :] + k3)

            y[i, :] = y[i-1, :] + (k1 + 2*k2 + 2*k3 + k4) / 6

        for i in range(2, len(t)):
            y[i, :] = y[i-1, :] + h * 0.5 * sum(alpha[j-1] * f(t[i-j], y[i-j, :]) for j in range(1, 3))

        return y



