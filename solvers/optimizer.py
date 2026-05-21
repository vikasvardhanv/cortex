"""
Genetic Algorithm Optimizer for Cortex CEM

Multi-objective optimization using evolutionary algorithms
for engineering design optimization.

Supports:
- Single and multi-objective optimization
- NSGA-II for Pareto-optimal solutions
- Constraint handling via penalty functions
- Real-valued and discrete parameters
- Integration with physics solvers

Applications:
- Topology optimization
- Shape optimization
- Material selection
- Process parameter tuning
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class DesignVariable:
    """Definition of a design variable."""
    name: str
    lower_bound: float
    upper_bound: float
    var_type: str = "continuous"  # "continuous", "integer", "discrete"
    discrete_values: Optional[List[float]] = None

    def sample(self, rng: np.random.Generator = None) -> float:
        """Sample a random value for this variable."""
        if rng is None:
            rng = np.random.default_rng()

        if self.var_type == "discrete" and self.discrete_values:
            return rng.choice(self.discrete_values)
        elif self.var_type == "integer":
            return float(rng.integers(int(self.lower_bound), int(self.upper_bound) + 1))
        else:
            return rng.uniform(self.lower_bound, self.upper_bound)

    def clip(self, value: float) -> float:
        """Clip value to bounds."""
        if self.var_type == "discrete" and self.discrete_values:
            # Find nearest discrete value
            return min(self.discrete_values, key=lambda x: abs(x - value))
        elif self.var_type == "integer":
            return float(round(np.clip(value, self.lower_bound, self.upper_bound)))
        else:
            return np.clip(value, self.lower_bound, self.upper_bound)


@dataclass
class Constraint:
    """Definition of an optimization constraint."""
    name: str
    func: Callable[[np.ndarray], float]  # g(x) <= 0 for feasible
    constraint_type: str = "inequality"  # "inequality" or "equality"
    tolerance: float = 1e-6  # For equality constraints

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate constraint value."""
        return self.func(x)

    def is_satisfied(self, x: np.ndarray) -> bool:
        """Check if constraint is satisfied."""
        g = self.evaluate(x)
        if self.constraint_type == "equality":
            return abs(g) <= self.tolerance
        else:
            return g <= 0


@dataclass
class OptimizationProblem:
    """
    Optimization problem definition.

    Example:
        problem = OptimizationProblem(
            name="nozzle_design",
            variables=[
                DesignVariable("throat_radius", 5.0, 15.0),
                DesignVariable("expansion_ratio", 1.5, 4.0),
                DesignVariable("wall_thickness", 1.0, 5.0),
            ],
            objectives=[
                lambda x: evaluate_thrust(x),     # Maximize (negate for min)
                lambda x: evaluate_mass(x),       # Minimize
            ],
            constraints=[
                Constraint("stress_limit", lambda x: max_stress(x) - 500e6),
                Constraint("min_thickness", lambda x: 1.0 - x[2]),
            ]
        )
    """
    name: str
    variables: List[DesignVariable]
    objectives: List[Callable[[np.ndarray], float]]  # All to be minimized
    constraints: List[Constraint] = field(default_factory=list)
    objective_names: List[str] = field(default_factory=list)

    @property
    def n_var(self) -> int:
        return len(self.variables)

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    @property
    def n_con(self) -> int:
        return len(self.constraints)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds as arrays."""
        lb = np.array([v.lower_bound for v in self.variables])
        ub = np.array([v.upper_bound for v in self.variables])
        return lb, ub


@dataclass
class Individual:
    """An individual in the population."""
    x: np.ndarray  # Design variables
    objectives: Optional[np.ndarray] = None  # Objective values
    constraints: Optional[np.ndarray] = None  # Constraint values
    rank: int = 0  # Pareto rank (NSGA-II)
    crowding_distance: float = 0.0  # Crowding distance (NSGA-II)
    feasible: bool = True

    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another (Pareto)."""
        if self.objectives is None or other.objectives is None:
            return False

        # Must be better in at least one objective and not worse in any
        better_in_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_one = True

        return better_in_one


@dataclass
class OptimizationResult:
    """Result from optimization."""
    problem_name: str
    algorithm: str
    generations: int
    evaluations: int
    solve_time: float

    # Best solutions
    best_x: np.ndarray  # Best design (single-objective)
    best_objectives: np.ndarray

    # Pareto front (multi-objective)
    pareto_x: Optional[np.ndarray] = None  # [n_solutions, n_var]
    pareto_objectives: Optional[np.ndarray] = None  # [n_solutions, n_obj]

    # History
    history: Optional[List[Dict]] = None

    def summary(self) -> str:
        lines = [
            f"=== Optimization Result: {self.problem_name} ===",
            f"Algorithm: {self.algorithm}",
            f"Generations: {self.generations}",
            f"Evaluations: {self.evaluations}",
            f"Solve time: {self.solve_time:.2f} s",
            "",
            "Best Solution:",
            f"  Design: {self.best_x}",
            f"  Objectives: {self.best_objectives}",
        ]

        if self.pareto_x is not None:
            lines.append(f"\nPareto front: {len(self.pareto_x)} solutions")

        return "\n".join(lines)


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer.

    Implements:
    - Tournament selection
    - SBX crossover (Simulated Binary Crossover)
    - Polynomial mutation
    - Elitism
    """

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta_c: float = 20.0,  # SBX distribution index
        eta_m: float = 20.0,  # Mutation distribution index
        tournament_size: int = 2,
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

    def optimize(self, problem: OptimizationProblem,
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """Run the genetic algorithm optimization."""
        start_time = time.time()

        # Initialize population
        population = self._initialize_population(problem)

        # Evaluate initial population
        self._evaluate_population(population, problem)
        evaluations = len(population)

        history = []
        best_individual = min(population, key=lambda ind: ind.objectives[0] if ind.feasible else float('inf'))

        for gen in range(self.max_generations):
            # Selection
            parents = self._tournament_selection(population)

            # Crossover
            offspring = self._sbx_crossover(parents, problem)

            # Mutation
            offspring = self._polynomial_mutation(offspring, problem)

            # Evaluate offspring
            self._evaluate_population(offspring, problem)
            evaluations += len(offspring)

            # Combine and select next generation (elitism)
            combined = population + offspring
            population = self._select_survivors(combined)

            # Track best
            gen_best = min(population, key=lambda ind: ind.objectives[0] if ind.feasible else float('inf'))
            if gen_best.feasible and (not best_individual.feasible or
                                       gen_best.objectives[0] < best_individual.objectives[0]):
                best_individual = gen_best

            # Record history
            history.append({
                "generation": gen,
                "best_objective": best_individual.objectives[0] if best_individual.feasible else None,
                "avg_objective": np.mean([ind.objectives[0] for ind in population if ind.feasible]),
                "feasible_ratio": sum(1 for ind in population if ind.feasible) / len(population),
            })

            if callback:
                callback(gen, population, best_individual)

        solve_time = time.time() - start_time

        return OptimizationResult(
            problem_name=problem.name,
            algorithm="GeneticAlgorithm",
            generations=self.max_generations,
            evaluations=evaluations,
            solve_time=solve_time,
            best_x=best_individual.x,
            best_objectives=best_individual.objectives,
            history=history,
        )

    def _initialize_population(self, problem: OptimizationProblem) -> List[Individual]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            x = np.array([var.sample(self.rng) for var in problem.variables])
            population.append(Individual(x=x))
        return population

    def _evaluate_population(self, population: List[Individual],
                             problem: OptimizationProblem):
        """Evaluate objectives and constraints for all individuals."""
        for ind in population:
            # Evaluate objectives
            ind.objectives = np.array([obj(ind.x) for obj in problem.objectives])

            # Evaluate constraints
            if problem.constraints:
                ind.constraints = np.array([c.evaluate(ind.x) for c in problem.constraints])
                ind.feasible = all(c.is_satisfied(ind.x) for c in problem.constraints)
            else:
                ind.feasible = True

    def _tournament_selection(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection for mating pool."""
        parents = []
        for _ in range(self.population_size):
            # Select tournament participants
            participants = self.rng.choice(population, size=self.tournament_size, replace=False)

            # Select winner (prefer feasible, then better objective)
            winner = min(participants, key=lambda ind: (
                not ind.feasible,
                ind.objectives[0] if ind.objectives is not None else float('inf')
            ))
            parents.append(winner)

        return parents

    def _sbx_crossover(self, parents: List[Individual],
                       problem: OptimizationProblem) -> List[Individual]:
        """Simulated Binary Crossover (SBX)."""
        offspring = []
        lb, ub = problem.bounds()

        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]

            if self.rng.random() < self.crossover_prob:
                c1_x = np.zeros(problem.n_var)
                c2_x = np.zeros(problem.n_var)

                for j in range(problem.n_var):
                    if self.rng.random() < 0.5:
                        if abs(p1.x[j] - p2.x[j]) > 1e-10:
                            if p1.x[j] < p2.x[j]:
                                y1, y2 = p1.x[j], p2.x[j]
                            else:
                                y1, y2 = p2.x[j], p1.x[j]

                            # Calculate beta
                            beta = 1.0 + (2.0 * (y1 - lb[j]) / (y2 - y1))
                            alpha = 2.0 - beta ** (-(self.eta_c + 1))
                            rand = self.rng.random()

                            if rand <= 1.0 / alpha:
                                betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1))
                            else:
                                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1))

                            c1_x[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                            c2_x[j] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                            c1_x[j] = problem.variables[j].clip(c1_x[j])
                            c2_x[j] = problem.variables[j].clip(c2_x[j])
                        else:
                            c1_x[j] = p1.x[j]
                            c2_x[j] = p2.x[j]
                    else:
                        c1_x[j] = p1.x[j]
                        c2_x[j] = p2.x[j]

                offspring.append(Individual(x=c1_x))
                offspring.append(Individual(x=c2_x))
            else:
                offspring.append(Individual(x=p1.x.copy()))
                offspring.append(Individual(x=p2.x.copy()))

        return offspring

    def _polynomial_mutation(self, offspring: List[Individual],
                             problem: OptimizationProblem) -> List[Individual]:
        """Polynomial mutation."""
        lb, ub = problem.bounds()

        for ind in offspring:
            for j in range(problem.n_var):
                if self.rng.random() < self.mutation_prob:
                    y = ind.x[j]
                    delta1 = (y - lb[j]) / (ub[j] - lb[j])
                    delta2 = (ub[j] - y) / (ub[j] - lb[j])

                    rand = self.rng.random()
                    mut_pow = 1.0 / (self.eta_m + 1)

                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1))
                        deltaq = 1.0 - val ** mut_pow

                    y = y + deltaq * (ub[j] - lb[j])
                    ind.x[j] = problem.variables[j].clip(y)

        return offspring

    def _select_survivors(self, combined: List[Individual]) -> List[Individual]:
        """Select survivors for next generation (elitism)."""
        # Sort by feasibility and objective value
        combined.sort(key=lambda ind: (
            not ind.feasible,
            ind.objectives[0] if ind.objectives is not None else float('inf')
        ))
        return combined[:self.population_size]


class NSGA2(GeneticAlgorithm):
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.

    For multi-objective optimization with Pareto-optimal solutions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def optimize(self, problem: OptimizationProblem,
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """Run NSGA-II optimization."""
        start_time = time.time()

        # Initialize population
        population = self._initialize_population(problem)
        self._evaluate_population(population, problem)
        evaluations = len(population)

        # Initial ranking
        self._fast_non_dominated_sort(population)
        self._calculate_crowding_distance(population)

        history = []

        for gen in range(self.max_generations):
            # Create offspring
            parents = self._binary_tournament_selection(population)
            offspring = self._sbx_crossover(parents, problem)
            offspring = self._polynomial_mutation(offspring, problem)

            # Evaluate offspring
            self._evaluate_population(offspring, problem)
            evaluations += len(offspring)

            # Combine populations
            combined = population + offspring

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined)

            # Select next generation
            population = []
            for front in fronts:
                if len(population) + len(front) <= self.population_size:
                    self._calculate_crowding_distance(front)
                    population.extend(front)
                else:
                    # Need to select from this front
                    self._calculate_crowding_distance(front)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.population_size - len(population)
                    population.extend(front[:remaining])
                    break

            # Track Pareto front
            pareto_front = [ind for ind in population if ind.rank == 0]

            history.append({
                "generation": gen,
                "pareto_size": len(pareto_front),
                "hypervolume": self._calculate_hypervolume(pareto_front) if problem.n_obj == 2 else None,
            })

            if callback:
                callback(gen, population, pareto_front)

        solve_time = time.time() - start_time

        # Extract Pareto front
        pareto_front = [ind for ind in population if ind.rank == 0]
        pareto_x = np.array([ind.x for ind in pareto_front])
        pareto_objectives = np.array([ind.objectives for ind in pareto_front])

        # Best solution (smallest first objective)
        best_idx = np.argmin(pareto_objectives[:, 0])

        return OptimizationResult(
            problem_name=problem.name,
            algorithm="NSGA-II",
            generations=self.max_generations,
            evaluations=evaluations,
            solve_time=solve_time,
            best_x=pareto_x[best_idx],
            best_objectives=pareto_objectives[best_idx],
            pareto_x=pareto_x,
            pareto_objectives=pareto_objectives,
            history=history,
        )

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting (NSGA-II)."""
        fronts = [[]]
        S = {i: [] for i in range(len(population))}  # Solutions dominated by i
        n = {i: 0 for i in range(len(population))}   # Domination count of i

        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if p.dominates(q):
                        S[i].append(j)
                    elif q.dominates(p):
                        n[i] += 1

            if n[i] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p_idx, p in enumerate(population):
                if p in fronts[i]:
                    orig_idx = population.index(p)
                    for q_idx in S[orig_idx]:
                        n[q_idx] -= 1
                        if n[q_idx] == 0:
                            population[q_idx].rank = i + 1
                            next_front.append(population[q_idx])
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for a Pareto front."""
        n = len(front)
        if n == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return

        n_obj = len(front[0].objectives)

        for m in range(n_obj):
            # Sort by objective m
            front.sort(key=lambda x: x.objectives[m])

            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Objective range
            f_max = front[-1].objectives[m]
            f_min = front[0].objectives[m]

            if f_max - f_min > 1e-10:
                for i in range(1, n - 1):
                    front[i].crowding_distance += (
                        (front[i + 1].objectives[m] - front[i - 1].objectives[m])
                        / (f_max - f_min)
                    )

    def _binary_tournament_selection(self, population: List[Individual]) -> List[Individual]:
        """Binary tournament selection based on rank and crowding distance."""
        parents = []
        for _ in range(self.population_size):
            i, j = self.rng.choice(len(population), size=2, replace=False)
            p1, p2 = population[i], population[j]

            # Select based on rank, then crowding distance
            if p1.rank < p2.rank:
                winner = p1
            elif p2.rank < p1.rank:
                winner = p2
            elif p1.crowding_distance > p2.crowding_distance:
                winner = p1
            else:
                winner = p2

            parents.append(winner)

        return parents

    def _calculate_hypervolume(self, front: List[Individual],
                               ref_point: Optional[np.ndarray] = None) -> float:
        """Calculate hypervolume indicator for 2D Pareto front."""
        if not front or front[0].objectives is None:
            return 0.0

        objectives = np.array([ind.objectives for ind in front])

        if ref_point is None:
            ref_point = np.max(objectives, axis=0) * 1.1

        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_indices]

        # Calculate hypervolume (2D only)
        hv = 0.0
        prev_y = ref_point[1]

        for obj in sorted_obj:
            if obj[1] < prev_y:
                hv += (ref_point[0] - obj[0]) * (prev_y - obj[1])
                prev_y = obj[1]

        return hv


class DifferentialEvolution:
    """
    Differential Evolution optimizer.

    Good for continuous optimization with box constraints.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        F: float = 0.8,  # Differential weight
        CR: float = 0.9,  # Crossover probability
        strategy: str = "best/1/bin",
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def optimize(self, problem: OptimizationProblem,
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """Run Differential Evolution optimization."""
        start_time = time.time()
        lb, ub = problem.bounds()

        # Initialize population
        population = lb + (ub - lb) * self.rng.random((self.population_size, problem.n_var))
        fitness = np.array([problem.objectives[0](x) for x in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        history = []

        for gen in range(self.max_generations):
            for i in range(self.population_size):
                # Mutation
                if "best" in self.strategy:
                    base = population[best_idx]
                else:
                    base = population[self.rng.integers(self.population_size)]

                # Select random individuals for difference vector
                idxs = [j for j in range(self.population_size) if j != i]
                r1, r2 = self.rng.choice(idxs, size=2, replace=False)

                mutant = base + self.F * (population[r1] - population[r2])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                cross_points = self.rng.random(problem.n_var) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(problem.n_var)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_f = problem.objectives[0](trial)
                evaluations += 1

                if trial_f < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f

                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f

            history.append({
                "generation": gen,
                "best_objective": best_f,
                "avg_objective": np.mean(fitness),
            })

            if callback:
                callback(gen, population, best_x, best_f)

        solve_time = time.time() - start_time

        return OptimizationResult(
            problem_name=problem.name,
            algorithm="DifferentialEvolution",
            generations=self.max_generations,
            evaluations=evaluations,
            solve_time=solve_time,
            best_x=best_x,
            best_objectives=np.array([best_f]),
            history=history,
        )


class DesignOptimizer:
    """
    High-level interface for engineering design optimization.

    Integrates with Cortex solvers for physics-based optimization.
    """

    def __init__(self, algorithm: str = "nsga2", **kwargs):
        if algorithm == "nsga2":
            self.optimizer = NSGA2(**kwargs)
        elif algorithm == "ga":
            self.optimizer = GeneticAlgorithm(**kwargs)
        elif algorithm == "de":
            self.optimizer = DifferentialEvolution(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def optimize_design(
        self,
        problem: OptimizationProblem,
        parallel_evaluations: int = 1,
        callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Optimize a design problem.

        Args:
            problem: Optimization problem definition
            parallel_evaluations: Number of parallel objective evaluations
            callback: Optional callback(generation, population, best)

        Returns:
            OptimizationResult with best solutions
        """
        return self.optimizer.optimize(problem, callback=callback)

    @staticmethod
    def create_thermal_objective(
        solver,
        problem_template,
        target_temp: float,
        param_map: Dict[str, int],
    ) -> Callable:
        """
        Create an objective function for thermal optimization.

        Args:
            solver: ThermalSolver instance
            problem_template: Base ThermalProblem to modify
            target_temp: Target temperature to minimize deviation from
            param_map: Mapping from variable index to problem attribute

        Returns:
            Objective function that evaluates thermal performance
        """
        def objective(x: np.ndarray) -> float:
            # Create modified problem
            import copy
            problem = copy.deepcopy(problem_template)

            # Apply design variables
            for attr, idx in param_map.items():
                setattr(problem, attr, x[idx])

            # Solve
            result = solver.solve(problem)

            # Calculate objective (deviation from target)
            return abs(result.T_avg - target_temp)

        return objective

    @staticmethod
    def create_structural_objective(
        solver,
        problem_template,
        max_stress: float,
        param_map: Dict[str, int],
    ) -> Callable:
        """
        Create an objective function for structural optimization.

        Minimizes mass while constraining stress.
        """
        def objective(x: np.ndarray) -> float:
            import copy
            problem = copy.deepcopy(problem_template)

            for attr, idx in param_map.items():
                setattr(problem, attr, x[idx])

            result = solver.solve(problem)

            # Objective: minimize max displacement
            # Constraint penalty: stress should be < max_stress
            penalty = max(0, result.max_stress - max_stress) * 1e-6

            return result.max_displacement + penalty

        return objective
