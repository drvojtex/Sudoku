# Sudoku (heuristic solver)
## Overview

This repository contains a heuristic solver based on simulated annealing for a sudoku puzzle.

> [sudoku.py](https://github.com/drvojtex/Sudoku/blob/master/sudoku.py) - solver class

> [main.py](https://github.com/drvojtex/Sudoku/blob/master/main.py) - example evaluation

> [explore.py](https://github.com/drvojtex/Sudoku/blob/master/explore.py) - script to explore time depandance on a decreasing factor (hyperparameter of the heuristic)

## Method

### Initialise
- Mark boxes filled in the task as fixed.
- Fill all boxes uniformly such in each block of sudoku are unique numbers according to the rules, violated are only uniqueness of rows and columns numbers.

### Iterate
- Uniformly select two boxes in the same block; neither of them is fixed. 
- For each box compute cost function in a way "2*n - unique(row) - unique(column)", where 'n' is a size of Sudoku. Sum these two costs and mark result as a "previous cost".
- Swap selected boxes.
- Compute again costs and mark as a "new cost".
- Compute rho = exp(-costDifference / sigma); sigma will be discussed later.
- If rho is greater than uniformly random number between 0 and 1, return new Sudoku (with swaped boxes), else return the previous on.
- Iterate until the Sudoku contains no violations of row or columns uniqueness.

### Sigma parameter
- Select hyperparameter "decreasing factor".
- Generate approx. 100 candidate swaps (without changind the Sudoku) and after each compute total error of Sudoku. From this error array compute standard devitation; mark as "d".
- sigma = d + epsilon, where epsilon is approx. 10e-5 (to avoid zero division).
- Mark count of fixed boxes as "iters".
- While iterating (in previous paragraph), after each "iters" iterations multiply "sigma" by "decreasing factor"
- If there is no change of Sudoku after 800 iterarions, increase sigma by two.
 
## Results
After evaluations the method for different hyperparameter setup (50 times for each setup) on 9x9 Sudoku, the lowes median time on Macbook M1 is for the "decreasing factor" 0.98855 with the median time 0.196549s. Results of exploration of the hyperparameter shows Figure below.
![Decreasing factor exploration](https://github.com/drvojtex/Sudoku/blob/master/exploration.png)

## Resources
Lewis, R. (2007). Metaheuristics can solve Sudoku puzzles. J. Heuristics, 13, 387-401.

challengingLuck. (2020, October 2). Youtube/sudoku.py at master · Challengingluck/youtube. GitHub. Retrieved April 18, 2023, from https://github.com/challengingLuck/youtube/blob/master/sudoku/sudoku.py 



