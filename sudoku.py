
import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"

from copy import deepcopy
from math import trunc, exp
from random import choice
from statistics import pstdev


class Sudoku:

    def __init__(self, task, decreaseFactor):

        self.error = []
        self.decreaseFactor = decreaseFactor

        self.taskBoard = np.array([
            int(x.replace('\n', '')) for x in task.split(' ')
            if len(x) > 0 and x.replace('\n', '').isnumeric()]
        )
        self.n = int(np.power(len(self.taskBoard), 1 / 4))
        self.taskBoard = self.taskBoard.reshape(self.n ** 2, self.n ** 2)

        self.fixed = self.get_fixed()
        self.blocksCoords = self.get_blocks_coords()

        self.board = self.fill_blocks()

    def pretty_print(self):
        print("\n")
        for i in range(self.n ** 2):
            if i % self.n == 0:
                print("-" * ((self.n ** 2) * 3 + 2 + (self.n-1)*2))
            line = ""
            for j in range(self.n ** 2):
                if j % self.n == 0 and j != 0:
                    line += " | "
                line += '{: >2d}'.format(self.board[i, j]) + " "
            print(line)
        print("-" * ((self.n ** 2) * 3 + 2 + (self.n - 1) * 2))

    def plot_error_chart(self):
        df = pd.DataFrame({'total error':self.error})
        df['Error 100-step moving average'] = df.rolling(100).mean()
        df.plot(labels=dict(index='iterations', value='error')).show()
        
    def get_fixed(self):
        fixed = deepcopy(self.taskBoard)
        fixed[fixed != 0] = 1
        return fixed

    def get_blocks_coords(self):
        listOfBlocksCoords = []
        for i in range(0, self.n ** 2):
            tmp = []
            for x in [j + self.n * (i % self.n) for j in range(0, self.n)]:
                for y in [j + self.n * trunc(i / self.n) for j in
                          range(0, self.n)]:
                    tmp.append([x, y])
            listOfBlocksCoords.append(tmp)
        return listOfBlocksCoords

    @staticmethod
    def get_block(blockCoords, board):
        return board[
           blockCoords[0][0]:(blockCoords[-1][0] + 1),
           blockCoords[0][1]:(blockCoords[-1][1] + 1)
        ]

    def fill_blocks(self):
        board = deepcopy(self.taskBoard)
        for blockCoords in self.blocksCoords:
            block = self.get_block(blockCoords, board)
            for coords in blockCoords:
                if board[coords[0], coords[1]] == 0:
                    board[coords[0], coords[1]] = \
                        choice(np.setdiff1d(
                            range(1, self.n ** 2 + 1),
                            np.unique(block)
                        ))
        return board

    def get_box_error(self, board, x, y):
        return ((self.n**2 - len(np.unique(board[x, :]))) +
                (self.n**2 - len(np.unique(board[:, y]))))

    def get_total_error(self, board):
        return np.sum([
            self.get_box_error(board, i, i)
            for i in range(0, self.n ** 2)
        ])

    def get_candidates(self, bIdx):
        block = self.blocksCoords[bIdx]
        box1 = choice(
            list(filter(lambda x: self.fixed[x[0], x[1]] == 0, block))
        )
        box2 = choice(
            list(
                filter(
                    lambda x: self.fixed[x[0], x[1]] == 0 and x != box1, block
                )
            )
        )
        return box1, box2

    def get_candidate_solution(self):
        bIdx = np.random.randint(0, self.n ** 2)
        b1, b2 = self.get_candidates(bIdx)
        newSolution = deepcopy(self.board)
        tmp = newSolution[b1[0], b1[1]]
        newSolution[b1[0], b1[1]] = newSolution[b2[0], b2[1]]
        newSolution[b2[0], b2[1]] = tmp
        return newSolution, b1, b2

    def step(self, sigma):
        newSolution, b1, b2 = self.get_candidate_solution()
        currentCost = self.get_box_error(self.board, b1[0], b1[1]) + \
                      self.get_box_error(self.board, b2[0], b2[1])
        newCost = self.get_box_error(newSolution, b1[0], b1[1]) + \
                  self.get_box_error(newSolution, b2[0], b2[1])
        costDifference = newCost - currentCost
        rho = exp(-costDifference / sigma)
        if np.random.uniform(1, 0, 1) < rho:
            self.board = newSolution
            return costDifference
        return 0

    def heuristic(self):
        listOfDifferences = []
        if self.get_total_error(self.board) == 0:
            return
        for _ in range(100):
            listOfDifferences.append(
                self.get_total_error(self.get_candidate_solution()[0])
            )
        sigma = pstdev(listOfDifferences) + 10e-5
        score = self.get_total_error(self.board)
        iters = len(self.fixed[self.fixed == 1])
        stuckCount = 0

        epochs = 0
        solutionFound = False
        while not solutionFound:
            epochs += 1
            previousScore = score
            for _ in range(0, iters):
                scoreDiff = self.step(sigma)
                score += scoreDiff
                self.error.append(self.get_total_error(self.board))
            sigma *= self.decreaseFactor
            if epochs % 10 == 0:
                if self.get_total_error(self.board) == 0:
                    break
            if score >= previousScore:
                stuckCount += 1
            else:
                stuckCount = 0
            if stuckCount > 80:
                sigma += 2
