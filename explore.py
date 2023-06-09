
import plotly.express as px
import pandas as pd
from numpy import logspace
from time import process_time
from sudoku import Sudoku


if __name__ == "__main__":

    s1 = """
        0 2 4 0 0 7 0 0 0
        6 0 0 0 0 0 0 0 0
        0 0 3 6 8 0 4 1 5
        4 3 1 0 0 5 0 0 0
        5 0 0 0 0 0 0 3 2
        7 9 0 0 0 0 0 6 0
        2 0 9 7 1 0 8 0 0
        0 4 0 0 9 3 0 0 0
        3 1 0 0 0 4 7 5 0
    """

    s2 = """
        0 15 0 1 0 2 10 14 12 0 0 0 0 0 0 0
        0 6 3 16 12 0 8 4 14 15 1 0 2 0 0 0
        14 0 9 7 11 3 15 0 0 0 0 0 0 0 0 0
        4 13 2 12 0 0 0 0 6 0 0 0 0 15 0 0
        9 2 6 15 14 1 11 7 3 5 10 0 4 8 13 12
        3 16 0 0 2 4 0 0 0 14 7 13 0 0 5 15
        11 0 5 0 0 0 0 0 0 9 4 0 0 6 0 0
        0 0 0 0 13 0 16 5 15 0 0 12 0 0 0 0
        0 0 0 0 9 0 1 12 0 8 3 10 11 0 15 0
        2 12 0 11 0 0 14 3 5 4 0 0 0 0 9 0
        6 3 0 4 0 0 13 0 0 11 9 1 0 12 16 2
        0 0 10 9 0 0 0 0 0 0 12 0 8 0 6 7
        12 8 0 0 16 0 0 10 0 13 0 0 0 5 0 0
        5 0 0 0 3 0 4 6 0 1 15 0 0 0 0 0
        0 9 1 6 0 14 0 11 0 0 2 0 0 0 10 8
        0 14 0 0 0 13 9 0 4 12 11 8 0 0 2 0
    """

    df = pd.DataFrame({"Decrease factor": [], "Time": []})
    for i in logspace(-0.001, -0.0055, num=10):
        for _ in range(50):
            t = process_time()
            s = Sudoku(s1, i)
            s.heuristic()
            df.loc[len(df.index)] = [i, process_time() - t]

    fig = px.box(df, x="Decrease factor", y="Time")
    fig.show()
