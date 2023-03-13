import numpy as np

from .cnn import CNNTemplate

CNNTemplateLib = {
    "hole_fill_template": CNNTemplate(
        A=np.array(
            [[0, 0.6, 0],
            [0.6, 4, 0.6],
            [0, 0.6, 0]]
        ),
        B=np.array(
            [[0, 0, 0],
            [0, 4, 0],
            [0, 0, 0]]
        ),
        I=-1
    ),

    "edge_detect_template": CNNTemplate(
        A=np.array(
            [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]
        ),
        B=np.array(
            [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]]
        ),
        I=-1
    ),

    "convex_corner_template": CNNTemplate(
        A=np.array(
            [[0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]]
        ),
        B=np.array(
            [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]]
        ),
        I=-8.5
    ),

    "set_difference_template": CNNTemplate(
        A=np.array(
            [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]
        ),
        B=np.array(
            [[0, 0, 0],
            [0, -1, 0],
            [0, 0, 0]]
        ),
        I=-1
    )
}
