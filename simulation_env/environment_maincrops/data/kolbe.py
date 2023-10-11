kolbe_matrix = {
    'PRECROP': {
    0: [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, -2.0, -1.0],
    1: [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, -1.0, -1.0],
    2: [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, -2.0, -1.0, 2.0],
    3: [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, -1.0, 2.0],
    4: [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, -2.0, -1.0, -1.0],
    5: [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, -2.0, -1.0, 2.0],
    6: [1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, -1.0, 1.0, 2.0],
    7: [1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, -1.0, 1.0, 2.0],
    8: [-2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, -1.0, 1.0, 2.0],
    9: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0],
    10: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0],
    11: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
    12: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0],
    13: [2.0, 2.0, 1.0, 1.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    14: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0],
    15: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 2.0, -2.0, 2.0, 1.0],
    16: [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, -1.0, -1.0, -1.0, -2.0, 2.0, -2.0, 2.0, 1.0],
    17: [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, -2.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, 1.0, 1.0, -1.0, -1.0, -1.0, -2.0, 1.0, -2.0, 1.0, 1.0],
    18: [1.0, 1.0, 2.0, -1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0, -1.0, -1.0, -2.0, 2.0, 2.0, -1.0, 2.0, 2.0, -2.0, 2.0, -2.0, 1.0, -1.0],
    19: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -1.0, 1.0, -1.0],
    20: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, -2.0, 1.0, -2.0, -1.0, -1.0],
    21: [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0, -2.0, -2.0, -1.0],
    22: [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0]
    }
}