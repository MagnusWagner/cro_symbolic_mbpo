cropbreaks = {
    0: {
        "AP": 4,
        "AP Groups": ["BLATT", "FL", "L"],
        "MF Groups": [],
        "crop": "KLEEGRAS",
    },
    1: {
        "AP": 5,
        "AP Groups": ["BLATT", "FL", "L"],
        "MF Groups": [],
        "crop": "LUZERNE",
    },
    2: {"AP": 3, "AP Groups": ["BLATT", "L"], "MF Groups": [], "crop": "ACKERBOHNE"},
    3: {"AP": 5, "AP Groups": ["BLATT", "L"], "MF Groups": [], "crop": "KÖRNERERBSE"},
    4: {"AP": 3, "AP Groups": ["BLATT", "L"], "MF Groups": [], "crop": "SOJABOHNE"},
    5: {
        "AP": 4,
        "AP Groups": ["BLATT", "L"],
        "MF Groups": [],
        "crop": "FUTTERLUPINE",
    },
    6: {
        "AP": 2,
        "AP Groups": ["Weizen/Dinkel/Triticale", "Weizen"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)", "Weizen/Triticale"],
        "crop": "WINTERWEIZEN",
    },
    7: {
        "AP": 2,
        "AP Groups": ["Weizen/Dinkel/Triticale", "Weizen"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)", "Weizen/Triticale"],
        "crop": "SOMMERWEIZEN",
    },
    8: {
        "AP": 2,
        "AP Groups": ["Weizen/Dinkel/Triticale", "Weizen"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)", "Weizen/Triticale"],
        "crop": "WINTERHARTWEIZEN",
    },
    9: {
        "AP": 2,
        "AP Groups": ["Weizen/Dinkel/Triticale"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)"],
        "crop": "WINTERDINKEL",
    },
    10: {
        "AP": 2,
        "AP Groups": ["Weizen/Dinkel/Triticale"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)", "Weizen/Triticale"],
        "crop": "WINTERTRITICALE",
    },
    11: {
        "AP": 1,
        "AP Groups": [],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)"],
        "crop": "WINTERROGGEN",
    },
    12: {
        "AP": 2,
        "AP Groups": ["Gerste"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)"],
        "crop": "WINTERGERSTE",
    },
    13: {
        "AP": 2,
        "AP Groups": ["Gerste"],
        "MF Groups": ["GE", "GE (ohne Mais/Hafer/Hirse)"],
        "crop": "SOMMERGERSTE",
    },
    14: {"AP": 4, "AP Groups": [], "MF Groups": ["GE"], "crop": "SOMMERHAFER"},
    15: {"AP": 2, "AP Groups": [], "MF Groups": ["GE"], "crop": "HIRSE"},
    16: {
        "AP": 1,
        "AP Groups": ["BLATT", "Mais"],
        "MF Groups": [],
        "crop": "SILOMAIS",
    },
    17: {"AP": 1, "AP Groups": ["Mais"], "MF Groups": ["GE"], "crop": "KÖRNERMAIS"},
    18: {
        "AP": 4,
        "AP Groups": ["BLATT", "Rüben/Kruziferen"],
        "MF Groups": [],
        "crop": "ZUCKERRÜBEN",
    },
    19: {"AP": 3, "AP Groups": ["BLATT"], "MF Groups": [], "crop": "KARTOFFELN"},
    20: {
        "AP": 3,
        "AP Groups": ["Raps/Sonnenblume", "Rüben/Kruziferen"],
        "MF Groups": [],
        "crop": "WINTERRAPS",
    },
    21: {
        "AP": 3,
        "AP Groups": ["Raps/Sonnenblume"],
        "MF Groups": [],
        "crop": "SONNENBLUMEN",
    },
    22: {"AP": 2, "AP Groups": [], "MF Groups": [], "crop": "ÖLKÜRBIS"},
}

mf_groups = {
    "group_indices": {
        "GE": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17],
        "GE (ohne Mais/Hafer/Hirse)": [6, 7, 8, 9, 10, 11, 12, 13],
        "Weizen/Triticale": [6, 7, 8, 10],
    },
    "max_frequencies": {
        "GE": (3, 4),
        "GE (ohne Mais/Hafer/Hirse)": (2, 3),
        "Weizen/Triticale": (1, 3),
    }
}

ap_groups = {
    "group_indices": {
        "BLATT": [0, 1, 2, 3, 4, 5, 16, 18, 19],
        "FL": [0, 1],
        "L": [0, 1, 2, 3, 4, 5],
        "Weizen/Dinkel/Triticale": [6, 7, 8, 9, 10],
        "Raps/Sonnenblume": [20, 21],
        "Mais": [16, 17],
        "Rüben/Kruziferen": [18, 20],
        "Weizen": [18, 20],
        "Gerste": [12, 13],
    },
    "min_breaks": {
        "BLATT": 1,
        "FL": 5,
        "L": 4,
        "Weizen/Dinkel/Triticale": 1,
        "Raps/Sonnenblume": 2,
        "Mais": 1,
        "Rüben/Kruziferen": 3,
        "Weizen": 2,
        "Gerste": 2,
    }
}
