#!/usr/bin/env python

from tfidf_matcher import ngrams
from tfidf_matcher import matcher
import pandas as pd

# SCRATCH
games = pd.read_table('~/Downloads/games.txt')
games = games.iloc[3:,:]
games.columns = ['name']
games = games.name.str.split(' \\(', expand = True)
games = pd.concat([games, games[1].str.split(', ', expand = True)], axis = 1)
games.columns = [str(x) for x in range(0, len(games.columns))]
games = games.iloc[:,[0, 5, 6]]

import random
chosen = random.sample(range(0, len(games)), 50)
games_original = games.iloc[chosen, :]
games_lookup = games[[x not in chosen for x in range(0, len(games))]]

matcher.matcher(games_original[0], games_lookup[0], k_matches = 3)
