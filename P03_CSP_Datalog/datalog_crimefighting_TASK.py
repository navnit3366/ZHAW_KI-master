#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:38:21 2017

@author: tugg
"""
import pandas as pa
from pyDatalog import pyDatalog

calls = pa.read_csv('calls.csv', sep='\t', encoding='utf-8')
texts = pa.read_csv('texts.csv', sep='\t', encoding='utf-8')

suspect = 'Quandt Katarina'
company_Board = ['Soltau Kristine', 'Eder Eva', 'Michael Jill']

pyDatalog.create_terms('X', 'Y', 'Z', 'N', 'Count', 'L', 'P1', 'P2', 'P3')
pyDatalog.create_terms('Knows','Has_link', 'Daves_link', 'Path', 'Paths')
pyDatalog.clear()

# First treat calls simply as social links (denoted knows), which have no date
for i in range(0,150):
    +Knows(calls.iloc[i,1], calls.iloc[i,2])

# Knowing someone is a bi-directional relationship
Knows(X,Y) <= Knows(Y,X)

# Are there links between the company board and the suspect (i.e. does a path between the two exist)
# find all paths between the board members and the suspect
# Hints:
# if a knows b, there is a path between a and b
# (X._not_in(P2)) is used to check whether x is not in path P2
# (P==P2+[Z]) declares P as a new path containing P2 and Z

# Direct
Has_link(X,Y) <= Knows(X,Y)

# Transitively via Z
Has_link(X,Y) <= Knows(X,Z) & Has_link(Z,Y) & (X!=Y)

Daves_link() <= Has_link('Quandt Katarina', company_Board[1])
Daves_link() <= Has_link('Quandt Katarina', company_Board[2])

assert(Has_link('Quandt Katarina', company_Board[1]))
assert(Has_link(company_Board[1], 'Quandt Katarina'))
assert(Has_link(company_Board[2], 'Quandt Katarina'))
assert(Daves_link())

# there are so many path, therefore we are only interested in short paths.
# find all the paths between the suspect and the company board, which contain five poeple or less

Path(X, Y) <= Path(X, Y, 5)
Path(X, Y, Count) <= Knows(X, Y)
Path(X, Y, Count) <= (Count > 0) & Knows(X, Z) & Path(Z, Y, Count - 1)

assert(Path(company_Board[2], 'Quandt Katarina'))
assert(Path('Quandt Katarina', company_Board[1]))

# Direct paths involving Quandt:
# print(Path('Quandt Katarina', Y, 0))

Paths(X, Y) <= Knows(X, Y)
Paths(X, P1, Y) <= Knows(X, Y) & (P1 == '-')
Paths(X, P1, Y) <= Paths(X, P1) & Paths(P1, Y) & P1._not_in(X + Y)
Paths(X, P1, P2, Y) <= Knows(X, Y) & (P1 == '-') & (P2 == '-')
Paths(X, P1, P2, Y) <= Paths(X, P1, P2) & Paths(P2, Y) & P1._not_in(X + Y) & P2._not_in(X + Y)
Paths(X, P1, P2, P3, Y) <= Knows(X, Y) & (P1 == '-') & (P2 == '-') & (P3 == '-')
Paths(X, P1, P2, P3, Y) <= Paths(X, P1, P2, P3) & Paths(P3, Y) & P1._not_in(X + Y) & P2._not_in(X + Y) & P3._not_in(X + Y)

print(Paths('Quandt Katarina', P1, P2, P3, company_Board[1]))

# Now we use the text and the calls data together their corresponding dates
# Call-Data analysis
date_board_decision = '12.2.2017'
date_shares_bought = '23.2.2017'

# add terms to datalog
pyDatalog.create_terms('Called,Texted,Linked')
pyDatalog.clear()

# calls
for i in range(0,50):
    +Called(calls.iloc[i,1], calls.iloc[i,2],calls.iloc[i,3])

# texts
for i in range(0,50):
    +Texted(texts.iloc[i,1], texts.iloc[i,2],texts.iloc[i,3])


# calls are bi-directional
Called(X,Y,Z) <= Called(Y,X,Z)



# we are are again interested in links, but this time a connection only valid the links are descending in date
# find out who could have actually sent an information, when imposing this new restriction

# Linked(X) <= (X == ['asdf'])
# print(Linked([';lkj']))

# at last find all the communication paths which lead to the suspect, again with the restriction that the dates have to be ordered correctly


# after seeing this information, who, if anybody, do you think has given a tipp to the suspect?



