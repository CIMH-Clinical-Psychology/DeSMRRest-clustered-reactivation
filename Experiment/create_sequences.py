# -*- coding: utf-8 -*-
"""
This file is used to create the sequences for the experiment
"Decoding of sequential memory reactivation from wake rest"

There are 16 items which are ordered on a complex grid with 4 gridpoints
which have two exit nodes. This is the example structure of the grid:
    Note: vertical bars are also directional, but there is no ascii arrow up
 

Seq16: ABCDEBFGHIFJKLMJANOP

    L <- K    P <- O
    |    ↑    ↓    ↑
    M -> J -> A -> N
         ↓    ↓
    G <- F <- B <- E
    ↓    ↑    ↓    ↑
    H -> I    C -> D

    12<- 11   16 <-15
    |    ↑    ↓    ↑
    13-> 10-> 1 -> 14
         ↓    ↓
    7 <- 6 <- 2 <- 5
    ↓    ↑    ↓    ↑
    8 -> 9    3 -> 4


Seq12: ABCDEFGEHIBJ
    A       C -> D      F
    | \   /       \   / |
    |   B           E   |
    | /   \       /   \ |
    J       I <- H      G

Seq10: ABCDEFGHIJ
    A -> B -> C -> D -> E
    |                   |
    J <- I <- H <- G <- H 



We will create N sequences of 5 items, where each 5-tuple consists of
two items that are in succession and 3 items that are the choice, of which
1 is the correct sequence, 1 is a close neighbour and 1 is an item far away
on the grid.

Examples:
    A - B - E - C - J, correct -> C
    E - B - G - O - F, correct -> F

"""
import os
import more_itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
np.random.seed(0)
random.seed(0)



def de_bruijn_2tuple(items):
    """
    finds the debruijn-sequence for a given alphabet, where there are
    no self connections within the graph
    """
    if isinstance(items, int): items = np.arange(1, items+1)
    atoms = set(items)

    max_tries = 1000
    n_block = 14 # block +- this surrounding each gray square
    tries = 0
    while tries<max_tries:
        try:
            transitions = {x:[y for y in atoms if not y==x] for x in atoms}
            seq1 = np.random.choice(list(transitions))
            sequence = []
            while len(transitions)>0:
                seq2 = np.random.choice(transitions[seq1])
                sequence.append(seq2)
                transitions[seq1].remove(seq2)
                if len(transitions[seq1])==0:
                    transitions.pop(seq1)
                seq1=seq2
        except:
            tries+=1
            continue
        sequence_str = [str(x) for x in sequence]
        _, counts = np.unique(sequence, return_counts=True)
        assert len(set(counts))==1, f'ERROR, unequal number of items {counts}'
        # max(tqdm._instances).set_description(f'found debruijn-sequence after {tries} tries\n')
        
        # next step is to assign the black squares
        square_positions = []
        # here we calculate an inverse probability of pseudo-counts
        # to ensure equal distribution of the squares within the sequence
        p_vect = np.ones_like(sequence, dtype=int)
        for _ in items:
            # where = np.where(sequence==i)[0]
            p = (1/p_vect)**2
            p = np.nan_to_num(p, posinf=0, neginf=0, nan=0)
            p = p/p.sum()
            square_idx = np.random.choice(np.arange(len(sequence)), p=p)
            
            # remove chance of taking the same letter again
            item = sequence[square_idx]
            p_vect[sequence==item] = 0
            # decrease chance for surrounding of square, we dont want them too 
            for i in np.arange(1, n_block+1):                
                p_vect[square_idx-i:square_idx-i+1] += (5**(n_block+1-i))*(p_vect[square_idx-i:square_idx-i+1]>0)
                p_vect[square_idx+i:square_idx+i+1] += (5**(n_block+1-i))*(p_vect[square_idx+i:square_idx+i+1]>0)

            # decrease chance for same follow-up letter
            if square_idx<len(sequence)-1:
                next_i = sequence[square_idx+1]
                p_vect[sequence==next_i] += (8)*(p_vect[sequence==next_i]>0)
            square_positions.append(square_idx)

        square_list = [('1' if i in square_positions else '0') for i in range(len(sequence))]
        sequence = np.array(sequence)
        items_with_square = sequence[square_positions]
        assert len(set(items_with_square))==len(items_with_square), \
            f'warning, not each item had one square {np.bincount(items_with_square)[1:]}'
        return list(map(list, zip(sequence_str, square_list)))
    max(tqdm._instances).set_description(f'did not find debruijn-sequence after {tries} tries')
    raise Exception(f'did not find debruijn-sequence after {tries} tries')





def get_learning_units(n_learning_cycles, sequence):
    """
    creates a number (n_learning_cycles) of learning cycles,
    where each learning cycle consists of 20 sequences, with each 
    starting position at least once. The start of each sequence is
    randomized within one learning cycle.
    """
    starting_points = list(range(len(sequence)))
      
    # here the final learning units are stored
    learning_units = []
    n_pos = len(sequence)

    for _ in range(n_learning_cycles):
        starting_points = list(range(len(sequence)))
        cycle = []

        # here we keep track how often a given position was correct
        # we do not want more than 3 times in a row e.g. first position
        # to be correct. Also the positions should be more or less
        # equally distributed
        prev_position = []
        correct_pos = [0, 0, 0]

        for _ in range(len(starting_points)):
            # this is the first item of the sequence
            # after that, remove the starting point from the list of starting points
            # this way we make sure that each sequence appears equally often.
            seq1 = np.random.choice(starting_points) 
            starting_points.remove(seq1)
            
            # second item is just the next one in the sequence
            seq2 = (seq1 + 1) % n_pos
        
            # we sample until we found a sequence that does not contain any double mentions
            correct     = (seq1 + 2) % n_pos

            range_close = range(seq1+3, seq1+6)
            range_far = range(seq1+9, seq1+12) if n_pos==20 else range(seq1+6, seq1+9)

            # now chose a wrong item to be shown that is sequentially close in the grid
            choice_close = [(i%n_pos) for i in range_close if sequence[i%n_pos] not in (sequence[correct], sequence[seq1], sequence[seq2])]
            wrong_close  = np.random.choice(choice_close)
            
            # now chose a wrong item to be shown that is sequentially far away in the grid
            choice_far  = [(i%n_pos) for i in range_far if sequence[i%n_pos] not in (sequence[correct], sequence[seq1], sequence[seq2], sequence[wrong_close])]
            wrong_far   = np.random.choice(choice_far)
            
            choices = [correct, wrong_close, wrong_far]

            # now shuffle the choices
            p = np.array(correct_pos)+1
            p = (1/p)**1.5
            p = p/p.sum()

            idx_order = list(np.random.choice([0, 1, 2], 3, replace=False, p=p))
            choices_ordered = [x for _,x in sorted(zip(idx_order, choices))]
            idx_correct = choices_ordered.index(correct)


            # if the correct position was the same for the last 3 times, change
            while len(prev_position)>3 and len(set(prev_position[-3:]))==1 and \
                    idx_correct==prev_position[-1]:
                idx_order = list(np.random.choice([0, 1, 2], 3, replace=False, p=p))
                choices_ordered = [x for _,x in sorted(zip(idx_order, choices))]
                idx_correct = choices_ordered.index(correct)


            prev_position.append(idx_correct)
            correct_pos[idx_correct] += 1

            idx_correct_str = str(idx_correct+1) # presentation starts indexing at 1 not 0
            unit = [sequence[seq1], sequence[seq2], *[sequence[c] for c in choices_ordered], idx_correct_str]
            
            # now make sure there are no items presented double in any learning.
            # as items are double, this is important to be not the case
            assert len(unit) == len(set(unit)), 'An item was double in the sequence. This should not happen.'
            # make sure the correct item is indeed correct
            assert sequence[correct] == unit[int(idx_correct_str)+1], f'correct item is not at position {idx_correct_str}: {unit}'
            # make sure sequence is actually in the correct sequence
            assert unit[0]+unit[1]+unit[idx_correct+2] in sequence+sequence, f'faulty sequence correct item not correct?\n sequence is {sequence+sequence}\ntuple is: {unit}'

            cycle.append(','.join(unit))
                    
        learning_units.append(';'.join(cycle))
        
    return learning_units

def get_sequence(folder='./stimuli/images/graph/'):
    files = os.listdir(folder)
    elements = [f for f in files if f.endswith('png')]
    np.random.shuffle(elements)
    return elements



### main
if __name__=='__main__':

    #%% first create sequences for the complex graph

    os.makedirs('./sequences/graph/', exist_ok=True)
    diffs = []
    square_items = []
    next_items = []

    # number of sessions for the localizer. We perform 3, but generate 5 j.i.c.
    n_sessions = 5
    n_participants = 40
    learning_sequence = 'ABCDEBFGHIFJKLMJANOP'
    delimiter = ','
    correct_dist = []

    # Make {n_participants} learning packages. For each participant one.
    # Each package contains 15 learning runs, that should be plenty enough.
    
    for i in tqdm(list(range(0,n_participants+1)), desc='Creating sequence graph',
                  bar_format ='{l_bar}{bar}|'):
        
        learning_units = get_learning_units(15, sequence=learning_sequence)
        sequence = get_sequence()
        items = [x for x in range(1,len( sequence)+1)]
        
        
        header = ['item1', 'item2', 'choice1', 'choice2', 'choice3', 'correct_choice (1-3)']
    
        with open(f'./sequences/graph/learning_units_participant_{i}.csv', 'w') as f:
            f.write(delimiter.join(header) + '\n')
            lines = '\n'.join(learning_units)
            f.write(lines)
            
        l = [[y[-1] for y in x.split(';')] for x in learning_units]
        w = [[''.join(t) for t in list(more_itertools.windowed(lx, 3))] for lx in l]
        w = np.array(w, dtype=str).flatten()
        correct_dist.extend(w)

    
        with open(f'./sequences/graph/sequence_participant_{i}.csv', 'w') as f:
            f.write(delimiter.join(sequence) + '\n')
    
        for n in range(1, n_sessions+1):
            with open(f'./sequences/graph/localizer_participant_{i}_{n}.csv', 'w') as f:
                localizer_sequence = de_bruijn_2tuple(items)
                if i==0: 
                    # for testing purposes, sequence 0 has more squares 
                    localizer_sequence[3][1] = '1'
                    localizer_sequence[5][1] = '1'
                    localizer_sequence[8][1] = '1'
                    localizer_sequence[13][1] = '1'
                else: # do not include testing sequence in distribution checks
                    # distribution checks
                    x = np.array(localizer_sequence, dtype=int)
                    where = np.where(x[:, 1]==1)[0]
                    diff = np.diff(where)
    
                    diffs.extend(diff)
                    square_items.extend(x[x[:,1]==1,0])
                    next_items.extend(x[np.roll(x[:,1]==1, 1),0])
                lines = '\n'.join([delimiter.join(x) for x in localizer_sequence])
                f.write(lines)

    # plt.close('all')
    plt.figure()
    plt.hist(diffs, bins=np.arange(max(diffs)))
    plt.title('Distance between distractors')
    plt.figure()
    plt.hist(square_items, bins=len(np.unique(learning_sequence)))
    plt.title('Object number with distractor')
    plt.figure()
    plt.hist(next_items, bins=len(np.unique(learning_sequence)))
    plt.title('Which item follows a distractor?')
    plt.figure()
    correct_dist = sorted(correct_dist)
    plt.hist(correct_dist, bins=len(set(w)))
    plt.title('Triplets of which answer position is correct')


    #%% now create sequences for the simpler category graph
    os.makedirs('./sequences/category/', exist_ok=True)
    diffs = []
    square_items = []
    next_items = []

    # number of sessions for the localizer. We perform 3, but generate 5 j.i.c.
    n_sessions = 5
    n_participants = 40
    delimiter = ','
    correct_dist = []
    learning_sequence = "ABCDEFGHIJ"
    images_dir = './stimuli/images/category/'

    # Make {n_participants} learning packages. For each participant one.
    # Each package contains 15 learning runs, that should be plenty enough.
    
    for i in tqdm(list(range(0,n_participants+1)), desc='Creating sequence category',
                  bar_format ='{l_bar}{bar}|'):
        
        learning_units = get_learning_units(15, sequence=learning_sequence)
        sequence = get_sequence(images_dir)
        items = [x for x in range(1,len( sequence)+1)]
        
        
        header = ['item1', 'item2', 'choice1', 'choice2', 'choice3', 'correct_choice (1-3)']
    
        with open(f'./sequences/category/learning_units_participant_{i}.csv', 'w') as f:
            f.write(delimiter.join(header) + '\n')
            lines = '\n'.join(learning_units)
            f.write(lines)
            
        l = [[y[-1] for y in x.split(';')] for x in learning_units]
        w = [[''.join(t) for t in list(more_itertools.windowed(lx, 3))] for lx in l]
        w = np.array(w, dtype=str).flatten()
        correct_dist.extend(w)

    
        with open(f'./sequences/category/sequence_participant_{i}.csv', 'w') as f:
            f.write(delimiter.join(sequence) + '\n')
    
        for n in range(1, n_sessions+1):
            with open(f'./sequences/category/localizer_participant_{i}_{n}.csv', 'w') as f:
                # do three rounds in one localizer trial
                localizer_sequence = de_bruijn_2tuple(items) + de_bruijn_2tuple(items) + de_bruijn_2tuple(items)
                if i==0:
                    # for testing purposes, sequence 0 has more squares 
                    localizer_sequence[3][1] = '1'
                    localizer_sequence[5][1] = '1'
                    localizer_sequence[8][1] = '1'
                    localizer_sequence[13][1] = '1'
                else: # do not include testing sequence in distribution checks
                    # distribution checks
                    x = np.array(localizer_sequence, dtype=int)
                    where = np.where(x[:, 1]==1)[0]
                    diff = np.diff(where)
    
                    diffs.extend(diff)
                    square_items.extend(x[x[:,1]==1,0])
                    next_items.extend(x[np.roll(x[:,1]==1, 1),0])
                lines = '\n'.join([delimiter.join(x) for x in localizer_sequence])
                f.write(lines)

    # plt.close('all')
    plt.figure()
    plt.hist(diffs, bins=np.arange(max(diffs)))
    plt.title('Distance between distractors for category')
    plt.figure()
    plt.hist(square_items, bins=len(np.unique(learning_sequence)))
    plt.title('Object number with distractor for category')
    plt.figure()
    plt.hist(next_items, bins=len(np.unique(learning_sequence)))
    plt.title('Which item follows a distractor for category')
    plt.figure()
    correct_dist = sorted(correct_dist)
    plt.hist(correct_dist, bins=len(set(w)))
    plt.title('Triplets of which answer position is correct for category')


    #%% now create sequences for the simpler category graph
    os.makedirs('./sequences/category_graph/', exist_ok=True)
    diffs = []
    square_items = []
    next_items = []

    # number of sessions for the localizer. We perform 3, but generate 5 j.i.c.
    n_sessions = 5
    n_participants = 40
    delimiter = ','
    correct_dist = []
    learning_sequence = "ABCDEFGEHIBJ"
    images_dir = './stimuli/images/category/'

    # Make {n_participants} learning packages. For each participant one.
    # Each package contains 15 learning runs, that should be plenty enough.
    
    for i in tqdm(list(range(0,n_participants+1)), desc='Creating sequence category_graph',
                  bar_format ='{l_bar}{bar}|'):
        
        learning_units = get_learning_units(15, sequence=learning_sequence)
        sequence = get_sequence(images_dir)
        items = [x for x in range(1,len( sequence)+1)]
        
        
        header = ['item1', 'item2', 'choice1', 'choice2', 'choice3', 'correct_choice (1-3)']
    
        with open(f'./sequences/category_graph/learning_units_participant_{i}.csv', 'w') as f:
            f.write(delimiter.join(header) + '\n')
            lines = '\n'.join(learning_units)
            f.write(lines)
            
        l = [[y[-1] for y in x.split(';')] for x in learning_units]
        w = [[''.join(t) for t in list(more_itertools.windowed(lx, 3))] for lx in l]
        w = np.array(w, dtype=str).flatten()
        correct_dist.extend(w)

    
        with open(f'./sequences/category_graph/sequence_participant_{i}.csv', 'w') as f:
            f.write(delimiter.join(sequence) + '\n')
    
        for n in range(1, n_sessions+1):
            with open(f'./sequences/category_graph/localizer_participant_{i}_{n}.csv', 'w') as f:
                # do three rounds in one localizer trial
                localizer_sequence = de_bruijn_2tuple(items) + de_bruijn_2tuple(items) + de_bruijn_2tuple(items)
                if i==0:
                    # for testing purposes, sequence 0 has more squares 
                    localizer_sequence[3][1] = '1'
                    localizer_sequence[5][1] = '1'
                    localizer_sequence[8][1] = '1'
                    localizer_sequence[13][1] = '1'
                else: # do not include testing sequence in distribution checks
                    # distribution checks
                    x = np.array(localizer_sequence, dtype=int)
                    where = np.where(x[:, 1]==1)[0]
                    diff = np.diff(where)
    
                    diffs.extend(diff)
                    square_items.extend(x[x[:,1]==1,0])
                    next_items.extend(x[np.roll(x[:,1]==1, 1),0])
                lines = '\n'.join([delimiter.join(x) for x in localizer_sequence])
                f.write(lines)

    # plt.close('all')
    plt.figure()
    plt.hist(diffs, bins=np.arange(max(diffs)))
    plt.title('Distance between distractors for category')
    plt.figure()
    plt.hist(square_items, bins=len(np.unique(learning_sequence)))
    plt.title('Object number with distractor for category')
    plt.figure()
    plt.hist(next_items, bins=len(np.unique(learning_sequence)))
    plt.title('Which item follows a distractor for category')
    plt.figure()
    correct_dist = sorted(correct_dist)
    plt.hist(correct_dist, bins=len(set(w)))
    plt.title('Triplets of which answer position is correct for category')
