

# Ali.cheraghian@anu.edu.au

# Code for loading word vectors of ModelNet40, McGill, and SHREC2015 datasets

import numpy as np
import scipy.io



### load word vectors of the seen set
seen_set_index =np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])

wordvector = scipy.io.loadmat('ModelNet40_w2v')
glovevector = scipy.io.loadmat('ModelNet40_glove')

w2v = wordvector['word']
w2v_seen_set = w2v[seen_set_index,:]

glove = glovevector['word']
glove_seen_set = glove[seen_set_index,:]

##### load unseen sets

# ModelNet10

unseen_set_index =np.int16([1,2,8,12,14,22,23,30,33,35])

w2v_ModelNet10_unseen_set = w2v[unseen_set_index,:]
glove_ModelNet10_useen_set = glove[unseen_set_index,:]

# McGill

wordvector = scipy.io.loadmat('McGill_w2v')
glovevector = scipy.io.loadmat('McGill_glove')

w2v_McGill_unseen_set = wordvector['word']
glove_McGill_seen_set = glovevector['word']

# SHREC2015

wordvector = scipy.io.loadmat('SHREC_w2v')
glovevector = scipy.io.loadmat('SHREC_glove')

w2v_SHREC_unseen_set = wordvector['word']
glove_SHREC_seen_set = glovevector['word']