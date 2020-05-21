import time
import threading
from multiprocessing import Process
import numpy as np
import random
import heapq
import os, inspect
import sys
import csv
from pathlib import Path
"""
Run code for different input values and store.
"""
mutation_probability = 40
fieldnames = [
    'score', 
    'N', 
    'E', 
    'population_size', 
    'mutation_probability', 
    'accept_threshold', 
    'coding_path'
]

initial_codings = dict()


def execute(*args):
    coding_dimension, E, mutation_probability, population_size, accept_threshold = [i for i in args[0:5]]

    ind = args[5]
    mutation_probability /= 100
    N = len(ind) // 4
    file_name = args[6]
    score = save_codings(ind, coding_dimension=coding_dimension, E=E, N=N)
    save_record(ind, score,
        store_mode=True,
        coding_dimension=coding_dimension, 
        N=N, E=E, accept_threshold=accept_threshold, 
        file_name=file_name, 
        population_size=population_size, mutation_probability=mutation_probability)

def get_index_in_coding(index, coding_dimension):
    a = index % (coding_dimension ** 2)
    return a // coding_dimension, a % coding_dimension

def get_codinggs_ind(individual, coding_dimension):
    ind_reshaped = np.reshape(individual, (4, len(individual) // 4))
    codings = []
    for i in ind_reshaped:
        coding = np.zeros((coding_dimension, coding_dimension))
        for index in i:
            ri_1, ri_2 = get_index_in_coding(index, coding_dimension)
            coding[int(ri_1)][int(ri_2)] = 1
        codings.append(coding)
    return codings

def fitness(individual, **kwargs):

    coding_dimension=kwargs['coding_dimension']
    N = kwargs['N']
    E=kwargs['E']
    threshold=N - E

    def cross_correlation(codings):
        lng = 4
        number_of_peaks = 0
        for i in range(lng):  # the one who is being correlated.
            for j in range(lng):  # upper left
                for k in range(lng):  # upper right
                    for l in range(lng):  # lower left
                        for m in range(lng):  # lower right
                            c = np.concatenate((
                                np.concatenate((codings[j], codings[k]), 1),
                                np.concatenate((codings[l], codings[m]), 1)
                            ), 0)
                            number_of_peaks += cc_22_1(c, codings[i], i, j, k, l, m)
        return number_of_peaks

    def cc_22_1(grid, item, item_index, j_index, k_index, l_index, m_index):
        g_size = coding_dimension * 2
        res = 0
        for i in range(g_size):
            for j in range(g_size):
                peak_flag = 0
                for index_1 in range(coding_dimension):
                    for index_2 in range(coding_dimension):
                        if index_1 + i >= g_size or index_2 + j >= g_size:
                            continue
                        peak_flag += grid[index_1 + i][index_2 + j] * item[index_1][index_2]
                if peak_flag >= threshold and not is_auto_cc(item_index, j_index, k_index, l_index, m_index, i, j):
                    res += 1
        return res

    def is_auto_cc(item_index, j_index, k_index, l_index, m_index, i, j):
        return (
                item_index == j_index and i is 0 and j is 0 or
                item_index == k_index and i is 0 and j is coding_dimension or
                item_index == l_index and i is coding_dimension and j is 0 or
                item_index == m_index and i is coding_dimension and j is coding_dimension
        )

    codings = get_codinggs_ind(individual, coding_dimension)
    cc = cross_correlation(codings)
    return cc, codings


def save_record(individual, ftn, **kwargs):
    p = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    partial_path = 'codings' + os.sep + 'dim_' + str(kwargs['coding_dimension']) + os.sep + 'N_' + str(kwargs['N']) + os.sep + 'E_' + str(kwargs['E'])
    local_path = '' + os.sep + str(ftn) + '.csv'
    if ftn < kwargs['accept_threshold'] and kwargs['store_mode']:
        file_path = partial_path + local_path
    else:
        file_path = '-'
    with open(p + os.sep + kwargs['file_name'] + '.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'score': ftn,
            'N': kwargs['N'],
            'E': kwargs['E'],
            'population_size': kwargs['population_size'],
            'mutation_probability': kwargs['mutation_probability'],
            'accept_threshold': kwargs['accept_threshold'],
            'coding_path': file_path
        })

def write_coding(coding_dimension, score, N, E, A, C, G, T):
    coding_path = 'codings' + os.sep + 'dim_' + str(coding_dimension) + os.sep + 'N_' + str(N) + os.sep + 'E_' + str(E)
    local_path = '' + os.sep + str(score) + '.csv'
    if not os.path.exists(coding_path):
        os.makedirs(coding_path)
    with open(coding_path + local_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in [A, C, G, T]:
            writer.writerow(row)
        csvfile.close()

def save_codings(individual, **kwargs):
    p = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ftn, cs = fitness(individual, **kwargs)
    arrs = []
    for c in cs:
        arr = []
        for row in c:
            for i in row:
                arr.append(i)
        arrs.append(arr)
    write_coding(kwargs['coding_dimension'], ftn, kwargs['N'], kwargs['E'], arrs[0], arrs[1], arrs[2], arrs[3])
    return ftn


def convert_path(path):
	return os.sep.join(path.split('\\'))


def find_min_path(coding_dimension, N):
    with open('window_size_' + str(coding_dimension) + '.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        min_score = float('inf')
        min_path = ''
        for row in reader:
            if row['N'] == 'N':
                continue
            if int(row['N']) == int(N):
                s = int(row['score'])
                p = convert_path(row['coding_path'])
                if min_score > s:
                    try:
                        with open(p, 'r', newline='') as f:
                            pass
                        min_score = s
                        min_path = p
                    except:
                    	pass
        return min_path


def initialize_codings(dims=range(3, 16)):
    """
    Get best initial coding for each N in each dimension.
    By best, we mean the coding with least score from records.
    """
    for i in dims:
        temp = dict()
        temp['size'] = i
        temp['population_size'] = 50
        temp['accept_threshold'] = float('1000000')
        temp['file_name'] = 'window_size_' + str(i)
        temp['n_based'] = dict()
        initial_codings['dim_' + str(i)] = temp
        for k in [j * i ** 2 // 100 for j in range(30, 80, 10)]:
            temp['n_based'][str(k)] = []
        for N in temp['n_based']:
            min_path = find_min_path(i, N)
            if not os.path.isfile(min_path):
            	continue
            with open(min_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                coding = [[float(m) for m in r] for r in reader]
                temp['n_based'][str(N)] = coding


def get_initial(cd, n):
    coding = initial_codings['dim_' + str(cd)]['n_based'][str(n)]
    base = 0
    ind = []
    for c in coding:
        assert (c.count(1) == n)
        for i, item in enumerate(c):
            if int(item) == 1:
                ind.append(i + base)
        base += cd ** 2
    return ind


def get_initial_from_arr(arr, n):
    coding = arr
    base = 0
    ind = []
    cd = np.sqrt(len(arr[0]))
    for c in coding:
        assert (c.count(1) == n)
        for i, item in enumerate(c):
            if item == 1:
                ind.append(i + base)
        base += cd ** 2
    return ind


def clear_records(coding_dimension):
    global fieldnames
    with open('window_size_' + str(coding_dimension) + '.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def target_function(coding_dimension, E, population_size, mutation_probability, accept_threshold, ind, file_name):


    N = len(ind) // 4
    threshold = N - int(E)
    key = 0
    iteration_stop = 0  

    d = dict()
    d['N'] = N
    d['E'] = E
    d['coding_dimension'] = coding_dimension
    d['accept_threshold'] = accept_threshold
    d['file_name'] = file_name
    d['population_size'] = population_size
    d['mutation_probability'] = mutation_probability

    
    def has_collision(codings):
        for i, c in enumerate(codings):
            for j, d in enumerate(codings):
                collide = True
                if i is j:
                    continue
                for k in range(coding_dimension):
                    for m in range(coding_dimension):
                        if abs(int(c[k][m]) - int(d[k][m])) > 1e-3:
                            collide = False
                if collide:
                    return True
        return False

    def mutate(individual, passive=True):
        global coding_dimension, mutation_probability
        a = random.random()
        if passive and a > mutation_probability:
            return individual.copy()
        index = random.randint(0, len(individual) - 1)
        victim = individual[index]
        cd = coding_dimension ** 2
        individual_copy = individual.copy()
        while True:
            suggestion = random.randint(victim // cd * cd, (victim // cd + 1) * cd - 1)
            if individual_copy.count(suggestion) == 0:
                individual_copy[index] = suggestion
                break
        return individual_copy

    def crossover(ind_1, ind_2):
        ind_1_ = list(ind_1).copy()
        ind_2_ = list(ind_2).copy()
        point = random.randint(0, 3) * N
        tmp = ind_2_[:point].copy()
        ind_2_[:point], ind_1_[:point] = ind_1_[:point], tmp
        ind_1_ = mutate(ind_1_)
        ind_2_ = mutate(ind_2_)
        return ind_1_, ind_2_


    print(coding_dimension, E, population_size, mutation_probability, accept_threshold, ind, file_name)

    p = [ind]
    for i in range(1, population_size):
    	k = random.randint(0, i - 1)
    	p.append(mutate(p[k], False))

    li = [(fitness(item, **d)[0], i, item) for i, item in enumerate(p)]
    heapq.heapify(li)
    key = len(li)
    best_fit = heapq.nsmallest(1, li)[0][0]
    worst_fit = heapq.nlargest(1, li)[0][0]
    best_ind = heapq.nsmallest(1, li)[0][2]
    if best_fit < accept_threshold and not has_collision(get_codinggs_ind(ind, coding_dimension)):
        save_codings(best_ind, **d)
    save_record(best_ind, best_fit, store_mode=True, **d)
    print('Genetic evolution begins...')
    while best_fit > iteration_stop:
        print('Best fit, E: ', E, str(best_fit), coding_dimension)
        parent_1 = heapq.heappop(li)
        parent_2 = heapq.heappop(li)
        p_new_1, p_new_2 = crossover(parent_1[2], parent_2[2])
        if fitness(p_new_1, **d)[0] > worst_fit or fitness(p_new_2, **d)[0] > worst_fit:
            p_new_1 = parent_1[2]
            p_new_2 = parent_2[2]
        key += 2
        heapq.heappush(li, (fitness(p_new_1, **d)[0], key - 1, p_new_1))
        heapq.heappush(li, (fitness(p_new_2, **d)[0], key, p_new_2))
        bs = heapq.nsmallest(1, li)
        best_fit = bs[0][0]
        worst_fit = heapq.nlargest(1, li)[0][0]
        sm = False
        if best_fit < accept_threshold and not has_collision(get_codinggs_ind(bs[0][2], coding_dimension)):
            save_codings(bs[0][2], **d)
            sm = True

        save_record(bs[0][2], bs[0][0], store_mode=sm, **d)


def create_random_coding(cd, n):
    arr = []
    base = np.repeat([0, 1], [cd ** 2 - n, n])
    for _ in range(4):
        arr.append(list(np.random.permutation(base)))
    return arr


def default_random_initialization(cd_arr=range(16,31)):
    for coding_dimension in cd_arr:
        n_range = [i * coding_dimension ** 2 // 100 for i in range(30, 80, 10)]
        for n in n_range:
            for portion in range(25, 75, 5):
                e = int(n) * portion // 100
                codings_arr = create_random_coding(coding_dimension, n)
                execute(coding_dimension, e, 0, 0, float('inf'),
                        get_initial_from_arr(codings_arr, n), 'window_size_' + str(coding_dimension))

        add_header(coding_dimension)


def add_header(coding_dimension):
    global fieldnames
    flag = False
    rows = None
    fn = 'window_size_' + str(coding_dimension) + '.csv'
    with open(fn, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header is None:
            flag = True
        elif header[0] != fieldnames[0]:
            flag = True
            rows = [header]
            rows.extend([row for row in reader])

    if flag:
        with open(fn, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            if not (rows is None):
                for r in rows:
                    d = dict()
                    for i, field in enumerate(fieldnames):
                        d[field] = r[i]
                    writer.writerow(d)

initialize_codings(dims=[5])

# default_random_initialization(cd_arr=[13, 14, 15])

threads = []
for key in initial_codings:
    c = initial_codings[key]
    coding_dimension = c['size']
    p_size = c['population_size']
    accept_threshold = c['accept_threshold']
    file_name = c['file_name']
    for n in c['n_based']:
        individual = get_initial(coding_dimension, int(n))
        for portion in range(25, 75, 5):
            e = int(n) * portion // 100
            threads.append(Process(target=target_function, args=(
                coding_dimension, e, p_size, mutation_probability, accept_threshold, individual, file_name)))
            threads[-1].start()
            # threads[-1].join(0.5)
            # break
        # break
    # break

time.sleep(60 * 60 * 90)
os._exit(-1)
sys.exit()
