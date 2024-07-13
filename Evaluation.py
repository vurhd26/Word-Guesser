import time as tm
import pickle
import os
import warnings

from submit import my_fit, my_predict


file_path = r"D:\dict"
with open(file_path, 'r') as f:
    words = f.read().split('\n')[:-1] 

num_words = len(words)
n_trials = 5

t_train = 0
m_size = 0
t_test = 0
prec = 0

def get_bigrams(word, lim=None):
    bg = map( ''.join, list( zip( word, word[1:] ) ) )
    bg = sorted( set( bg ) )
    return tuple( bg )[:lim]

lim_bg = 5
lim_out = 5

for t in range(n_trials):
    # Measure the training time
    tic = tm.perf_counter()
    model = my_fit(words)
    toc = tm.perf_counter()
    t_train += toc - tic

    # Save the model to a file
    model_filename = f"model_dump_{t}.pkl"
    with open(model_filename, "wb") as outfile:
        pickle.dump(model, outfile)

    # Measure the model size
    m_size += os.path.getsize(model_filename)

    # Measure the testing time
    tic = tm.perf_counter()
    for word in words:
        bigrams = get_bigrams(word, lim=lim_bg)
        guess_list = my_predict(model, bigrams)
        # Only consider the first few guesses
        guess_list = guess_list[:lim_out]
        print(guess_list)
        # Calculate precision
        if word in guess_list:
            prec += 1 / len(guess_list)
    toc = tm.perf_counter()
    t_test += toc - tic

# Compute the average metrics
t_train /= n_trials
m_size /= n_trials
t_test /= n_trials
prec /= (n_trials * num_words)

print(t_train, m_size, prec, t_test)