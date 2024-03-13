from gensim.models import KeyedVectors
import numpy as np

def load_wordvec():
    filename = 'GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    class_to_index = []
    with open('Data/classes.txt') as f:
        for line in f:
            class_name = line.split('\t')[1].strip()
            class_to_index.append(class_name)

    vec_array = np.zeros((50, 300))
    for i, animal in enumerate(class_to_index):
        animal_words = animal.split('+')
        animal_vecs = []
        for word in animal_words:
            try:
                animal_vecs.append(model[word])
            except KeyError:
                pass  # Handle unknown words here, maybe use a placeholder vector
        if animal_vecs:
            vec_array[i] = np.mean(animal_vecs, axis=0)
        else:
            print(f"No vectors found for {animal}")

    np.save('vec_array.npy', vec_array)

load_wordvec()