import os

# https://stackoverflow.com/a/57896232
def unique_path(initial_path):
    counter = 1
    path = initial_path + str(counter)

    while os.path.exists(path):
        path = initial_path + str(counter)
        counter += 1

    return path