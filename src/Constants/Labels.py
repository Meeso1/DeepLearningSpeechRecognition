labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"]
known_folders = labels[:-1]
unknown_folders = ["bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]
all_folders = known_folders + unknown_folders
background_noise_folder = ["_background_noise_"]