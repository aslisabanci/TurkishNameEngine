from preprocessor import *
from simple_rnn import *

first_names_file = 'names_10k.txt'
first_names = get_names_from_file(first_names_file)
preprocessed_names = clean_and_split(first_names)

preprocessed_filename = "preprocessed_" + first_names_file
with open(preprocessed_filename, mode="w") as outfile:
	for clean_name in preprocessed_names:
		outfile.write("%s\n" % clean_name)

generate_names(preprocessed_filename)