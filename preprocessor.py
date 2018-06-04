def has_turkish_chars(name_string):

	unwanted_chars = ['.', 'X', 'W', 'Q', "'", '(', ')', '-', '.', '/', '0', '1', '5', '7', '<', '>', '?', '_', '`']

	for char in unwanted_chars:
		if char in name_string:
			return False

	return True


def get_names_from_file(filename):
	file = open(filename, 'r')
	all_names = [line.rstrip('\n') for line in file]
	return all_names


def clean_and_split(all_names):

	cleaned_names = []

	for name in all_names:
		if has_turkish_chars(name):

			if ' ' in name:
				split_name = name.split(' ')
				for single_name in split_name:
					if single_name not in cleaned_names:
						cleaned_names.append(single_name)

	return cleaned_names
