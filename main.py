import random
import os
from nltk.stem import *		#word stemming
from nltk.corpus import stopwords

def main():
	LEARNING_RATE = 0.1
	bag_of_words = dict()
	stemmer = SnowballStemmer('english')
	specialCharacters = [
				'subject:', '.', ',', '/', '\\', '!', '@',
				'`', '~', '#', '$', '%', '^', '&', '*', '(',
				')', '[', ']', '{', '}', ';', ':', '\'', '\"',
				'<', '>', '?', '-', '+', '=', '_'
	]

	directories = ["./train/spam/", "./train/ham", "./test/spam", "./test/ham"]
	for directory in directories:
		for filename in os.listdir(directory):
			if filename.endswith(".txt"):
				fo = open(os.path.join(directory, filename), 'r', encoding='utf-8')
				for line in fo:
					for word in line.split():
						if word.lower() in specialCharacters:  # or (stop and word.lower() in stops):
							continue
						stem = stemmer.stem(word).lower()
						if stem in bag_of_words.keys():
							bag_of_words[stem] += 1
						else:
							bag_of_words[stem] = 1
				fo.close()
	weights = dict.fromkeys(bag_of_words.keys(), 0)
	print(weights)


if __name__ == '__main__':
	main()