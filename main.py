#import random
import os
#import sys
#import nltk
from nltk.stem import *		#word stemming
from nltk.corpus import stopwords

LEARNING_RATE = 0.12
ITERATIONS = 100

def main():
	#nltk.download('stopwords')
	accuracy = algo(True)
	print("TOTAL ACCURACY WITH " + str(ITERATIONS) + " ITERATIONS:")
	#print(str(round(accuracy, 3)) + "%")
	print(str(accuracy) + "%")

def algo(stop=False):
	count = 0
	print("Initializing Data...")
	stops = set(stopwords.words('english'))

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
						if word.lower() in specialCharacters or (stop and word.lower() in stops):
							continue
						stem = stemmer.stem(word).lower()
						if stem not in bag_of_words.keys():
							bag_of_words[stem] = 0
				fo.close()
	#Initialize weights with random values.
	weights = {key: 0 for key in bag_of_words.keys()}
	assert len(weights) == len(bag_of_words)

	print("Iterating through training data, updating weights...")
	directories = ["./train/spam", "./train/ham"]
	for i in range(ITERATIONS):
		if ((i*100.0)/ITERATIONS).is_integer():
			print(str((i*100)/ITERATIONS) + "% complete ...")
		for directory in directories:
			for file in os.listdir(directory):
				# index -7 of text file str determines the ground truth
				# examples: "blablabla.s[p]am.txt" --> spam, blablabla.[h]am.txt" --> ham
				spamOrHam = str(file)[-7]
				fo = open(os.path.join(directory, file), 'r', encoding='utf-8')
				for line in fo:
					for word in line.split():
						if word.lower() in specialCharacters or (stop and word.lower() in stops):
							continue
						stem = stemmer.stem(word).lower()
						assert stem in bag_of_words.keys()
						bag_of_words[stem] += 1
				fo.close()
				truth = 0 if spamOrHam == "h" else 1
				prediction = perceptron(bag_of_words, weights)
				prediction = 0 if prediction < 0 else 1
				for w in weights.keys():
					if bag_of_words[w] == 0 or truth == prediction:
						count += 1
						continue
					delta_w = LEARNING_RATE * (truth - prediction) * bag_of_words[w]
					weights[w] += delta_w

				resetFeatures(bag_of_words)

	print("Verifying accuracies with test data...")
	total_guesses = 0
	correct_guesses = 0
	directories = ["./test/spam", "./test/ham"]
	for directory in directories:
		for file in os.listdir(directory):
			spamOrHam = str(file)[-7]
			fo = open(os.path.join(directory, file), 'r', encoding='utf-8')
			for line in fo:
				for word in line.split():
					if word.lower() in specialCharacters or (stop and word.lower() in stops):
						continue
					stem = stemmer.stem(word).lower()
					assert stem in bag_of_words.keys()
					bag_of_words[stem] += 1
			fo.close()
			truth = 0 if spamOrHam == "h" else 1
			prediction = perceptron(bag_of_words, weights)
			prediction = 0 if prediction < 0 else 1
			if prediction == truth:
				correct_guesses += 1
			total_guesses += 1
	print("count: " + str(count))
	return (correct_guesses*100.0)/total_guesses

def resetFeatures(words):
	for word in words.keys():
		words[word] = 0

def perceptron(bag_of_words, weights):
	s = 0
	for i in bag_of_words.keys():
		s += weights[i] * bag_of_words[i]
	return s


if __name__ == '__main__':
	main()