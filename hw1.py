import argparse
import math
import random
from tracemalloc import start
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator
import nltk


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    start_text = [START_TOKEN] * (n - 1)
    start_text.extend(text)
    start_text.append(END_TOKEN)

    # This basically creates arrays shifted by a position i which allows us to create the pairings between the word and the context
    ngram_list = list(zip(*[start_text[i:] for i in range(n)]))

    for i in range(len(ngram_list)):
        ngram = ngram_list[i]
        ngram_list[i] = tuple((ngram[n - 1], tuple(ngram[: n - 1])))

    return ngram_list

# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings


def load_corpus(corpus_path: str) -> List[List[str]]:
    corpus = open(corpus_path).read()
    paragraphs = corpus.split("\n\n")
    sentences = []
    for paragraph in paragraphs:
        sentence = sent_tokenize(paragraph)
        sentences.extend(sentence)

    for i in range(len(sentences)):
        sentences[i] = word_tokenize(sentences[i])
    return sentences


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    ngram_model = NGramLM(n)
    sentences = load_corpus(corpus_path)
    for text in sentences:
        ngram_model.update(text)
    return ngram_model


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        self.vocabulary.update(set(text))
        self.vocabulary.add("</s>")
        for ngram in get_ngrams(self.n, text):
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
            self.context_counts[ngram[1]] = self.context_counts.get(ngram[1], 0) + 1

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float

    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta=.0) -> float:
        if delta == 0:
            if self.context_counts.get(context, 0) == 0:
                return 1/len(self.vocabulary)
            return self.ngram_counts.get((word, context), 0)/self.context_counts[context]
        else:
            n_gram_laplace_prob = (self.ngram_counts.get((word, context), 0) + delta)/(
                self.context_counts.get(context, 0) + (delta * len(self.vocabulary)))
            return n_gram_laplace_prob

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        prob_sum = 0
        for ngram in get_ngrams(self.n, sent):
            ngram_prob = self.get_ngram_prob(ngram[0], ngram[1], delta)
            if ngram_prob == 0:
                prob_sum += -math.inf
            else:
                prob_sum += math.log(ngram_prob, 2)
        return prob_sum

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        #corpus_test_size keeps a track of the number of words (not unique) in the test set.
        corpus_test_size = 0
        corpus_prob = 0
        for sentence in corpus:
            corpus_test_size += len(sentence)
            corpus_prob += self.get_sent_log_prob(sentence)
        perplexity = math.pow(2, (-corpus_prob/corpus_test_size))
        return perplexity

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        sorted_vocab_list = sorted(self.vocabulary)
        random_number = random.random()
        lower_bound = 0
        upper_bound = 0
        for word in sorted_vocab_list:
            ngram_prob = self.get_ngram_prob(word, context, delta)
            upper_bound += ngram_prob
            if lower_bound < random_number and upper_bound >= random_number:
                return word
            lower_bound = upper_bound

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        length = 0
        start_tokens = ['<s>'] * (self.n - 1)
        first_word = self.generate_random_word(tuple(start_tokens), delta)
        length += 1
        generated_text = start_tokens + [first_word]
        # isEndToken = False
        while length < max_length:
            test1 = [generated_text[-i] for i in range(1, self.n - 1)]
            test2 = generated_text[-(self.n - 1)]
            if self.n == 1:
                context = ()
            else:
                context = tuple(generated_text[-(self.n - 1) :])
            next_word = self.generate_random_word(context, delta)
            length += 1
            generated_text.append(next_word)
            if next_word == '</s>':
                # isEndToken = True
                break
            
        return ' '.join(generated_text[self.n - 1 :])


def main(corpus_path: str, delta: float, seed: int):
    bigram_lm = create_ngram_lm(2, corpus_path)
    s1 = 'love Science'
    s2 = 'I love Computer Science'

    print(bigram_lm.get_perplexity([word_tokenize(s1)]))
    print(bigram_lm.get_perplexity([word_tokenize(s2)]))

    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'
    print(trigram_lm.get_sent_log_prob(word_tokenize(s1), delta))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2), delta))
    print(trigram_lm.get_perplexity([word_tokenize(s1), word_tokenize(s2)]))

    print('Unigram Text Generation:')
    unigram_lm = create_ngram_lm(1, corpus_path)
    for i in range(5):
        print(unigram_lm.generate_random_text(10))
    
    print('Trigram Text Generation:')
    trigram_lm = create_ngram_lm(3, corpus_path)
    for i in range(5):
        print(trigram_lm.generate_random_text(10))
    
    print('Pentagram Text Generation:')
    pentagram_lm = create_ngram_lm(5, corpus_path)
    for i in range(5):
        print(pentagram_lm.generate_random_text(10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='/Users/jayasuryaagovindraj/Documents/NLP Assignments/Assignment 1/Programming/warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.5, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)