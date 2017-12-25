from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# you need the below lines executed once
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# txt = """
#       FRANK and Joe Hardy clutched the grips of their motorcycles and stared in
#       horror at the oncoming car. It was careening from side to side on the narrow
#       road.
#       "He'll hit us! We'd better climb this hillside- and fast!" Frank exclaimed, as
#       the boys brought their motorcycles to a screeching halt and leaped off.
#       "On the double!" Joe cried out as they started up the steep embankment.
#       To their amazement, the reckless driver suddenly pulled his car hard to the
#       right and turned into a side road on two wheels. The boys expected the car to
#       turn over, but it held the dusty ground and sped off out of sight.
#       """

txt = 'WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s'

tokenized = word_tokenize(txt)

# Prints the word in this sentence
print(tokenized)

# Tags each word
tagged_words_list = nltk.pos_tag(tokenized)

for tagged_words in tagged_words_list:
  print(nltk.help.upenn_tagset(tagged_words[1]))
  print(tagged_words[0])

