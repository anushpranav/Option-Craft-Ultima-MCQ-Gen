#import all the neccessary libraries
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from textwrap3 import wrap
import random
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('brown')
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('omw-1.4')
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
#from similarity.normalized_levenshtein import NormalizedLevenshtein
import pickle
import time
import spacy
import os

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Importing necessary libraries
import os

# Getting the current directory
current_dir = os.getcwd()

# Constructing the path to the strings.json file
strings_json_path = os.path.join(current_dir, 'tests', 'data', 'strings.json')

# Now you can use strings_json_path in your code


# Importing necessary libraries
import os
import time
import pickle
from sense2vec import Sense2Vec

# Define the path to the sense2vec model
model_path = r'C:/Users/Anush Pranav/Desktop/Hackathon/sense2vec/tests/data'

# Load the Sense2Vec model
s2v = Sense2Vec().from_disk(model_path)

# Define the path to the summary model and tokenizer
summary_model_path = 't5_summary_model.pkl'
summary_tokenizer_path = 't5_summary_tokenizer.pkl'

# Check if summary model exists
if os.path.exists(summary_model_path):
    # Load summary model
    with open(summary_model_path, 'rb') as f:
        summary_model = pickle.load(f)
    print("Summary model found on disk. Loaded successfully.")
else:
    # Download and save summary model
    print("Summary model not found on disk. Downloading...")
    start_time = time.time()
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    end_time = time.time()
    print("Downloaded summary model in", (end_time - start_time) / 60, "minutes. Saving to disk...")
    with open(summary_model_path, 'wb') as f:
        pickle.dump(summary_model, f)
    print("Summary model saved to disk.")

# Check if summary tokenizer exists
if os.path.exists(summary_tokenizer_path):
    # Load summary tokenizer
    with open(summary_tokenizer_path, 'rb') as f:
        summary_tokenizer = pickle.load(f)
    print("Summary tokenizer found on disk. Loaded successfully.")
else:
    # Download and save summary tokenizer
    print("Summary tokenizer not found on disk. Downloading...")
    start_time = time.time()
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    end_time = time.time()
    print("Downloaded summary tokenizer in", (end_time - start_time) / 60, "minutes. Saving to disk...")
    with open(summary_tokenizer_path, 'wb') as f:
        pickle.dump(summary_tokenizer, f)
    print("Summary tokenizer saved to disk.")

# Similarly, repeat the above steps for question model and tokenizer, and sentence transformer model


#Getting question model and tokenizer
if os.path.exists("C:/Users/Anush Pranav/Desktop/Hackathon/t5_question_model.pkl"):
    with open('C:/Users/Anush Pranav/Desktop/Hackathon/t5_question_model.pkl', 'rb') as f:
        question_model = pickle.load(f)
    print("Question model found in the disc, model is loaded successfully.")
else:
    print("Question model does not exists in the path specified, downloading the model from web....")
    start_time= time.time()
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    end_time = time.time()

    print("downloaded the question model in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("C:/Users/Anush Pranav/Desktop/Hackathon/t5_question_model.pkl", 'wb') as f:
        pickle.dump(question_model,f)

    print("Done. Saved the model to disc.")

if os.path.exists("C:/Users/Anush Pranav/Desktop/Hackathon/t5_question_tokenizer.pkl"):
    with open('C:/Users/Anush Pranav/Desktop/Hackathon/t5_question_tokenizer.pkl', 'rb') as f:
        question_tokenizer = pickle.load(f)
    print("Question tokenizer found in the disc, model is loaded successfully.")
else:
    print("Question tokenizer does not exists in the path specified, downloading the model from web....")

    start_time = time.time()
    question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
    end_time=time.time()

    print("downloaded the question tokenizer in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("C:/Users/Anush Pranav/Desktop/Hackathon/t5_question_tokenizer.pkl",'wb') as f:
        pickle.dump(question_tokenizer,f)

    print("Done. Saved the tokenizer to disc.")

#Loading the models in to GPU if available
summary_model = summary_model.to(device)
question_model = question_model.to(device)

#Getting the sentence transformer model and its tokenizer
# paraphrase-distilroberta-base-v1
if os.path.exists("C:/Users/Anush Pranav/Desktop/Hackathon/sentence_transformer_model.pkl"):
    with open("C:/Users/Anush Pranav/Desktop/Hackathon/sentence_transformer_model.pkl",'rb') as f:
        sentence_transformer_model = pickle.load(f)
    print("Sentence transformer model found in the disc, model is loaded successfully.")
else:
    print("Sentence transformer model does not exists in the path specified, downloading the model from web....")
    start_time=time.time()
    sentence_transformer_model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
    end_time=time.time()

    print("downloaded the sentence transformer in ",(end_time-start_time)/60," min , now saving it to disc...")

    with open("C:/Users/Anush Pranav/Desktop/Hackathon/sentence_transformer_model.pkl",'wb') as f:
        pickle.dump(sentence_transformer_model,f)

    print("Done saving to disc.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocesstext (content):
  """
  this function takes a piece of text (content), tokenizes it into sentences, capitalizes the first letter of each sentence, and then concatenates the processed sentences into a single string, which is returned as the final result. The purpose of this function could be to format the input content by ensuring that each sentence starts with an uppercase letter.
  """
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text,model,tokenizer):
  """
  This function takes the given text along with the model and tokenizer, which summarize the large text into useful information
  """
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)

  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary

def get_nouns_multipartite(content):
    """
    This function takes the content text given and then outputs the phrases which are build around the nouns , so that we can use them for context based distractors
    """
    out=[]
    try:
        nlp=spacy.load("en_core_web_sm")
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        #pos = {'PROPN','NOUN',}
        pos = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}

        #pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection( pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)


        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        #traceback.print_exc()

    return out

def get_keywords(originaltext):
  """
  This function takes the original text and the summary text and generates keywords from both which ever are more relevant
  This is done by checking the keywords generated from the original text to those generated from the summary, so that we get important ones
  """
  keywords = get_nouns_multipartite(originaltext)
  #print ("keywords unsummarized: ",keywords)
  #keyword_processor = KeywordProcessor()
  #for keyword in keywords:
    #keyword_processor.add_keyword(keyword)

  #keywords_found = keyword_processor.extract_keywords(summarytext)
  #keywords_found = list(set(keywords_found))
  #print ("keywords_found in summarized: ",keywords_found)

  #important_keywords =[]
  #for keyword in keywords:
    #if keyword in keywords_found:
      #important_keywords.append(keyword)

  #return important_keywords
  return keywords

def get_question(context,answer,model,tokenizer):
  """
  This function takes the input context text, pretrained model along with the tokenizer and the keyword and the answer and then generates the question from the large paragraph
  """
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

def filter_same_sense_words(original,wordlist):

  """
  This is used to filter the words which are of same sense, where it takes the wordlist which has the sense of the word attached as the string along with the word itself.
  """
  filtered_words=[]
  base_sense =original.split('|')[1]
  #print (base_sense)
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  """
  This function takes the given word along with the wordlist and then gives out the max-score which is the levenshtein distance for the wrong answers
  because we need the options which are very different from one another but relating to the same context.
  """
  score=[]
  normalized_levenshtein = NormalizedLevenshtein()
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    """
    This function takes the input word, sentence to vector model and top similar words and also the question
    Then it computes the sense of the given word
    then it gets the words which are of same sense but are most similar to the given word
    after that we we return the list of words which satisfy the above mentioned criteria
    """
    output = []
    #print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      output = filter_same_sense_words(sense,most_similar)
      #print ("Similar ",output)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)

    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """
    The mmr function takes document and word embeddings, along with other parameters, and uses the Maximal Marginal Relevance (MMR) algorithm to extract a specified number of keywords/keyphrases from the document. The MMR algorithm balances the relevance of keywords with their diversity, helping to select keywords that are both informative and distinct from each other.
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    """
    the get_distractors_wordnet function uses WordNet to find a relevant synset for the input word and then generates distractor words by looking at hyponyms of the hypernym associated with the input word. These distractors are alternative words related to the input word and can be used, for example, in educational or language-related applications to provide choices for a given word.
    """
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]

      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0:
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors(word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval):
    """
    This function generates distractor words (answer choices) for a given target word in the context of a provided sentence.
    It selects distractors based on their similarity to the target word's context and ensures that the target word itself is not included among the distractors.
    """
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    if len(distractors) == 0:
        return distractors

    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)

    embedding_sentence = origsentence + " " + word.capitalize()
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    max_keywords = min(len(distractors_new), 4)  # Ensure max 4 distractors
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)

    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    return final[1:]  # Return distractors excluding the correct answer


def get_mca_questions(context, num_questions):
    """
    This function generates multiple-choice questions based on a given context.
    It summarizes the context, extracts important keywords, generates questions related to those keywords,
    and provides randomized answer choices, including the correct answer, for each question.
    """
    summarized_text = summarizer(context, summary_model, summary_tokenizer)

    imp_keywords = get_keywords(context)
    output_list = []
    # Loop until the desired number of questions is reached
    while len(output_list) < num_questions:
        for answer in imp_keywords:
            output = ""
            ques = get_question(summarized_text, answer, question_model, question_tokenizer)

            distractors = get_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2)

            output = output + ques + "\n"
            if len(distractors) == 0:
                distractors = imp_keywords

            if len(distractors) > 0:
                random_integer = random.randint(0, 3)
                alpha_list = ['(a)', '(b)', '(c)', '(d)']
                options = [answer.capitalize()]  # Correct answer
                options.extend(random.sample(distractors, 3))  # Randomly select 3 distractors
                random.shuffle(options)  # Shuffle the options
                output_list.append((output, options, alpha_list[random_integer]))

                # Check if the desired number of questions is reached
                if len(output_list) == num_questions:
                    break

    return output_list[:num_questions]  # Ensure only the requested number of questions are returned




    






