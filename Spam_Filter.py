#!/usr/bin/env python
# coding: utf-8

#    # Building A Spam Filter with Naive Bayes
#    
#    ### In this Project i am making a spam filter for sms messages using the multinominal Naive Bayes algorithm. The goal of this project is to find spam messages with accuracy of more than 80% 
# .
# 
# The data set I am using is obtained from Machine Learning Repository (UCI) 
# 
# 
# ## Exploring The data set

# In[1]:


import pandas as pd

spam_sms = pd.read_csv('/home/ashuyadav00/Downloads/smsspamcollection/SMSSpamCollection.csv', sep ='\t',header = None, names =['Label','SMS'])

print (spam_sms.shape)
spam_sms.head()


# Now lets find percentage of ham and spam messages.

# In[2]:


spam_sms['Label'].value_counts(normalize=True)


# As we can Spam messages are about 87% and ham sms are 13%

# ## Training And Test Set
# 
# I am splitting this data set in ratio of 4:1 for training as test set respectively
# 

# In[3]:


# Radomizing the data set
random_dataset = spam_sms.sample(frac = 1, random_state =1)

# Calculate index for split
training_test_index = round(len(random_dataset)*0.8)

# Training/Test Split
training_set = random_dataset[:training_test_index].reset_index(drop = True)
test_set = random_dataset[training_test_index:].reset_index(drop = True)

print(training_set.shape)
print(test_set.shape)


# Now lets see the the new percentage of spam and ham sms in both test and training data set, should be same as the full data set

# In[4]:


training_set['Label'].value_counts(normalize=True)


# In[5]:


test_set['Label'].value_counts(normalize=True)


# The result looks good and have same percentage as the full dataset

# ## Data Cleaning
# 
# To calculate all the probabilities required by the algorithm, we'll first need to perform a bit of data cleaning to bring the data in a format that will allow us to extract easily all the information we need.
# 
# Essentially we have to convert the data into word count format 
# 
# ### Letter Case and Punctuation 
# 
# Now removing all punctuations and converting all capital letter to small
# 

# In[6]:


# Before cleaning
training_set.head()


# In[7]:


# After Cleaning
training_set['SMS']= training_set['SMS'].str.replace('\W'," ")
training_set['SMS']= training_set['SMS'].str.lower()
training_set.head()


# ## Creating The Vocabulary
# 
# Let's now move to creating the vocabulary, which in this context means a list with all the unique words in our training set.

# In[8]:


training_set = training_set.dropna()
training_set['SMS'] = training_set['SMS'].str.split()
vocabulary = []

for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))        


# In[9]:


len(vocabulary)


# ## The Final Training Set
# 
# Now going to use the vocabulary we just created to make the data transformation we want.

# In[10]:


word_count_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_count_per_sms[word][index] += 1       


# In[11]:


word_counts = pd.DataFrame(word_count_per_sms)
word_counts.head()


# In[12]:


training_set_clean = pd.concat ([training_set, word_counts],axis= 1)
training_set_clean.head()


# ## Calculating Constants First
# 
# We're now done with cleansing.
# The Naive Bayes algorithm will need to answer these two probability questions to be able to classify new messages:
# 
# $$
# P(Spam | w_1,w_2, ..., w_n) \propto P(Spam) \cdot \prod_{i=1}^{n}P(w_i|Spam)
# $$$$
# P(Ham | w_1,w_2, ..., w_n) \propto P(Ham) \cdot \prod_{i=1}^{n}P(w_i|Ham)
# $$
# Also, to calculate P(wi|Spam) and P(wi|Ham) inside the formulas above, we'll need to use these equations:
# 
# $$
# P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}}
# $$$$
# P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
# $$
# Some of the terms in the four equations above will have the same value for every new message. We can calculate the value of these terms once and avoid doing the computations again when a new messages comes in. Below, we'll use our training set to calculate:
# 
# P(Spam) and P(Ham)
# NSpam, NHam, NVocabulary
# We'll also use Laplace smoothing and set $\alpha = 1$.

# In[13]:


# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label']== 'spam']
ham_messages = training_set_clean[training_set_clean['Label']== 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages)/ len(training_set_clean)
p_ham = len(ham_messages)/ len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1


# ## Calculating Parameters
# 
# Now that we have the constant terms calculated above, we can move on with calculating the parameters $P(w_i|Spam)$ and $P(w_i|Ham)$. Each parameter will thus be a conditional probability value associated with each word in the vocabulary.
# 
# The parameters are calculated using the formulas:
# 
# $$
# P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}}
# $$$$
# P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
# $$

# In[14]:


# Initiate Parameters 
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate Parameters
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_messages[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    parameters_ham[word] = p_word_given_ham


# ## Classifying a new Message
# 
# Now that we have all our parameters calculated, we can start creating the spam filter. The spam filter can be understood as a function that:
# 
# - Takes in as input a new message (w1, w2, ..., wn).
# - Calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn).
# - Compares the values of P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn), and:
#      - If P(Ham|w1, w2, ..., wn) > P(Spam|w1, w2, ..., wn), then the message is classified as ham.
#      - If P(Ham|w1, w2, ..., wn) < P(Spam|w1, w2, ..., wn), then the message is classified as spam.
#      - If P(Ham|w1, w2, ..., wn) = P(Spam|w1, w2, ..., wn), then the algorithm may request human help.

# In[15]:


import re

def classify(message):
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
            
    print('P(Spam|message):' ,p_spam_given_message)
    print('P(Ham|message):' ,p_ham_given_message)
    
    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal probabilities, require human assistance for classification')


# Lets try this classifier for one own test message

# In[16]:


classify('How are you today')


# ## Measuring The Filter's Accuracy
# 
# Lets see how this filter do for our test set

# In[17]:


def classify_test_set(message):
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        elif word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message < p_ham_given_message:
        return 'spam'
    else:
        return 'Needs Human Verification'


# Now we can use this fuction to create a new column in the test set

# In[18]:


test_set['Predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()


# Now creating a function for measuring the accuracy of the spam filter

# In[19]:


correct = 0
total = test_set.shape[0]
    
for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['Predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)


# Our Filter has a accuracy of 86% , as I was trying to achieving 80% accuracy so it looks pretty good.
# 
# I will updating the code for increasig the accuraccy in future.

# In{20]

message = input("Enter your message")
classify(message)
