# CS5710_11595_Homework_4
# Q1 NLP Preprocessing Pipeline (Tokenization, Stopwords, POS, Lemmatization)

This script demonstrates a simple **Natural Language Processing (NLP) preprocessing pipeline** using **NLTK**.

## What the Code Does

### 1. Downloads Required NLTK Resources

The script downloads tokenizers, stopwords, POS taggers, WordNet data, and mapping files required for processing text.

### 2. Tokenizes Input Text

The text is split into individual words using `word_tokenize()`.

### 3. Removes Stopwords

Common English stopwords (like *the, in, while, and*) are removed to keep only meaningful words.

### 4. POS Tagging

Each remaining word is assigned a **Part of Speech (POS)** tag (noun, verb, adjective, adverb).

### 5. Lemmatization Using POS Tags

Each word is converted into its base form (lemma) using WordNet lemmatizer, guided by accurate POS tags.

### 6. Filters Only Nouns and Verbs

From the lemmatized output, only **nouns** and **verbs** are kept.

## Output Includes

* Tokens
* Tokens without stopwords
* POS tags
* Lemmatized tokens
* Final cleaned tokens (verbs + nouns only)

## Purpose

This script provides a clear, hands-on demonstration of how raw text can be cleaned, normalized, and prepared for NLP tasks such as text classification, information extraction, or semantic analysis.


#Q2 Named Entity Recognition & Pronoun Ambiguity Detection

This script performs **Named Entity Recognition (NER)** using a transformer-based model and then checks the text for ambiguous pronouns that may cause confusion.

## What the Code Does

### 1. Loads a Pretrained NER Model

Uses Hugging Face’s `dslim/bert-base-NER` model with the `pipeline()` API and simple aggregation to group multi-word entities.

### 2. Performs Named Entity Recognition

Extracts entities such as:

* **Person names**
* **Organizations**
* **Locations**
* **Miscellaneous entities**

For each entity, the script prints:

* Extracted text
* Entity type
* Confidence score

### 3. Detects Pronoun Ambiguity

The script scans the text for pronouns like *he, she, him, her, they,* etc.
If found, it warns that the sentence may contain ambiguous references.

## Output Includes

* List of identified named entities
* Their types and confidence values
* A warning if ambiguous pronouns appear

## Purpose

This code demonstrates how transformer models can extract semantic information from text and highlights potential ambiguity issues useful for tasks like coreference resolution, text analysis, or conversational AI.
