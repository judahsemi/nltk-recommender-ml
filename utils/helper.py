import os
from collections import Counter

import nltk
import numpy as np
import pandas as pd

from nltk.wsd import lesk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn



### Comment out to download them if you don't have them
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("sentiwordnet")

### If you already have them append their path here
nltk.data.path.append(os.path.abspath("./data/nltk/"))

ps = PorterStemmer()
wn_lemmatizer = nltk.WordNetLemmatizer()
tag_to_pos = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}


def lemmatize(token, tag):
    """ Lemmatize token """
    return wn_lemmatizer.lemmatize(token, tag_to_pos.get(tag[0], wn.NOUN))


def extract_nouns(text_tokens):
    """ Extract words that are nouns """
    # Tag each token with it's part-of-speech, remove stopwords, extract nouns, and lemmatize
    tokens = nltk.pos_tag(text_tokens)
    tokens = [t for t in tokens if t[0] not in sw.words()]
    noun_tokens = [lemmatize(*t) for t in tokens if t[1].upper().startswith("NN")]
    return noun_tokens


def get_related_synsets(synset):
    """ Returns noun-synsets that are semantically-related to synset """
    related_synsets = set()
    
    for l in synset.lemmas():
        if not l:
            continue

        # Add lemmas (could be the same thing as synset)
        related_synsets.add(l.synset())
        
        # Add antonyms
        related_synsets.update([a.synset() for a in l.antonyms() if a])
        
    # Add hyponyms, hypernyms, holonyms, meronyms, and entailments
    related_synsets.update([s for s in synset.hyponyms() if s])
    related_synsets.update([s for s in synset.hypernyms() if s])
    related_synsets.update([s for s in synset.part_holonyms() if s])
    related_synsets.update([s for s in synset.part_meronyms() if s])
    related_synsets.update([s for s in synset.entailments() if s])
    return list(related_synsets)


def is_similar_to(a_synset, b_synset, use_relatives=True, threshold=0.5, verbose=True):
    """ Check if b_synset is similar to a_synset """
    # Check if b_synset and a_synset are similar
    score = a_synset.wup_similarity(b_synset)
    score = score if score else 0
    match = a_synset
    
    # If they are not similar enough use a_synset relatives
    if score < threshold and use_relatives:
        for a_relative in get_related_synsets(a_synset):
            rel_score = a_relative.wup_similarity(b_synset)
            rel_score = rel_score if rel_score else 0
            
            # Check if a_synset relative and b_synset are similar
            if rel_score > score:
                score = rel_score
                if score >= threshold:
                    match = a_relative
                    break
                    
    if verbose:
        print("a: {}, b: {}, score: {:.3f}, thres: {:.3f}".format(
            a_synset.name(), b_synset.name(), score, threshold))
    return score >= threshold, score, match


def cluster_synsets(synsets, use_relatives=True, threshold=0.5, verbose=False):
    """ Group a list of synsets that are similar to each other """
    curr_cluster_num = 1
    clusters = {curr_cluster_num: [synsets.pop(0)]}
    
    # Repeat process until all synset as been placed in a cluster
    while synsets:
        has_changed = False
        synsets_in_cluster = clusters[curr_cluster_num]
        
        # For each remaining synsets check if they belong to the current cluster
        for i, syn in enumerate(synsets):
            for syn_b in synsets_in_cluster:
                if is_similar_to(syn_b, syn, use_relatives, threshold, verbose)[0]:
                    clusters[curr_cluster_num].append(synsets.pop(i))
                    has_changed = True
                    break
            if has_changed:
                break
                
        # If none belongs to the current cluster create a new one
        if synsets and not has_changed:
            curr_cluster_num += 1
            clusters[curr_cluster_num] = [synsets.pop(0)]
    return clusters


def calculate_text_sentiment(text, verbose=True):
    """ Calculate the sum sentiment of words in text_tokens """
    sentiment_score = 0

    # Split text into sentence for better sentiment analysis
    sent_tokens = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text.lower())]

    # If each sentence in text calculate it's sentiment and add to overall sentiment
    for tokens in sent_tokens:
        # keep track of "not" streak
        not_streak = 0

        tokens_and_tags = nltk.pos_tag(tokens)
        for i, t_and_tag in enumerate(tokens_and_tags):
            t, tag = t_and_tag

            apply_negate = False

            # If token is a "not" add *-2 the sentiment of the next/prev token
            if t in ["n't", "not"]:
                not_streak += 1
                t_synset = None

                # Use the next token if token is not last token
                if i+1 < len(tokens_and_tags):
                    next_t, next_tag = tokens_and_tags[i+1]
                    if next_t in ["n't", "not"]:
                        continue

                    t_synset = lesk(tokens, next_t, pos=tag_to_pos.get(next_tag[0], None))

                # Use the previous token if token is not first token
                if not t_synset and i > 0:
                    prev_t, prev_tag = tokens_and_tags[i-1]

                    # If this is the last token (streak is definately > 0), apply not_streak
                    if prev_t in ["n't", "not"]:
                        apply_negate = True
                        prev_t, prev_tag = tokens_and_tags[i-not_streak]

                    t_synset = lesk(tokens, prev_t, pos=tag_to_pos.get(prev_tag[0], None))

                # Add "not" sentiment score if it is the only word.
                if not t_synset:
                    sentiment_score += -0.625
                    continue

                t_senti = swn.senti_synset(t_synset.name())
                if apply_negate and not_streak%2 == 0:
                    not_streak = 0
                    continue

                elif apply_negate:
                    score = -2 * (t_senti.pos_score() - t_senti.neg_score())

                else:
                    score = 2 * ((-1)**(not_streak+1)) * (t_senti.pos_score() - t_senti.neg_score())

                not_streak = 0
                score = -0.625 if score == 0 else score

            else:
                t_synset = lesk(tokens, t, pos=tag_to_pos.get(tag[0], None))
                if not t_synset:
                    continue

                t_senti = swn.senti_synset(t_synset.name())
                score = (t_senti.pos_score() - t_senti.neg_score())
                not_streak = 0

            # Calculate token's sentiment
            sentiment_score += score
        
    if verbose:
        print("sentiment: {:>7.2f}; {}".format(sentiment_score, sent_tokens))
    return sentiment_score


def score_clusters(clusters, comments_noun_synsets, comments, verbose=True):
    """ Score clusters """
    # Map each synset to it's cluster; reverse of clusters
    synset_to_cluster = {syn: n for n, synsets in clusters.items() for syn in synsets}
    
    # Map each comment to it's sentiment score
    comment_to_senti = {c: calculate_text_sentiment(c, verbose) for c in comments}
    
    # Map each comment to clusters
    cluster_to_comments = {n: [] for n in clusters}
    for c_synsets, c in zip(comments_noun_synsets, comments):
        for syn in c_synsets:
            if syn not in synset_to_cluster or c in cluster_to_comments[synset_to_cluster[syn]]:
                continue
            cluster_to_comments[synset_to_cluster[syn]].append(c)
            
    # Score each cluster
    cluster_to_score = {cluster: sum([comment_to_senti[c] for c in cluster_comments])\
        for cluster, cluster_comments in cluster_to_comments.items()}
    return cluster_to_score


def extract_user_prefs(comments, use_relatives=True, threshold=0.5, verbose=True):
    """ Extract user preferences based on their previous comments or reviews """
    # Tokenize: Split text into a list of words
    each_comment_tokens = [nltk.word_tokenize(c.lower()) for c in comments]

    # Extract the nouns in each comment
    each_comment_noun_synsets = [
        [lesk(c, n, pos=wn.NOUN) for n in extract_nouns(c)] for c in each_comment_tokens]

    if verbose:
        print("EXTRACTED NOUNS:\n=====\n\n", each_comment_noun_synsets)

    # Cluster noun synsets
    all_synsets = set()
    for synsets in each_comment_noun_synsets:
        all_synsets.update([syn for syn in synsets if syn])
    all_synsets = sorted(list(all_synsets))
    
    clusters = cluster_synsets(all_synsets, use_relatives, threshold, verbose)

    if verbose:
        print("\nCLUSTERS:\n=====\n")
        for k, v in clusters.items():
            print("cluster {}: count={}, {}".format(k, len(v), v))

    # Score clusters and get preferneces
    cluster_to_score = score_clusters(clusters, each_comment_noun_synsets, comments, verbose)
    max_score = max(list(cluster_to_score.values()))
    
    user_prefs = []
    for cluster, score in cluster_to_score.items():
        if score == max_score:
            user_prefs.extend(clusters[cluster])

    if verbose:
        print("\n\nCLUSTERS SCORE:\n=====\n")
        for k, v in cluster_to_score.items():
            print("cluster: {}, score: {}".format(k, v))
        
    print("\n\nUSER PREFERENCES:\n=====\n\n", user_prefs)
    return user_prefs


def extract_sites_feats(site_and_reviews):
    site_and_feats = {}
 
    for site, reviews in site_and_reviews.items():
        noun_list = []
 
        # For each review extract it's nouns and add it to a compilation list
        for r in reviews:
            r_nouns = extract_nouns(nltk.word_tokenize(r))
            noun_list.extend(r_nouns)
 
        # Return the 5 most common nouns which is the tourist site's features
        site_and_feats[site] = [lesk(nltk.word_tokenize(r), w[0], pos=wn.NOUN) for w in Counter(noun_list).most_common(5)]
 
    print("SITE FEATURES:\n\n", site_and_feats)
    return site_and_feats


def calc_site_score(user_prefs, site_feats):
    max_score_list = []
 
    # Calculate the maximum similarity between each user_prefs and all the site_feats
    for pref_synset in user_prefs:
        if not pref_synset:
            continue
 
        max_score = max([pref_synset.wup_similarity(ft_synset) for ft_synset in site_feats if ft_synset])
        max_score_list.append(max_score)
 
    # Calculate the average of the max similarities; this represent the site's score
    return sum(max_score_list) / len(max_score_list)


def make_recommendation(user_prefs, site_and_feats):
    site_and_score = {}
    best_score = float("-inf")
    best_site = None
 
    for site, site_feats in site_and_feats.items():
        site_and_score[site] = calc_site_score(user_prefs, site_feats)
        if site_and_score[site] > best_score:
            best_score = site_and_score[site]
            best_site = site
 
    print("Best score and site with the best score:\n\n", round(best_score, 4), best_site)
    print("\n\nSites sorted by score:\n\n", sorted(site_and_score.items(), key=lambda t: t[1], reverse=True))
    return site_and_score

