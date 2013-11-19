# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import cPickle as pickle
from time import gmtime, strftime, time
import os, warnings, collections, csv, nltk, re, requests, zipfile
from dateutil import parser

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.lda import LDA
from sklearn.preprocessing import Normalizer

from gensim import corpora, models, similarities

warnings.filterwarnings('ignore')

project_dir = './project/'
structured_dir = project_dir + 'structured/'
keyword_dir = project_dir + 'keyword/'
linguistic_dir = project_dir + 'linguistic/'
topic_dir = project_dir + 'topic/'
svd_description_dir = project_dir + 'svd_description/'
svd_summary_dir = project_dir + 'svd_summary/'
combined_dir = project_dir + 'combined/'

all_dirs = [project_dir, structured_dir, keyword_dir, linguistic_dir, topic_dir,
            svd_description_dir, svd_summary_dir, combined_dir]

for each_dir in all_dirs:
    if not os.path.exists(each_dir):
        os.makedirs(each_dir)

english_stemmer = nltk.stem.SnowballStemmer('english')
english_lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
english_stopwords = nltk.corpus.stopwords.words('english')
punctuation = re.compile(r'[.?!,":;()#\'|0-9]')

def get_stemmed_exlude_tokens():
    exclude_tokens = ['die', 'dying', 'death', 'dead', 'decease', 'killed', 'demise', 'suffocate', 'electrocute',
                      'drown', 'fatal', 'asphyxiate', 'asphyxia', 'employee', 'hospital']
    stemmed_exclude_tokens = set([english_stemmer.stem(t) for t in exclude_tokens])
    return stemmed_exclude_tokens

def get_classifiers():
    clfs = {'AdaBst': AdaBoostClassifier(), 'LogReg': LogisticRegression(), 'RanFst': RandomForestClassifier(),
            'DTree': DecisionTreeClassifier(), 'KNN': KNeighborsClassifier()}
    return clfs

def get_partition_sets(df_struct, test_set_proportion):
    validation_set = df_struct[df_struct.event_year > 2008].fatality_ind
    train_test_set = df_struct[df_struct.event_year <= 2008].fatality_ind
    train_test_set_shuffled = train_test_set.reindex(np.random.permutation(train_test_set.index))
    train_cutoff = len(train_test_set)-int((test_set_proportion * len(train_test_set)))
    train_set = train_test_set_shuffled[:train_cutoff]
    test_set = train_test_set_shuffled[train_cutoff:]
    train_set = pd.DataFrame({'fatality_ind':train_set, 'partition':'train'}, index = train_set.index)
    test_set = pd.DataFrame({'fatality_ind':test_set, 'partition':'test'}, index = test_set.index)
    validation_set = pd.DataFrame({'fatality_ind':validation_set, 'partition':'validation'}, index = validation_set.index)
    df_partition = pd.concat([train_set, test_set, validation_set])
    df_partition.index.name = 'summary_nr'
    df_partition.sort_index(inplace=True)
    return df_partition

def get_partition(df, partition):
    return df[df.partition == partition]

def get_top_k_features(df, score_fn, k):
    X_train = get_partition(df,'train').drop(['fatality_ind','partition'],axis=1)
    y_train = get_partition(df,'train').fatality_ind
    selector = SelectKBest(score_func=score_fn,k=k)
    selector.fit(X_train, y_train)
    scores = -np.log10(selector.pvalues_)
    df_top_features = pd.DataFrame({'scores':scores, 'feature':X_train.columns}).sort('scores',ascending=False)
    return df_top_features[:k]


def get_excluded_words():
    lst = ['asphyxia', 'asphyxiant', 'asphyxiate', 'asphyxiated', 'asphyxiates', 'asphyxiating', 'asphyxiation', 'dead',
           'deadly', 'death', 'deaths', 'decease', 'deceased', 'demise', 'die', 'died', 'dies', 'drown', 'drowned',
           'drownes', 'drowning', 'drowns', 'dying', 'electrocute', 'electrocuted', 'electrocutes', 'electrocuting',
           'electrocution', 'electrocutions', 'employee', 'employees', 'employee\'s', 'employee #1', 'employee #1\'s',
           'fatal', 'fatalities', 'fatality', 'fatally', 'hospitable', 'hospital', 'hospitality', 'hospitalization',
           'hospitalizations', 'hospitalize', 'hospitalized', 'hospitalizes', 'hospitalizing', 'hospitals', 'kill',
           'kille', 'killed', 'killing', 'kills', 'suffocate', 'suffocated', 'suffocates', 'suffocating', 'suffocation']
    lst.sort(key=len, reverse=True)
    return lst


def get_accident_detail_url(inspection_id):
    accident_detail_url_prefix = 'https://www.osha.gov/pls/imis/establishment.inspection_detail?id='
    return accident_detail_url_prefix + str(inspection_id)


def clean_keyword(keywords, empty_return_phrase='___EMPTY___'):
    exclude_words = get_excluded_words()
    kws = keywords.lower().split(',')
    cleaned = ','.join([k for k in kws if k not in exclude_words])
    if cleaned == '':
        cleaned = empty_return_phrase
    return cleaned


def clean_text(raw_txt, empty_return_phrase='___EMPTY___'):
    exclude_words = get_excluded_words()
    txt = raw_txt.lower()
    for word in exclude_words:
        txt = txt.replace(word, '')
    cleaned = ' '.join(txt.split()).strip()
    if cleaned == '':
        cleaned = empty_return_phrase
    return cleaned


def get_description_vectorizer_info():
    vlist = [['Description - Stem - Ngrams 1 Only', 'desc_stem_n1', 5, (1, 1), 'stem'],
             ['Description - Stem - Ngrams 1,2,3', 'desc_stem_n123', 5, (1, 3), 'stem'],
             ['Description - Lemma - Ngrams 1 Only', 'desc_lem_n1', 5, (1, 1), 'lemma'],
             ['Description - Lemma - Ngrams 1,2,3', 'desc_lem_n123', 5, (1, 3), 'lemma']]
    return vlist

def get_summary_vectorizer_info():
    vlist = [['Summary - Stem - Ngrams 1 Only', 'summ_stem_n1', 20, (1, 1), 'stem'],
             ['Summary - Stem - Ngrams 1,2,3', 'summ_stem_n123', 20, (1, 3), 'stem'],
             ['Summary - Lemma - Ngrams 1 Only', 'summ_lem_n1', 20, (1, 1), 'lemma'],
             ['Summary - Lemma - Ngrams 1,2,3', 'summ_lem_n123', 20, (1, 3), 'lemma']]
    return vlist

def get_vectorizer(min_df, ngram_range, morphtype):
    if morphtype == 'lemma':
        return LemmatizedTfidfVectorizer(min_df=min_df, stop_words=english_stopwords, ngram_range=ngram_range)
    else:
        return StemmedTfidfVectorizer(min_df=min_df, stop_words=english_stopwords, ngram_range=ngram_range)


def plot_accident_by_year(df_acc, norm=False, save=False):
    xtab = pd.crosstab(df_acc.event_year, df_acc.fatality_ind)
    xtab.columns = ['Not Fatal', 'Fatal']
    xtab.columns.name = 'Accident Outcome'
    ylabel = 'Accident Count'
    if norm:
        xtab = xtab.div(xtab.sum(axis=1).astype('float'), axis=0)
        ylabel = 'Accident Percent'
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    xtab.plot(kind='bar', ax=axes, stacked=True, alpha=0.5, sort_columns=True)
    axes.set_title('OSHA Accident Investigations')
    axes.set_ylabel(ylabel)
    axes.set_xlabel('Event Year')
    if save:
        plt.savefig(structured_dir + 'osha_accident_by_year_plot.png')


def plot_time_visuals(df_struct, save=False):
    time_vars = ['event_year', 'event_month', 'event_weekday', 'event_hour']
    time_labels = [[str(x)[2:] for x in range(1990, 2013)],
                   ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                   ['12a'] + [str(x) + 'a' for x in range(1, 12)] + ['12p'] + [str(x - 12) + 'p' for x in
                                                                               range(13, 24)]]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    xtab_col = np.array(time_vars)
    lbls_col = np.array(time_labels)
    for i in range(4):
        xtab = pd.crosstab(df_struct[xtab_col[i]], df_struct.fatality_ind)
        xtab.columns = ['Not Fatal', 'Fatal']
        #xtab.columns.name='Accident Outcome'
        norm_xtab = xtab.div(xtab.sum(axis=1).astype('float'), axis=0)
        xtab.plot(kind='bar', ax=axes[i, 0], alpha=0.5, legend=False, sort_columns=True)
        norm_xtab.plot(kind='bar', stacked=True, ax=axes[i, 1], alpha=0.5, sort_columns=True)
        xlabel = xtab_col[i].replace('_', ' ').title()
        axes[i, 0].set_title('Accident Outcome by %s' % xlabel)
        axes[i, 1].set_title('Normalized Histogram of %s' % xlabel)
        axes[i, 0].set_xlabel(xlabel)
        axes[i, 1].set_xlabel(xlabel)
        axes[i, 0].set_ylabel('Accident Count')
        axes[i, 1].set_ylabel('Portion of Accidents')
        axes[i, 1].legend(loc="lower left")
        axes[i, 0].set_xticklabels(lbls_col[i], rotation=(60 if i == 3 else 0))
        axes[i, 1].set_xticklabels(lbls_col[i], rotation=(60 if i == 3 else 0))
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    if save:
        plt.savefig(structured_dir + 'osha_time_plots.png')


def plot_categorical_heatmaps(df_struct, top_n=30, norm=False, save=False):
    struct_cat_vars = ['nature_of_inj', 'part_of_body', 'src_of_injury', 'event_type', 'evn_factor',
                       'hum_factor', 'occ_code', 'sic_desc']
    vars_desc = {'site_state': 'Site State', 'site_city': 'Site City', 'nature_of_inj': 'Nature of Injury',
             'part_of_body': 'Part of Body', 'src_of_injury': 'Source of Injury', 'event_type': 'Event Type',
             'evn_factor': 'Event Factor', 'hum_factor': 'Human Factor', 'occ_code': 'Occupation',
             'sic_desc': 'SIC Group Description'}
    for i, cat_var in enumerate(struct_cat_vars):
        df_filter = df_struct[df_struct.event_year <= 2008]
        df = df_filter.pivot_table('fatality_ind', rows=[cat_var], cols=['event_year'], aggfunc='sum').fillna(0)
        df.index.name = cat_var
        num_cats = df.shape[0]
        if norm:
            df = (df - df.mean()) / (df.max() - df.min())
        df_sort = df.sort_index(by=[2008], ascending=False)[:top_n].T
        fig, ax = plt.subplots()
        im = ax.pcolor(df_sort, cmap=plt.cm.YlOrRd)
        cbar = fig.colorbar(im)
        cbar.set_label('# of Fatalities', rotation=90, size=16)
        if num_cats <= top_n:
            prefix = 'All ' + str(num_cats)
        else:
            prefix = 'Top ' + str(top_n)
        ax.set_xlabel(vars_desc[cat_var].upper() + ' - ' + prefix + ' Categories', size=20)
        fig.set_size_inches(12, 5)
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(df_sort.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(df_sort.shape[1]) + 0.5, minor=False)
        ax.xaxis.tick_top()
        ax.set_xticklabels(df_sort.columns, minor=False)
        ax.set_yticklabels(df_sort.index, minor=False)
        plt.xticks(rotation=45, ha='left')
        ax.grid(False)
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        if save:
            plt.savefig(structured_dir + cat_var + '_heatmap.png')


def plot_categorical_distribution_charts(df_struct, top_n=10, save=False):
    struct_cat_vars = ['nature_of_inj', 'part_of_body', 'src_of_injury', 'event_type', 'evn_factor',
                       'hum_factor', 'occ_code', 'sic_desc']
    vars_desc = {'site_state': 'Site State', 'site_city': 'Site City', 'nature_of_inj': 'Nature of Injury',
             'part_of_body': 'Part of Body', 'src_of_injury': 'Source of Injury', 'event_type': 'Event Type',
             'evn_factor': 'Event Factor', 'hum_factor': 'Human Factor', 'occ_code': 'Occupation',
             'sic_desc': 'SIC Group Description'}
    for i, cat_var in enumerate(struct_cat_vars):
        df = df_struct.reset_index().pivot_table('summary_nr', rows=[cat_var], cols=['fatality_ind'],
                                                 aggfunc='count').fillna(0)
        df.columns = ['Not Fatal', 'Fatal']
        num_cats = df.shape[0]
        df['Total Accidents'] = df['Not Fatal'] + df['Fatal']
        df.columns.name = 'Accident Outcome'
        df_top = df.sort_index(by='Total Accidents', ascending=False)[:top_n]
        df_rest = df.sort_index(by='Total Accidents', ascending=False)[top_n:]
        df_rest = pd.DataFrame(df_rest.sum(), columns=['*** ALL OTHER CATEGORIES ***']).T
        df_comb = pd.concat([df_top, df_rest])
        df_comb = df_comb.sort_index(by=['Total Accidents'])
        df_comb = df_comb.drop(['Total Accidents'], axis=1)
        df_norm = df_comb.div(df_comb.sum(axis=1).astype('float'), axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        fig.subplots_adjust(wspace=0.001)
        subtitle = 'Top ' + str(min(num_cats, top_n)) + ' Categories Plus All Other'
        df_comb.plot(kind='barh', stacked=True, ax=axes[0], alpha=0.5,
                     title=vars_desc[cat_var].upper() + '\n' + subtitle, sort_columns=True)
        df_norm.plot(kind='barh', stacked=True, ax=axes[1], alpha=0.4,
                     title='Normalized Accident Count\n ', sort_columns=True)
        axes[0].legend(loc='bottom right')
        axes[1].legend().set_visible(False)
        axes[0].set_xlabel('Accident Count (000\'s)')
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda k, pos: ('%.0f') % (k * 1e-3)))
        axes[1].xaxis.tick_top()
        axes[1].set_xticks(np.arange(0, 1.1, 0.1))
        axes[1].set_yticklabels([])
        if save:
            plt.savefig(structured_dir + cat_var + '_distribution_chart.png')


def get_senti_scores():
    senti_scores = collections.defaultdict(list)
    with open('SentiWordNet_3.0.0_20130122.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            for term in SynsetTerms.split(" "):
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term)
                senti_scores[key].append((float(PosScore), float(NegScore)))
    for key, value in senti_scores.iteritems():
        senti_scores[key] = np.mean(value, axis=0)
    return senti_scores


def extract_linguistic_features(documents):
    senti_scores = get_senti_scores()
    feat_list = []
    columns = ['len_doc', 'num_sents', 'num_chars', 'num_tokens', 'num_unique_tokens', 'num_alpha_tokens',
               'avg_pos_val', 'avg_neg_val', 'num_nouns', 'num_adjectives', 'num_verbs', 'num_adverbs',
               'num_female_prps', 'num_male_prps']
    female_preps = ['she', 'her', 'hers', 'herself']
    male_preps = ['he', 'him', 'his', 'himself']

    for document in documents:

        len_doc, num_sents, num_chars, num_tokens, num_unique_tokens, num_alpha_tokens, avg_pos_val, \
        avg_neg_val = 0, 0, 0, 0, 0, 0, 0, 0
        num_nouns, num_adjectives, num_verbs, num_adverbs, num_female_prps, num_male_prps = 0, 0, 0, 0, 0, 0

        tokens_list = []
        pos_vals = []
        neg_vals = []

        len_doc = len(document)
        sents = nltk.sent_tokenize(document)
        for sent in sents:
            num_sents += 1
            tokens = nltk.word_tokenize(sent)
            tokens = [punctuation.sub('', w) for w in tokens]
            tokens = [w.lower() for w in tokens if len(w) > 0]
            tags = nltk.pos_tag(tokens)
            for token, tag in tags:
                num_tokens += 1
                num_chars += len(token)
                pos_type = None
                p, n = 0, 0
                if tag.startswith('NN'):
                    num_nouns += 1
                    pos_type = 'n'
                elif tag.startswith('JJ'):
                    num_adjectives += 1
                    pos_type = 'a'
                elif tag.startswith('VB'):
                    num_verbs += 1
                    pos_type = 'v'
                elif tag.startswith('RB'):
                    num_adverbs += 1
                    pos_type = 'r'
                if token.isalpha():
                    num_alpha_tokens += 1
                if token.lower() in female_preps:
                    num_female_prps += 1
                if token.lower() in male_preps:
                    num_male_prps += 1
                if pos_type is not None:
                    sent_word = '%s/%s' % (pos_type, token)
                    if sent_word in senti_scores:
                        p, n = senti_scores[sent_word]
                pos_vals.append(p)
                neg_vals.append(n)
            tokens_list.extend(tokens)
        num_unique_tokens = len(set(tokens_list))
        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)

        feat_list.append(
            [len_doc, num_sents, num_chars, num_tokens, num_unique_tokens, num_alpha_tokens, avg_pos_val, avg_neg_val,
             num_nouns, num_adjectives, num_verbs, num_adverbs, num_female_prps, num_male_prps])
    return pd.DataFrame(feat_list, columns=columns)


def ie_preprocess(document, morphtype='none', stopwords='yes'):
    token_list = []
    sents = nltk.sent_tokenize(document)
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        tokens = [punctuation.sub('', w) for w in tokens]
        tokens = [w.lower() for w in tokens if w.isalpha() and len(w) > 2]
        if stopwords == 'yes':
            tokens = [w for w in tokens if w not in english_stopwords]
        if morphtype == 'lemma':
            tokens = [english_lemmatizer.lemmatize(w) for w in tokens]
        elif morphtype == 'stem':
            tokens = [english_stemmer.stem(w) for w in tokens]
        else:
            pass
        token_list.extend(tokens)
    return token_list


def join_abstract_lines(group):
    return ' '.join(group.abstract_text).strip()

def get_description_token_table(df_txt):
    desc_tokens_list = []
    stemmed_exclude_tokens = get_stemmed_exlude_tokens()
    columns = ['summary_nr', 'fatality_ind', 'var_name', 'token', 'stem', 'excluded']
    for i, txt in enumerate(zip(df_txt.index, df_txt.fatality_ind, df_txt.event_desc)):
        tokens = ie_preprocess(txt[2])
        for token in tokens:
            stemmed_token = english_stemmer.stem(token)
            if stemmed_token in stemmed_exclude_tokens:
                excluded = 1
            else:
                excluded = 0
            desc_tokens_list.append([txt[0], txt[1], 'desc', token, stemmed_token, excluded])
    df_desc_tokens = pd.DataFrame(desc_tokens_list, columns=columns).set_index('summary_nr')
    return df_desc_tokens


def get_summary_token_table(df_txt):
    summ_tokens_list = []
    stemmed_exclude_tokens = get_stemmed_exlude_tokens()
    columns = ['summary_nr', 'fatality_ind', 'var_name', 'token', 'stem', 'excluded']
    for i, txt in enumerate(zip(df_txt.index, df_txt.fatality_ind, df_txt.summary_txt)):
        tokens = ie_preprocess(txt[2])
        for token in tokens:
            stemmed_token = english_stemmer.stem(token)
            if stemmed_token in stemmed_exclude_tokens:
                summ_tokens_list.append([txt[0], txt[1], 'summ', token, stemmed_token, 1])
    df_summ_tokens = pd.DataFrame(summ_tokens_list, columns=columns).set_index('summary_nr')
    return df_summ_tokens


def plot_svd_concepts(df, partition, svd_name, graphs_dir, save=False):
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('SVD Component Scatterplots of Top Four Concepts\n' + svd_name + ' - ' + partition.title(), size=20)
    for row, col, cx, cy in [(0, 0, 1, 2), (1, 0, 1, 3), (2, 0, 1, 4), (0, 1, 2, 3), (1, 1, 2, 4), (2, 1, 3, 4)]:
        axes[row, col].scatter(df[(df.partition == partition) & (df.fatality_ind == 0)]['SVD_Comp_' + str(cx)],
                               df[(df.partition == partition) & (df.fatality_ind == 0)]['SVD_Comp_' + str(cy)],
                               c='g', label='Non-Fatal', alpha=0.01)
        axes[row, col].scatter(df[(df.partition == partition) & (df.fatality_ind == 1)]['SVD_Comp_' + str(cx)],
                               df[(df.partition == partition) & (df.fatality_ind == 1)]['SVD_Comp_' + str(cy)],
                               c='r', label='Fatal', alpha=0.01)
        axes[row, col].set_xlabel('Concept ' + str(cx))
        axes[row, col].set_ylabel('Concept ' + str(cy))
        #fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
        plt.subplots_adjust(top=0.9)
        if save:
            fig.savefig(graphs_dir + 'svd_concept_graphs_' + partition + '_' + svd_name + '.png')


def process_model_batch(feature_sets, clfs, batch_dir):
    met_list = []
    res_list = []
    cols_dict = {}

    for fset_name, df in feature_sets.items():
        print fset_name + ' begin: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())

        cols_dict[fset_name] = df.drop(['partition', 'fatality_ind'], axis=1).columns

        X_train = df[df.partition == 'train'].drop(['partition', 'fatality_ind'], axis=1)
        y_train = df[df.partition == 'train'].fatality_ind
        X_test = df[df.partition == 'test'].drop(['partition', 'fatality_ind'], axis=1)
        y_test = df[df.partition == 'test'].fatality_ind

        for model_name, model in clfs.items():
            print '.....' + model_name + ' begin: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())

            pair_name = fset_name + ' - ' + model_name
            try:
                t0 = time()
                model.fit(X_train, y_train)
                duration = time() - t0
                test_pred = model.predict(X_test)
                test_pred_prob = model.predict_proba(X_test)
                cm = confusion_matrix(y_test, test_pred)

                pickle.dump(model, open(batch_dir + pair_name + '.p', 'wb'))

                met_list.append([pair_name, fset_name, model_name, duration, round(np.mean(y_test) * 100, 1),
                                 accuracy_score(y_test, test_pred), precision_score(y_test, test_pred),
                                 recall_score(y_test, test_pred), f1_score(y_test, test_pred),
                                 cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]])

                dfr = pd.DataFrame({'summary_nr': X_test.index, 'pair_name': pair_name, 'fset_name': fset_name,
                                    'model_name': model_name, 'target_ind': y_test, 'target_pred': test_pred,
                                    'pred_0_prob': test_pred_prob[:, 0], 'pred_1_prob': test_pred_prob[:, 1]})
                res_list.append(dfr)
            except:
                print '..... ### process model error: ', fset_name, model_name

            print '.....' + model_name + ' end: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print fset_name + ' end: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print ''

    df_met = pd.DataFrame(met_list, columns=['pair_name', 'fset_name', 'model_name', 'duration', 'act_target_pct',
                                             'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'TN',
                                             'FN', 'FP', 'TP'])
    df_res = pd.concat(res_list).set_index('summary_nr')

    pickle.dump(df_res, open(batch_dir + 'df_results.p', 'wb'))
    pickle.dump(df_met, open(batch_dir + 'df_metrics.p', 'wb'))
    pickle.dump(cols_dict, open(batch_dir + 'feature_set_columns.p', 'wb'))


def get_combined_separate_fsets(feature_sets, fs_fn='pct', ptile=10, nFeatures=5, score_fn=f_classif):
    df_lst = []
    for fset_name, df in feature_sets.items():
        X_train = df[df.partition == 'train'].drop(['partition', 'fatality_ind'], axis=1)
        y_train = df[df.partition == 'train'].fatality_ind
        df_X = df.drop(['partition', 'fatality_ind'], axis=1)
        if fs_fn == 'pct':
            featureSelector = SelectPercentile(score_func=score_fn, percentile=ptile)
        else:
            featureSelector = SelectKBest(score_func=score_fn, k=nFeatures)
        featureSelector.fit(X_train, y_train)
        fs = featureSelector.transform(df.drop(['partition', 'fatality_ind'], axis=1))
        cols_fs = df_X.columns[list(featureSelector.get_support(indices=True))]
        cols_fs_ref = [fset_name + ' ' + c for c in cols_fs]
        df_fs = pd.DataFrame(fs, index=df_X.index, columns=cols_fs_ref)
        df_lst.append(df_fs)
    df_comb = df[['partition', 'fatality_ind']].join(pd.concat(df_lst, axis=1))
    return df_comb


def get_combined_united_fsets(feature_sets, fs_fn='pct', ptile=10, nFeatures=5, score_fn=f_classif):
    df_lst = []
    for fset_name, df in feature_sets.items():
        df_X = df.drop(['partition', 'fatality_ind'], axis=1)
        df_X.columns = [fset_name + ' ' + c for c in df_X.columns]
        df_lst.append(df_X)
    df_comb = df[['partition', 'fatality_ind']].join(pd.concat(df_lst, axis=1))
    X_train = df_comb[df_comb.partition == 'train'].drop(['partition', 'fatality_ind'], axis=1)
    y_train = df_comb[df_comb.partition == 'train'].fatality_ind
    if fs_fn == 'pct':
        featureSelector = SelectPercentile(score_func=score_fn, percentile=ptile)
    else:
        featureSelector = SelectKBest(score_func=score_fn, k=nFeatures)
    featureSelector.fit(X_train, y_train)
    fs = featureSelector.transform(df_comb.drop(['partition', 'fatality_ind'], axis=1))
    cols_fs = df_comb.drop(['partition', 'fatality_ind'], axis=1).columns[
        list(featureSelector.get_support(indices=True))]
    df_fs = pd.DataFrame(fs, index=df_comb.index, columns=cols_fs)
    return df_comb[['partition', 'fatality_ind']].join(df_fs)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class LemmatizedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmatizedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_lemmatizer.lemmatize(w) for w in analyzer(doc))


class LemmatizedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmatizedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_lemmatizer.lemmatize(w) for w in analyzer(doc))


def get_outcome(ind_pred):
    ind = ind_pred[0]
    pred = ind_pred[1]
    if ind == 0 and pred == 0:
        return 'TN'
    elif ind == 1 and pred == 1:
        return 'TP'
    elif ind == 1 and pred == 0:
        return 'FN'
    else:
        return 'FP'


def get_tree_model_info(fsets, clfs, model_dict, cols_dict):
    df_list = []
    for fset in fsets:
        for clf in clfs:
            if clf in ['RanFst', 'DTree']:
                pair_name = fset + ' - ' + clf
                try:
                    df = pd.DataFrame({'pair_name': pair_name, 'importance': model_dict[pair_name].feature_importances_,
                                       'feature': cols_dict[fset]})
                    df = df.sort_index(by=['importance'], ascending=False)
                    df_list.append(df)
                except:
                    print pair_name, ' model does not exist...'
    df_tree_info = pd.concat(df_list)
    return df_tree_info


def plot_tree_importance(fsets, clfs, top_n, model_dict, cols_dict, graph_files_dir, save=False):
    for fset in fsets:
        for clf in clfs:
            if clf in ['RanFst', 'DTree']:
                pair_name = fset + ' - ' + clf
                try:
                    fig = plt.figure()
                    tree_imp = pd.Series(model_dict[pair_name].feature_importances_,
                                         index=cols_dict[fset]).order(ascending=False)[:top_n]
                    tree_imp.order(ascending=True).plot(kind='barh', title='Feature Set: ' + fset + '\n' + clf +
                                                                           ' Feature Importance (Top ' + str(
                        top_n) + ')')
                    if save:
                        fig.savefig(graph_files_dir + 'feat imp - ' + pair_name + '.png')
                except:
                    print pair_name, ' model does not exist...'


def get_logreg_model_info(fsets, clfs, model_dict, cols_dict):
    df_list = []
    for fset in fsets:
        for clf in clfs:
            if clf in ['LogReg']:
                pair_name = fset + ' - ' + clf
                try:
                    df = pd.DataFrame({'pair_name': pair_name, 'coefficient': model_dict[pair_name].coef_[0].T,
                                       'feature': cols_dict[fset]})
                    df['intercept'] = model_dict[pair_name].intercept_[0]
                    df_list.append(df)
                except:
                    print pair_name, ' model does not exist...'
    df_logreg_info = pd.concat(df_list)
    return df_logreg_info


def plot_logreg_coef_importance(fsets, clfs, df_logreg, ends_n, model_dict, cols_dict, graph_files_dir, save=False):
    for fset in fsets:
        for clf in clfs:
            if clf in ['LogReg']:
                pair_name = fset + ' - ' + clf
                fig = plt.figure()
                df = df_logreg[df_logreg.pair_name == pair_name][['coefficient', 'feature']].set_index(['coefficient'])
                if len(df) > ends_n * 2:
                    df_neg = df.sort_index(ascending=True)[:ends_n]
                    df_pos = df.sort_index(ascending=False)[:ends_n]
                    df = pd.concat([df_neg, df_pos])
                    df = df.sort_index().reset_index().set_index(['feature'])
                    ax = df.plot(kind='barh', title='Feature Set: ' + fset + '\n' + clf + ' Feature Importance (Top '\
                                  + str(min(ends_n * 2, len(df))) + ')')
                    ax.set_ylabel('')
                    if save:
                        fig.savefig(graph_files_dir + 'feat imp - ' + pair_name + '.png')


def gains_curve(y_test, pos_class_probas):
    df = pd.DataFrame({'ind': y_test, 'prob': pos_class_probas})
    df['prob_rank'] = df['prob'].rank(method='first', ascending=False)
    df['n_grp'] = pd.qcut(df['prob_rank'], 100).labels + 1
    dfg = df.groupby('n_grp')['ind', 'obs'].sum()
    dfg['cum_actual'] = dfg['ind'].cumsum()
    dfg['pct_actual'] = dfg['cum_actual'].astype(float) / dfg['cum_actual'].max() * 100
    dfg = dfg.reset_index()
    return dfg.n_grp, dfg.pct_actual


def plot_gains_charts(df_result, classifiers, feature_sets, grouping, graph_files_dir, save=False, dfs=None, dfsn=''):
    target_mean = np.mean(df_result.target_ind)

    if grouping == 'fset':
        for fset in feature_sets:
            fig, axes = plt.subplots(1, 1, figsize=(10, 7))
            if dfs is not None:
                xi, yi = gains_curve(dfs.target_ind, dfs.pred_1_prob)
                axes.plot(xi, yi, label=dfsn)
            for clf in classifiers:
                try:
                    df = df_result[(df_result.model_name == clf) & (df_result.fset_name == fset)]
                    xi, yi = gains_curve(df.target_ind, df.pred_1_prob)
                    axes.plot(xi, yi, label=clf)
                except:
                    pass
            plt.title('Feature Set: ' + fset + '\nCumulative Gains Chart')
            plt.grid(True)
            plt.plot(range(100), range(100), 'r-.', lw=3, label='Baseline Model')
            path = Path([(0., 0.), (target_mean * 100, 100.), (100., 100.)])
            patch = patches.PathPatch(path, facecolor='black', alpha=0.05, lw=3, label='Best Model')
            axes.add_patch(patch)
            plt.yticks(range(0, 105, 5))
            plt.xticks(range(0, 105, 5))
            plt.xlabel('% Accidents')
            plt.ylabel('% Fatalities')
            plt.legend(loc="lower right")
            leg = plt.gca().get_legend()
            leg.set_title('Classification Model')
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize='small')
            if save:
                plt.savefig(graph_files_dir + 'gains chart - ' + fset + '.png')

    if grouping == 'clf':
        for clf in classifiers:
            fig, axes = plt.subplots(1, 1, figsize=(10, 7))
            if dfs is not None:
                xi, yi = gains_curve(dfs.target_ind, dfs.pred_1_prob)
                axes.plot(xi, yi, label=dfsn)
            for fset in feature_sets:
                try:
                    df = df_result[(df_result.model_name == clf) & (df_result.fset_name == fset)]
                    xi, yi = gains_curve(df.target_ind, df.pred_1_prob)
                    axes.plot(xi, yi, label=fset)
                except:
                    pass
            plt.title('Classification Model: ' + clf + '\nCumulative Gains Chart')
            plt.grid(True)
            plt.plot(range(100), range(100), 'r-.', lw=3, label='Baseline Model')
            path = Path([(0., 0.), (target_mean * 100, 100.), (100., 100.)])
            patch = patches.PathPatch(path, facecolor='black', alpha=0.05, lw=3, label='Best Model')
            axes.add_patch(patch)
            plt.yticks(range(0, 105, 5))
            plt.xticks(range(0, 105, 5))
            plt.xlabel('% Accidents')
            plt.ylabel('% Fatalities')
            plt.legend(loc="lower right")
            leg = plt.gca().get_legend()
            leg.set_title('Feature Set')
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize='small')
            if save:
                plt.savefig(graph_files_dir + 'gains chart - ' + clf + '.png')


def plot_precision_recall_charts(df_result, classifiers, feature_sets, grouping, graph_files_dir, save=False):
    if grouping == 'fset':
        for fset in feature_sets:
            fig, axes = plt.subplots(1, 1, figsize=(10, 7))
            for clf in classifiers:
                try:
                    df = df_result[(df_result.model_name == clf) & (df_result.fset_name == fset)]
                    precision, recall, thresholds = precision_recall_curve(df.target_ind, df.pred_1_prob)
                    axes.plot(recall, precision, label=clf)
                except:
                    pass
            plt.title('Feature Set: ' + fset + '\nPrecision Recall Chart')
            plt.grid(True)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
            leg = plt.gca().get_legend()
            leg.set_title('Classification Model')
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize='small')
            if save:
                plt.savefig(graph_files_dir + 'pr chart - ' + fset + '.png')

    if grouping == 'clf':
        for clf in classifiers:
            fig, axes = plt.subplots(1, 1, figsize=(10, 7))
            for fset in feature_sets:
                try:
                    df = df_result[(df_result.model_name == clf) & (df_result.fset_name == fset)]
                    precision, recall, thresholds = precision_recall_curve(df.target_ind, df.pred_1_prob)
                    axes.plot(recall, precision, label=fset)
                except:
                    pass
            plt.title('Classification Model: ' + clf + '\nPrecision Recall Chart')
            plt.grid(True)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
            leg = plt.gca().get_legend()
            leg.set_title('Feature Set')
            ltext = leg.get_texts()
            plt.setp(ltext, fontsize='small')
            if save:
                plt.savefig(graph_files_dir + 'pr chart - ' + clf + '.png')


def get_mrp_results(df_results):
    mrp_results = []
    for mrp in range(25, 80, 5):
        df = df_results.reset_index().groupby(['summary_nr', 'fset_name', 'target_ind'], as_index=False) \
            ['pred_0_prob', 'pred_1_prob'].mean()
        df['model_name'] = 'MRP ' + str(mrp)
        df['pair_name'] = df.fset_name + ' - ' + df['model_name']
        df['target_pred'] = df['pred_1_prob'].map(lambda x: 1 if x * 100 >= mrp else 0)
        df = df[['summary_nr', 'fset_name', 'model_name', 'pair_name', 'pred_0_prob', 'pred_1_prob', 'target_ind',
                 'target_pred']]
        mrp_results.append(df)
    df_mrp = pd.concat(mrp_results)
    return df_mrp.set_index('summary_nr')


def get_voting_results(df_results, num_models):
    voting_results = []
    for vote in range(1, num_models + 1):
        df = df_results.reset_index().groupby(['summary_nr', 'fset_name', 'target_ind'], as_index=False) \
            ['pred_0_prob', 'pred_1_prob', 'target_pred'].sum()
        df['model_name'] = 'Vote ' + str(vote) + '+'
        df['pair_name'] = df.fset_name + ' - ' + df['model_name']
        df['target_pred'] = df['target_pred'].map(lambda x: 1 if x >= vote else 0)
        df['pred_0_prob'] = df['pred_0_prob'] / num_models
        df['pred_1_prob'] = df['pred_1_prob'] / num_models
        df = df[['summary_nr', 'fset_name', 'model_name', 'pair_name', 'pred_0_prob', 'pred_1_prob', 'target_ind',
                 'target_pred']]
        voting_results.append(df)
    df_vote = pd.concat(voting_results)
    return df_vote.set_index('summary_nr')


def plot_combination_model_chart(df_mrp_metrics, df_voting_metrics, feature_sets,\
                                 num_models, graph_files_dir, save=False):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for fset in feature_sets:
        dfm = df_mrp_metrics[df_mrp_metrics.fset_name == fset]
        dfm['xi'] = dfm.model_name.map(lambda x: int(x[-2:]))
        axes[0].plot(dfm.xi, dfm.accuracy_score, label=fset, marker='o')
        axes[0].set_xticks(range(25, 80, 5))
        axes[0].set_xlabel('MRP Threshold')
        axes[0].set_ylabel('Accuracy Score')
        axes[0].set_title('Mean Response Probababity Model Results')
        axes[0].set_xlim([25, 75])
        axes[0].grid(True)
        axes[0].legend(loc='lower center', prop={'size': 10}).set_title('Feature Set')
        dfv = df_voting_metrics[df_voting_metrics.fset_name == fset]
        dfv['xi'] = dfv.model_name.map(lambda x: int(x[-2:-1]))
        axes[1].plot(dfv.xi, dfv.accuracy_score, label=fset, marker='o')
        axes[1].set_xticks(range(0, 8, 1))
        axes[1].set_xlabel('Voting Model [ >= X Agree ]')
        axes[1].set_title('Voting Model Results')
        axes[1].set_xlim([1, num_models])
        axes[1].grid(True)
        fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
        if save:
            plt.savefig(graph_files_dir + 'mrp voting results chart - ' + fset + '.png')


def plot_outcome_comparison(df_outcome, graph_files_dir, figsize=(10, 4), save=False):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=.8)
    fig.suptitle('Outcome Comparison - Best "Structured" Model Versus All Others', fontsize=12)
    ax.set_frame_on(False)
    ax.set_xlabel('Accidents Ordered by the Best "Structured" Model\n\
                   Probability of Fatal Accident (High to Low)', size=12)
    ax.set_yticks(np.arange(df_outcome.shape[0]) + 0.5, minor=False)
    ax.set_xlim([0, len(df_outcome.T)])
    ax.set_xticks(np.arange(0, len(df_outcome), 5000))
    ax.set_yticklabels(df_outcome.columns, minor=False)
    cmap = mpl.colors.ListedColormap(['floralwhite', 'red', 'blue', 'lightgrey'])
    bounds = [-2.5, -1.5, 0, 1.5, 2.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.pcolor(df_outcome.T, cmap=cmap, norm=norm)
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    fig.text(0.16, 0.85, 'True Positive', fontsize=12, bbox={'facecolor': 'floralwhite', 'pad': 10})
    fig.text(0.35, 0.85, 'False Positive', fontsize=12, bbox={'facecolor': 'blue', 'pad': 10, 'alpha': 0.5})
    fig.text(0.54, 0.85, 'False Negative', fontsize=12, bbox={'facecolor': 'red', 'pad': 10})
    fig.text(0.75, 0.85, 'True Negative', fontsize=12, bbox={'facecolor': 'lightgrey', 'pad': 10})
    if save:
        plt.savefig(graph_files_dir + 'pcolor_outcome_compare.png')


def plot_best_FN(df_struct_FN, graph_files_dir, save=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(top=0.88)
    fig.suptitle('Best "Structured" Model False Negatives (' + str(len(df_struct_FN)) +
                 ')\nIncorrect Classification of Accident as Non-Fatal', size=12)
    ax.set_frame_on(False)
    ax.set_xlabel('Accidents Ordered by the Best "Structured" Model\n\
                   Probability of Fatal Accident (High to Low)', size=12)
    ax.set_yticks(np.arange(df_struct_FN.shape[0]) + 0.5, minor=False)
    ax.set_xlim([0, len(df_struct_FN)])
    ax.set_xticks(np.arange(0, len(df_struct_FN), 500))
    ax.set_yticklabels(df_struct_FN.columns, minor=False)
    im = ax.pcolor(df_struct_FN.T, cmap=plt.cm.RdYlGn)
    cbar = fig.colorbar(im)
    cbar.set_label('Probability of Fatal Accident', rotation=90, size=12)
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    if save:
        plt.savefig(graph_files_dir + 'pcolor_FN.png')


def plot_best_FP(df_struct_FP, graph_files_dir, save=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(top=0.88)
    fig.suptitle('Best "Structured" Model False Positives (' + str(len(df_struct_FP)) +
                 ')\nIncorrect Classification of Accident as Fatal', size=12)
    ax.set_frame_on(False)
    ax.set_xlabel('Accidents Ordered by the Best "Structured" Model\n\
                   Probability of Non-Fatal Accident (High to Low)', size=12)
    ax.set_yticks(np.arange(df_struct_FP.shape[0]) + 0.5, minor=False)
    ax.set_xlim([0, len(df_struct_FP)])
    ax.set_xticks(np.arange(0, len(df_struct_FP), 500))
    ax.set_yticklabels(df_struct_FP.columns, minor=False)
    im = ax.pcolor(df_struct_FP.T, cmap=plt.cm.RdYlGn)
    cbar = fig.colorbar(im)
    cbar.set_label('Probability of Non-Fatal Accident', rotation=90, size=12)
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    if save:
        plt.savefig(graph_files_dir + 'pcolor_FP.png')


def get_model_training_info(batch_dir):
    df_results = pickle.load(open(batch_dir + 'df_results.p', 'rb'))
    df_metrics = pickle.load(open(batch_dir + 'df_metrics.p', 'rb'))
    col_dict = pickle.load(open(batch_dir + 'feature_set_columns.p', 'rb'))
    fsets = list(df_results.fset_name.unique())
    clfs = list(df_results.model_name.unique())
    model_dict = {}
    for clf in clfs:
        for fset in fsets:
            pair_name = fset + ' - ' + clf
            try:
                model_dict[pair_name] = pickle.load(open(batch_dir + pair_name + '.p', 'rb'))
            except:
                print pair_name, ' model does not exist...'
    return df_results, df_metrics, col_dict, model_dict, fsets, clfs


def get_combination_metrics(df_combination_results):
    combination_metrics = []
    for group, data in df_combination_results.groupby(['pair_name', 'fset_name', 'model_name']):
        y_test = data.target_ind
        test_pred = data.target_pred
        cm = confusion_matrix(y_test, test_pred)
        combination_metrics.append([group[0], group[1], group[2], 0,
                                    round(np.mean(y_test) * 100, 1), accuracy_score(y_test, test_pred),
                                    precision_score(y_test, test_pred), recall_score(y_test, test_pred),
                                    f1_score(y_test, test_pred), cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]])
        columns = ['pair_name', 'fset_name', 'model_name', 'duration', 'act_target_pct', 'accuracy_score',
                   'precision_score', 'recall_score', 'f1_score', 'TN', 'FN', 'FP', 'TP']
    df_combination_metrics = pd.DataFrame(combination_metrics, columns=columns)
    return df_combination_metrics


def get_best_models(group, top_n, score_fn):
    return group.sort_index(by=[score_fn], ascending=False)[:top_n]


def get_model_type(model_name):
    if 'mrp' in model_name.lower():
        return 'mrp'
    elif 'vote' in model_name.lower():
        return 'vote'
    else:
        return 'base'

def plot_average_sentiment(df_ling, partition):
    plt.figure()
    dfs = df_ling[df_ling.partition==partition]
    plt.scatter(dfs.avg_neg_val[dfs.fatality_ind == 0], \
                dfs.avg_pos_val[dfs.fatality_ind == 0], c='g', label='Not Fatal', alpha=0.30)
    plt.scatter(dfs.avg_neg_val[dfs.fatality_ind == 1], \
                dfs.avg_pos_val[dfs.fatality_ind == 1], c='r', label='Fatal', alpha = 0.30)
    plt.xlim([.02,.11])
    plt.ylim([.02,.11])
    plt.legend()
    plt.title('Avg Negative Sentiment VS Avg Positive Sentiment\n' + partition.upper() + ' Set')

def plot_pca(df_ling, partition):
    plt.figure()
    dfs = df_ling[df_ling.partition==partition]
    plt.scatter(dfs.PCA_Comp_1[dfs.fatality_ind==0], dfs.PCA_Comp_2[dfs.fatality_ind==0], c='g', label='Not Fatal',alpha=0.1)
    plt.scatter(dfs.PCA_Comp_1[dfs.fatality_ind==1], dfs.PCA_Comp_2[dfs.fatality_ind==1], c='r', label='Fatal',alpha=0.1)
    plt.xlabel('Principal Component 1')
    plt.ylim([-5,10])
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA - ' + partition.upper() + ' Set')

def plot_scatter_var1_vs_var2(df, var1, var2, partition, title):
    plt.figure()
    dfs = df[df.partition==partition]
    plt.scatter(dfs[var1][dfs.fatality_ind == 0], \
                dfs[var2][dfs.fatality_ind == 0], c='g', label='Not Fatal', alpha=0.30)
    plt.scatter(dfs[var1][dfs.fatality_ind == 1], \
                dfs[var2][dfs.fatality_ind == 1], c='r', label='Fatal', alpha = 0.30)
    plt.legend()
    plt.title(title+ '\n' + partition.upper() + ' Set')