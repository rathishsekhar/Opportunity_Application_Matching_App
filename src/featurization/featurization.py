# src/featurization/featurization.py

import numpy as np 
import pandas as pd 
import gensim

import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

def w2vbased_embedder(data, uid_column_name, str_column, bool_column, float_column, ApplicationId_w2v_dict, ApplicationId_word2weight_dict):
    """
     This function embeds TF-IDF weighted Word2Vec for string columns and encodes boolean and float columns,
    using the components provided in `ApplicationId_w2v_dict` and `ApplicationId_word2weight_dict`.
    
    The function assumes that the components for the test data are already generated, and it uses those instead of creating new ones.
    The nested function `tfidf_weighted_word2vec` has been removed as it's not needed for test data generation.

    In this modified version:
    1. It uses precomputed Word2Vec components from the training data.
    2. It does not save any components with the test data, as it's redundant.
    3. It accepts `ApplicationId_w2v_dict` and `ApplicationId_word2weight_dict` as inputs to ensure that only the application-specific components are included, excluding candidate components.


    Overall, this function embeds TF-IDF weighted Word2Vec for string columns and encodes 
    boolean and float columns, padding them as needed, before concatenating the results into 
    horizontally and vertically stacked vectors.
    Args:
        data (pandas.DataFrame): Input dataset.
        uid_column_name (str): Name of the user ID column.
        str_column (list): List of string column names.
        bool_column (list): List of boolean column names.
        float_column (list): List of float column names.
        vector_dim (int): Dimension of the word vectors.
        ApplicationId_w2v_dict (dict): Dictionary containing Word2Vec vectors for each string column.
        ApplicationId_word2weight_dict (dict): Dictionary containing TF-IDF weights for each string column.


    Returns:
        dict_hstack (dict): Dictionary with user ID as keys and hstacked 
        vectors as values.
        dict_vstack (dict): Dictionary with user ID as keys and vstacked
          vectors as values.

    """
    def tfidfw2v_vectorizer(text, w2v, word2weight, vector_dim = 768):
        """

        Perform TF-IDF weighted Word2Vec embdedding on a tet column in a DataFrame
        using the word2vec related components provided on the text. 

        Function calculates the TFIDF (from scikit-learn's TFIDFfVectorizer) 
        weighted word2vec (from gensim.Word2Vec) as per the following formulae:
        Tfidf w2v (w1,w2..) = 
        (tfidf(w1) * w2v(w1) + tfidf(w2) * w2v(w2) + …)/(tfidf(w1) + tfidf(w2) + …
        from various inputs. 

        Args:
            text (str): Input text for which to calculate the TF-IDF weighted 
            Word2Vec vector.
            w2v (dict): Dictionary with keys as words and values as their 
            respective vectors.
            word2weight (dict): Dictionary with words and their corresponding
            TF-IDF  weights.

        Returns:
            np.ndarray: TF-IDF weighted Word2Vec vector for the input text.

        """
        words = text.split() 

        if len(words) == 0:
            
            return np.zeros(vector_dim) 

        else:
            numerator_vector = np.zeros(vector_dim)
            denominator_value = 0.0
            
            for word in words:
                
                if word in w2v.keys() and word in word2weight.keys():
                    
                    numerator_val = words.count(word)*word2weight[word]*w2v[word]
                    numerator_vector += numerator_val
                
                    denominator_val = words.count(word)*word2weight[word]
                    denominator_value += denominator_val
            
            if denominator_value == 0.0:
                
                return np.zeros(vector_dim)
        
            else: 
                
                return np.round(numerator_vector/denominator_value, 3)
    
    # Defining functions that encode and pad boolean and float values
    def encode_and_pad_boolean_columns(fdata, bool_column, vector_dim = 768):
        """
        Encode bookean columns in a pandas DataFrame using OneHot Encoder
        
        Args:
            fdata (pandas DataFrame): upon whose boolean columns the encoding is to 
            executed

            bool_column (list): List containing the boolean columns names to be 
            encoded

            vector_dim (int): Dimension of the w2v_vectors
        
        Returns:
            None, modifies the DataFrame in place adding new columns with one hot 
            encoded data
        
        """
        onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        
        for colname in bool_column:
            fdata[colname + "__w2v"] = [
                np.pad(x,  
                    (0, vector_dim - (len(x) % vector_dim)), 
                    'constant') for x in onehotencoder.fit_transform(
                        np.reshape(np.array(fdata[colname]), (-1, 1))
                        )
                        ]

    def pad_float_columns(fdata, float_column, vector_dim = 768):
        """
        Pads the specified float columns in the fdata pandas DataFrame so that the
        final value has a length equal to vector_dim

        Args:
            fdata (pandas DataFrame): Data frame containing the float value
            float_column (list): List of column names containig the float data
            vector_dim (int): Dimension of the vector the columns will be padded

        Returns:
            None: Converts/ modifies the data and generates the new columns
        """
        for colname in float_column:
            fdata[colname + "__w2v"] = [np.pad(
                x, 
                (0, vector_dim - (len(x) % vector_dim)), 
                'constant'
            ) for x in (np.reshape(
                np.array(fdata[colname]), (-1, 1)
            ))]
            
    def hstacker(row_arrays):
        """
        Function that concatenates each of the column data for each row
        """
        return np.concatenate(row_arrays)

    def vstacker(row_arrays):
        """
        Gives the mean vector for the vectors in columns row-wise
        """
        return np.mean(row_arrays)
    
    # Gathering the data and dropping duplicates
    data__ = data[[uid_column_name] + [x + "__w2vpp" for x in str_column] + bool_column + float_column]

    # Applying encode_pad_boolean_columns and pad_float_columns 

    encode_and_pad_boolean_columns(data__, bool_column)
    pad_float_columns(data__, float_column)

     # Gathering and applying BERT base embedded vector for opportunity columns
    
    dict_hstack = {}
    dict_vstack = {}

    # Gathering string data only along with uid_column_name

    for colname in  str_column:
        w2v = ApplicationId_w2v_dict[colname + "__w2vpp"]
        word2weight = ApplicationId_word2weight_dict[colname + "__w2vpp"]
        data__[colname + "__w2v"] = data__[colname + "__w2vpp"].apply(lambda x: tfidfw2v_vectorizer(x, w2v, word2weight))
    
    data__[uid_column_name + "__w2v_hstack"] = data__[[m + "__w2v" for m in str_column + bool_column + float_column]].apply(hstacker, axis = 1)
    data__[uid_column_name + "__w2v_vstack"] = data__[[m + "__w2v" for m in str_column + bool_column + float_column]].apply(vstacker, axis = 1)
    
    for index, row in data__.iterrows():
        dict_hstack[data__.at[index, uid_column_name]] = data__.at[index, uid_column_name + "__w2v_hstack"]
        dict_vstack[data__.at[index, uid_column_name]] = data__.at[index, uid_column_name + "__w2v_vstack"]
    
    return dict_hstack, dict_vstack