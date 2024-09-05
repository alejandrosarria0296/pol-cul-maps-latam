#packages
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import ast
import os
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_adapter(x, y):
    """ 
    Calcs the cosine similarity between two vectors
    """
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]


def create_df():
    #set initial variables
    file_path = "pending"

    tagged_data = []

    for df in os.listdir(file_path):
        #sample docs
        name = df.split(".")[0]
        pres_df = pd.read_csv(f"{file_path}/{df}")
        pres_df = pres_df.sample(593) # number of documents in the smallest president corpus we are looking into

        #prep data
        pres_df['normalized_text'] = pres_df['normalized_text'].apply(lambda x: ast.literal_eval(x))

        #create list of lists of tagged docs
        tagged_data += [TaggedDocument(words=doc, tags=[name + '_' + str(i)]) for i, doc in enumerate(pres_df['normalized_text'])]

    #train model
    model = Doc2Vec(vector_size=100, window=6, min_count=2, epochs = 40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    cs_dict = {'President': []}

    for i, vector in enumerate(model.dv.vectors):
        pres = model.dv.index_to_key[i].split('_')[0]
        cs_dict['President'].append(pres)
        row = [cosine_similarity_adapter(vector, vector2) for vector2 in model.dv.vectors]
        cs_dict[model.dv.index_to_key[i]] = row

    df = pd.DataFrame(cs_dict)

    presidents = df['President'].unique()
    ideology = [12.45, 13.27, 6.24, 16.45, 15.7, 2.65, 12.45, 8.35, 4.2, 12.71, 2, 17.33, 10, 12.71, 17.33, 5.4, 16.24, 2.65, 17.87, 11.71, 3.44, 5.23, 5.36,
                8.55, 11.55, 16.85, 16.33, 2, 17.67, 6.36, 6.64, 18.2, 16.88, 15]

    populism = [1, 0, 0.21250001, 0.125, 0.2375, 1.9166666, 0, 0.125, 1.7333333, 0.25, 0.625, 0.46250001, 1, 0, 0.33333334, 0, 0.037500001,
                1.6375, 0.5, 0.375, 1.4625, 0.125, 0.5, 1.25, 0, 0.34999999, 0.15000001, 0.625, 0.0625, 0.4, 0.21250001, 0, 0.25, 0.5]


    p_id = { p: ideology[i] for i, p in enumerate(presidents)}
    p_pol = { p: populism[i] for i, p in enumerate(presidents)}

    #assign populism and ideology values to df
    df['populist'] = df['President'].map(p_pol)
    df['ideology'] = df['President'].map(p_id)

    return df

if __name__ == "__main__":
    df = create_df()
    df.to_csv("presidents.csv", index=False)