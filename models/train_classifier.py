import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import sys
from tempfile import mkdtemp

def load_data(database_filepath):
    """ Load data stored on sqlite database

    Arguments:
    database_filepath (string): Path to sqlite database filename

    Returns:
    X (array): The string messages mx1 to feed the model 
    Y (array): The labels of the model (mx36)
    category_names (list): List of strings with the name of the 36 categories
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='disaster_messages', con=engine)
    
    category_names = set(df.columns.tolist()) - set(['id','message','original','genre'])

    X = df['message'].values
    Y = df[category_names].values
    
    return X, Y, category_names 


def tokenize(text):
    """ Converts a text sentence to a list of tokens using nltk libraries

    Arguments:
    text (string): The sentence to be tokenized

    Returns:
    clean_tokens (list): A list of string tokens after applying nltk libraries
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Build the model structure (Pipeline) to used to train fit
    and predict

    Returns:
    pipeline:  A scikit-learn Pipeline
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-2)))
    ])

    # Best parameters search was done on ML_Pipeline_Preparation.ipynb
    best_params = { 
                    'vect__max_df': 0.5,
                    'vect__ngram_range': (1, 1),
                    'tfidf__use_idf': True,
                    'clf__estimator__min_samples_split': 2,
                    'clf__estimator__n_estimators': 50,
                  }

    pipeline.set_params(**best_params)

    return pipeline 


def evaluate_model(model, X_test, Y_test, category_names):
    """ Make the prediction based on input and outputs model metrics
    for each category

    Arguments:
    model: Model used to evaluate metrics
    X_test: The input features to the model
    Y_test: The true labels
    category_names: The name of categories
    """
    preds = model.predict(X_test)

    for idx, category in enumerate(category_names):
        print(f'results for {category}: \n', 
            classification_report(Y_test[idx], preds[idx]))


def save_model(model, model_filepath):
    """ Save the given model to the especified path """

    joblib.dump(model, model_filepath)


def main():
    """ Run the full training Pipeline
    
    Load training data
    Build the model
    Train the model model
    Evalute model metrics
    Save the trained model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()