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
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='disaster_messages', con=engine)
    
    category_names = set(df.columns.tolist()) - set(['id','message','original','genre'])

    X = df['message'].values
    Y = df[category_names].values
    
    return X, Y, category_names 


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-2)))
    ])

    #parameters selected after grid search
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
    preds = model.predict(X_test)

    for idx, category in enumerate(category_names):
        print(f'results for {category}: \n', 
            classification_report(Y_test[idx], preds[idx]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
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