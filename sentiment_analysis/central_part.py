from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas
import re
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle


class pridictor():
    modelname = []
    modelscore = []



    def __init__(self,):
        feat = []
        # Loading the data
        data = pandas.read_csv('data_airline.csv')
        print('data is loaded and below are starting five data rows')
        print(data.head())
        # As we can see first column is of no use so lets delete it
        data = data.drop(data.columns[0], axis=1)
        # now lets convert the sentiments values in boolean values
        target_var = data['airline_sentiment'].map({'positive': True, 'negative': False})
        print('target variable is loaded and below are starting five data rows')
        print(target_var.head())
        print(target_var.describe())
        # saving the learning data in different variable
        learning_data = data.iloc[:, 1]
        print('learning data is loaded and below are starting five data rows')
        print(learning_data.head())
        length = len(learning_data)
        for i in range(length):
            # Remove all the special characters
            x = re.sub(r'\W', ' ', str(learning_data[i]))
            # remove all single characters
            x = re.sub(r'\s+[a-zA-Z]\s+', ' ', x)
            # Remove single characters from the start
            x = re.sub(r'\^[a-zA-Z]\s+', ' ', x)
            # Converting to Lowercase
            x = x.lower()
            feat.append(x)
        print(feat)
        # spliting data in trainning and testing data
        x_train, x_test, y_train, y_test = train_test_split(feat, target_var, test_size=0.2, random_state=1000)
        # splited data in trainning and testing data
        print(x_train, x_test)
        print(y_train, y_test)
        # changing text sentences to vector form using sklearn vectorizer
        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        # saving vectorizer
        pickle.dump(vectorizer, open('vectoring.pkl', 'wb'))
        self.trainee=x_train
        # fitting vector form in x_train and x_test
        self.x_train = vectorizer.transform(x_train)
        self.x_test = vectorizer.transform(x_test)
        self.y_train= y_train
        self.y_test=y_test

    # Logistic regression model
    def log_reg(self):
        self.l = LogisticRegression()
        self.l.fit(self.x_train, self.y_train)
        print(self.l.score(self.x_test, self.y_test))
        y_pred = self.l.predict(self.x_test)
        self.modelname.append('Logistic Regression')
        self.modelscore.append(metrics.accuracy_score(self.y_test, y_pred))

    # decision tree model
    def decision_tree(self):
        self.clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        self.clf_gini.fit(self.x_train, self.y_train)
        self.modelname.append('Decision tree')
        self.modelscore.append(self.clf_gini.score(self.x_test, self.y_test))

    # naive bayes model
    def naive_bayes_mod(self):
        self.gnb = GaussianNB()
        self.gnb.fit(self.x_train.todense(), self.y_train)
        self.modelname.append('Naive bayes')
        self.modelscore.append(self.gnb.score(self.x_test.todense(), self.y_test))

    # Random forest model
    def random_forest_(self):
        self.rf = RandomForestClassifier(n_estimators=200, random_state=0)
        self.rf.fit(self.x_train, self.y_train)
        self.modelname.append('random forest')
        self.modelscore.append(self.rf.score(self.x_test, self.y_test))

    # table for each model with score and select the specific model and save it in pickle file
    def results(self):
        self.log_reg()
        self.decision_tree()
        self.random_forest_()
        self.naive_bayes_mod()
        Final_results = pandas.DataFrame(self.modelname, self.modelscore)
        print(Final_results)
        max_index=self.modelscore.index(max(self.modelscore))
        if max_index == 0:
            print("logistic regression is selected")
            pickle.dump(self.l, open('model.pkl', 'wb'))
        elif max_index == 1:
            print("Decision tree is selected")
            pickle.dump(self.clf_gini, open('model.pkl', 'wb'))
        elif max_index == 2:
            print("random forest is selected")
            pickle.dump(self.rf, open('model.pkl', 'wb'))
        elif max_index == 3:
            print("Naive bayes is selected")
            pickle.dump(self.gnb, open('model.pkl', 'wb'))



if __name__ == '__main__':
    p = pridictor()
    p.results()
