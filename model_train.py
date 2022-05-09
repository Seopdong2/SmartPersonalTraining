import pandas as pd
import pickle  # Object serialization.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # 변수들의 스케일 조정 ex) 정규화
from sklearn.linear_model import LogisticRegression, RidgeClassifier # 로지스틱 회귀
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score  # Accuracy metrics


def load_dataset(csv_data):
    df = pd.read_csv(csv_data)

    # print(f'Top5 datas: \n{df.head()}')
    # print(f'Last5 datas: \n{df.tail()}')
    # print(f'Specific class: \n', df[df['class']=='pushup'])  # Show specific class data.

    features = df.drop('class', axis=1)  # Features, drop the colum 1 of 'class' --> 위에 x1 y1 z1 v1 좌표를 삭제하고 학습.
    target_value = df['class']  # target value.

    x_train, x_test, y_train, y_test = train_test_split(features, target_value, test_size=0.3, random_state=1234)

    return x_train, x_test, y_train, y_test


def evaluate_model(fit_models, x_test, y_test):
    print('\nEvaluate model accuracy:')
    # Evaluate and Serialize Model.
    for key_algo, value_pipeline in fit_models.items():
        yhat = value_pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, yhat) * 100
        print(f'Classify algorithm: {key_algo}, Accuracy: {accuracy}%')


if __name__ == '__main__':

    dataset_csv_file = 'C:/Users/liaca/PycharmProjects/pythonProject2/coo2.csv'
    model_weights = 'C:/Users/liaca/PycharmProjects/pythonProject2/weights_body_language.pkl'

    x_train = load_dataset(csv_data=dataset_csv_file)[0]
    y_train = load_dataset(csv_data=dataset_csv_file)[2]
    x_test = load_dataset(csv_data=dataset_csv_file)[1]
    y_test = load_dataset(csv_data=dataset_csv_file)[3]

    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    # print('key:', pipelines.keys())
    # print('value:', list(pipelines.values())[0]) # 0~3

    fit_models = {}
    print('Model is Training ....')
    for key_algo, value_pipeline in pipelines.items():
        model = value_pipeline.fit(x_train, y_train)
        fit_models[key_algo] = model
    print('Training done.')

    # Using x_test data input to Ridge Classifier model to predict.
    rc_predict = fit_models['rc'].predict(x_test)
    print(f'\nPredict 5 datas: {rc_predict[0:100]}')

    # Save model weights.
    with open(model_weights, 'wb') as f:
        # pickle.dump(obj, file, [,protocol=0])
        # obj 개체를 열린 파일로 직렬화함.
        pickle.dump(fit_models['rf'], f)
    print('\nSave model done.')

    evaluate_model(fit_models, x_test, y_test)