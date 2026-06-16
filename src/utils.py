import os
import joblib


def save_object(file_path, obj):

    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    joblib.dump(obj, file_path)


def load_object(file_path):

    return joblib.load(file_path)