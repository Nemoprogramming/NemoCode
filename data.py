import datasets
import random
import convert
import pandas as pd
def clean_data_defect(dataset="semeru/code-text-javascript"):
    dataset= datasets.load_dataset(dataset)
    consolidated_ds = datasets.DatasetDict({})
    new_ds_train = []
    for ds in dataset['train']:
        label = random.randint(0, 1)
        new_code = ''
        if label == 1:
            new_code = convert.chatbot_program(ds['code'],ds['func_name'])
        else:
            new_code = ds['code']
        newdata_json = {'code':new_code, 'labels':label}
        new_ds_train.append(newdata_json)
    consolidated_ds['train'] = datasets.Dataset.from_pandas(pd.DataFrame(data=new_ds_train))
        
    new_ds_validation = []
    for ds in dataset['validation']:
        label = random.randint(0, 1)
        new_code = ''
        if label == 1:
            new_code = convert.chatbot_program(ds['code'],ds['func_name'])
        else:
            new_code = ds['code']
        newdata_json = {'code':new_code, 'labels':label}
        new_ds_validation.append(newdata_json)
    consolidated_ds['validation'] = datasets.Dataset.from_pandas(pd.DataFrame(data=new_ds_validation))

    new_ds_test = []
    for ds in dataset['test']:
        label = random.randint(0, 1)
        new_code = ''
        if label == 1:
            new_code = convert.chatbot_program(ds['code'],ds['func_name'])
        else:
            new_code = ds['code']
        newdata_json = {'code':new_code, 'labels':label}
        new_ds_test.append(newdata_json)
    consolidated_ds['test'] = datasets.Dataset.from_pandas(pd.DataFrame(data=new_ds_test))
    return consolidated_ds

def clean_data_review(dataset="semeru/code-text-javascript"):
    dataset= datasets.load_dataset(dataset)
    consolidated_ds = datasets.DatasetDict({})

    new_ds_train = []
    for ds in dataset['train']:
        new_code = convert.chatbot_program(ds['code'],ds['func_name'])
        newdata_json = {'code':new_code, 'docstring':ds['docstring']}
        new_ds_train.append(newdata_json)
    consolidated_ds['train'] = datasets.Dataset.from_pandas(pd.DataFrame(data=new_ds_train))

    new_ds_validation = []
    for ds in dataset['validation']:
        newdata_json = {'code':new_code, 'docstring':ds['docstring']}
        new_ds_validation.append(newdata_json)
    consolidated_ds['validation'] = datasets.Dataset.from_pandas(pd.DataFrame(data=new_ds_validation))

    new_ds_test = []
    for ds in dataset['test']:
        newdata_json = {'code':new_code, 'docstring':ds['docstring']}
        new_ds_test.append(newdata_json)
    consolidated_ds['test'] = datasets.Dataset.from_pandas(pd.DataFrame(data=new_ds_test))
    return consolidated_ds
        