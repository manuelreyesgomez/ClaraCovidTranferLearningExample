#!/usr/bin/env python3
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from io import StringIO
import json

def transform(scanDirectory, newDirectory,verbose=False):
    imageCount = 0
    newImageSize = (256, 256)
    for entry in os.scandir(scanDirectory):
        if (entry.path.endswith('.jpg') or entry.path.endswith('.jpeg') or entry.path.endswith('.png')) and entry.is_file():
            if (verbose):
                print("processing ", entry.path)
            image = Image.open(entry.path)
            image = image.resize(newImageSize)
            image = image.convert('L')
            newFile = entry.name.replace('.jpg', '.png').replace('.jpeg', '.png')
            newPath = newDirectory + '/' + newFile
            image.save(newPath, 'PNG', icc_profile='')
            imageCount = imageCount + 1    
    print (imageCount, ' images transformed.')
    return

def transformMetadataToJSON(inputFile,outputFile,testingFile,testing=[],validation_size=0.20,countThreshold=2):
    vectorLen = 15
    exclude = ['covid-19-pneumonia-mild.JPG']
    columnsToKeep = ['patientid','finding','view','filename']
    data = pd.read_csv(inputFile)
    data=data[columnsToKeep]
    data = data[(data.view == 'PA') | (data.view == 'AP')]
    data = data[~data.filename.isin(exclude)]
    data.finding = data.finding.apply(lambda x: x.split(',')[0])
    data = data.groupby('finding').filter(lambda x : len(x)>countThreshold)
    labels = list(set(data.finding))
    oneHotEncoding = dict([(label,[0] * vectorLen)  for label in labels])
    for index in range(len(labels)):
        oneHotEncoding[labels[index]][index]=1
        oneHotEncoding[labels[index]]=str(oneHotEncoding[labels[index]])
    data['onehot'] = data.finding.apply(lambda x : oneHotEncoding[x])
    dataForTesting = data[data.filename.isin(testing)]
    data = data[~data.filename.isin(testing)]
    indicesbyLabel = data.groupby('finding').apply(lambda x: x.index.tolist())
    partitionTypes = ['training','validation']
    data['partition'] = partitionTypes[0]
    dataForTesting['partition'] = partitionTypes[1]
    for label in labels:
        data_train, data_test, labels_train, labels_test = train_test_split(indicesbyLabel[label], np.zeros(len(indicesbyLabel[label])), test_size = validation_size)
        data['partition'].loc[data_test]=partitionTypes[1]
    labelstr = '{\n"label_format":' + str([1] * vectorLen) + ',\n'
    createFile(data,partitionTypes,labelstr,True,outputFile)
    createFile(dataForTesting,['validation'],labelstr,False,testingFile)
    labels = ["Pneumocystis","SARS","COVID-19","ARDS","Streptococcus"]
    return labels

def createFile(data,partitionTypes,labelstr,labelflag,filename):
    stream = StringIO()
    stream.write(labelstr)
    for partition in partitionTypes:
        stream.write('"'+partition+ '":[')
        tempdata = data[data.partition==partition]
        for index, row in tempdata.iterrows():
            if labelflag:
                stream.write('{"image":"'+ row['filename'] +'", "label":'+ row['onehot'] +'},\n')
            else:
                stream.write('{"image":"'+ row['filename'] +'"},\n')
        stream.write('],\n')
    stream.write('}')
    strng = str(stream.getvalue())
    strng = strng.replace('.jpeg','.png').replace('.jpg','.png').replace('.JPG','.png').replace('.JPEG','.png').replace(',\n]','\n]').replace('],\n}',']\n}')
    with open(filename, 'w') as out_file:
        out_file.write(strng)
    stream.close()
    return

def adaptFineTuningConfig(inputFile,outputFile,datafile,numepochs,learningrate):
    file = open(inputFile)
    strg = file.read()
    file.close()
    newstrg = strg.replace('DATASET_JSON=$MMAR_ROOT/config/plco.json','DATASET_JSON=' + datafile)
    newstrg = newstrg.replace('epochs=40','epochs={}'.format(numepochs)).replace('learning_rate=0.0002','learning_rate={:f}'.format(learningrate))
    with open(outputFile, 'w') as out_file:
        out_file.write(newstrg)
    return

def adaptScript(inputFile,outputFile,oldValues,newValues):
    file = open(inputFile)
    strg = file.read()
    file.close()
    numChanges = len(oldValues)    
    if numChanges != len(newValues):
        raise("Sorry, the number of old values {} and the number of new values do not match {}",numChanges,len(newValues))
    
    for index in range(numChanges):
        strg = strg.replace(oldValues[index],newValues[index])
    
    with open(outputFile, 'w') as out_file:
        out_file.write(strg)
    return

def adaptJSONTrainConfigFile(inputFile,outputFile,labels,numepochs,learningrate,subtrahend,divisor,image_pipeline):
    offset = 2
    
    with open(inputFile) as f:
        data = json.load(f)
    
    num_labels = len(labels)
    sections = ['train','validate']
    sectionIndexes = {'train':6,'validate':4}
    
    data['epochs'] = numepochs
    data['learning_rate'] = learningrate
    data[sections[0]]['image_pipeline']['args']["sampling"]="automatic"
    
    for section in sections:
        data[section]['image_pipeline']['name']=image_pipeline
    
    for section in sections:
        data[section]['pre_transforms'][sectionIndexes[section]]['args']['subtrahend']=subtrahend
        data[section]['pre_transforms'][sectionIndexes[section]]['args']['divisor']=divisor

    num_reference_labels =  len(data[sections[1]]['metrics']) - offset
    
    if(num_labels>num_reference_labels):
        raise Exception("Sorry, the number of new labels {} can not e larger than the one from the reference model {}",num_labels,num_reference_labels)
    
    del data[sections[1]]['metrics'][offset-1]['args']['is_key_metric']
    
    for index in range(num_labels):
        index_off = offset+index
        data[sections[1]]['metrics'][index_off]['args']['name'] = labels[index]
        
        if (labels[index]=='COVID-19'):
            data[sections[1]]['metrics'][index_off]['args']['is_key_metric'] = True
    
    del data[sections[1]]['metrics'][offset+num_labels:offset+num_reference_labels]
    
    with open(outputFile, 'w') as json_file:
        json.dump(data, json_file, indent = 2)
    
    return

def adaptJSONValidationConfigFile(inputFile,outputFile,labels,numepochs,learningrate,subtrahend,divisor):
    offset = 1
    
    with open(inputFile) as f:
        data = json.load(f)
    
    num_labels = len(labels)
    
    data['pre_transforms'][4]['args']['subtrahend']=subtrahend
    data['pre_transforms'][4]['args']['divisor']=divisor

    num_reference_labels =  len(data['val_metrics']) - offset
    
    if(num_labels>num_reference_labels):
        raise Exception("Sorry, the number of new labels {} can not e larger than the one fro the reference model {}",num_labels,num_reference_labels)
    
    for index in range(num_labels):
        index_off = offset+index
        data['val_metrics'][index_off]['args']['name'] = labels[index]
        
    del data['val_metrics'][offset+num_labels:offset+num_reference_labels]
    
    with open(outputFile, 'w') as json_file:
        json.dump(data, json_file, indent = 2)
    
    return

def adaptJSONFile(inputFile,outputFile,keys,values):

    keyNumber = len(keys)
    if keyNumber != len(values):
        raise Exception("Sorry, the number of keys {} and the number of values do not match {}",keyNumber,len(values))
    
    with open(inputFile) as f:
        data = json.load(f)

    for index in range(keyNumber):
        data[keys[index]] = values[index]

    with open(outputFile, 'w') as json_file:
        json.dump(data, json_file, indent = 2)
    
    return

def processPredictions(filename,labels):
    vectorLen = 15
    num_labels = len(labels)
    todrop = ["P{}".format(x) for x in range(num_labels,vectorLen)]
    names = ['image']
    names.extend(labels)
    names.extend(todrop)
    data = pd.read_csv(filename,names=names)
    data.drop(columns=todrop,inplace=True)
    data[labels]=np.exp(data[labels])
    data[labels] = data[labels].div(data[labels].sum(axis=1), axis=0)
    results = {}
    for index, row in data.iterrows():
        results[row["image"]]= "\n".join(["{} = {}".format(x,y) for x,y in zip(labels,row[labels])])
    return results