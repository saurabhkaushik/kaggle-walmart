"""

Walmart 2015 @ Kaggle
@author:Abhishek


"""

import pandas as pd
import numpy as np


data = pd.read_csv('../data/full_data_enc.csv', usecols=['VisitNumber', 'ScanCount', 'company'])
visit = np.unique(data.VisitNumber.values)
with open('../features/fulldata_company_bought.libsvm', 'w') as f:
    for k, v in enumerate(visit):
        trip = data[data['VisitNumber'] == v]
        trip = trip.drop('VisitNumber', axis=1)
        if trip.empty:
            f.write('{}\n'.format(str(-1)))
            continue
        pos_trip = trip[trip['ScanCount'] > 0]
        if pos_trip.empty:
            f.write('{}\n'.format(str(-1)))
            continue
        pos_trip = pos_trip.sort_values(by='company')
        p = np.array(pos_trip.groupby(
            'company').agg(sum).reset_index())
        f.write('{}\n'.format(
            ' '.join([str(-1)] + [str(i[0] + 1) + ':' + str(i[1]) for i in p])))
        if k % 1000 == 0:
            print k

data = pd.read_csv('../data/full_data_enc.csv')
visit = np.unique(data.VisitNumber.values)
with open('../features/fulldata_company_count.libsvm', 'w') as f:
    for k, v in enumerate(visit):
        trip = data[data['VisitNumber'] == v]
        trip = trip.drop('VisitNumber', axis=1)
        if trip.empty:
            f.write('{}\n'.format(str(-1)))
            continue
        trip = trip.sort_values(by='company')
        p = np.array(trip.groupby(
            'company').agg('count').reset_index())
        f.write('{}\n'.format(
            ' '.join([str(-1)] + [str(i[0] + 1) + ':' + str(i[1]) for i in p])))
        if k % 1000 == 0:
            print k

data = pd.read_csv('../data/full_data_enc.csv', usecols=['VisitNumber', 'ScanCount', 'company'])
visit = np.unique(data.VisitNumber.values)
with open('../features/fulldata_company_return.libsvm', 'w') as f:
    for k, v in enumerate(visit):
        trip = data[data['VisitNumber'] == v]
        trip = trip.drop('VisitNumber', axis=1)
        if trip.empty:
            f.write('{}\n'.format(str(-1)))
            continue
        neg_trip = trip[trip['ScanCount'] < 0]
        if neg_trip.empty:
            f.write('{}\n'.format(str(-1)))
            continue
        neg_trip = neg_trip.sort_values(by='company')
        p = np.array(neg_trip.groupby(
            'company').agg(sum).reset_index())
        f.write('{}\n'.format(
            ' '.join([str(-1)] + [str(i[0] + 1) + ':' + str(i[1]) for i in p])))
        if k % 1000 == 0:
            print k

data = pd.read_csv('../data/fulldata.csv')
visit = np.unique(data.VisitNumber.values)
with open('../features/fulldata_company_per_trip.libsvm', 'w') as f:
    for i, v in enumerate(visit):
        df = data[data['VisitNumber'] == v]
        if i %100 == 0: print i
        x = ','.join(list(df.apply(lambda x:'%s' % (x['company']),axis=1).ravel()))
        f.write('{}\n'.format(x))
