"""
Utility functions
"""

import csv


###################################################################################################
"""
Write iterable data structure to a csv file
"""

def dump_data_to_csv(name, data):
    print("dump_data_to_csv; name={}, data={}".format(name, data))

    with open('data.csv', 'a') as f:
        f.write("{}, ".format(name))
        writer = csv.writer(f)
        writer.writerow(data)

