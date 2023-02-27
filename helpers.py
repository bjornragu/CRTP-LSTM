import numpy as np

def get_trace_helper(data):
    trace_helper = {}

    trace_helper['vocabulary'] = list(set([act for trace in data.trace for act in trace.split(', ')]))
    trace_helper['vocabulary'].sort()
    trace_helper['vocab_size'] = len(trace_helper['vocabulary']) + 1 
    trace_helper['integer_to_activity'] = { (i + 1): trace_helper['vocabulary'][i] for i in range(len(trace_helper['vocabulary'])) }
    trace_helper['activity_to_integer'] = { trace_helper['integer_to_activity'][i]: i for i in trace_helper['integer_to_activity'].keys() }
    
    return trace_helper


def time_helper(data, col):
    values = list(set([values for value in data[col] for values in value.split(', ')]))
    values.sort(key = int)
    size = len(values) + 1
    
    integer_to_val = { (i + 1): values[i] for i in range(len(values)) }
    value_to_integer = { integer_to_val[i]: i for i in integer_to_val.keys() } 
    
    return values, size, integer_to_val, value_to_integer

def get_time_helpers(data, time_feat):
    
    time_helpers = {}

    for col in [col for col in time_feat if ('std' not in col)]:

        values, size, integer_to, to_integer = time_helper(data, col)
        time_helpers[col] = {}
        time_helpers[col]['values'] = values
        time_helpers[col]['size'] = size
        time_helpers[col]['integer_to_{}'.format(col)] = integer_to
        time_helpers[col]['{}_to_integer'.format(col)] = to_integer
        
    return time_helpers

def cat_helper(data, col):
    values = list(set([val for seq in data[col] for val in seq.split(', ')]))
    if (type(values[1]) == int) | (type(values[1]) == float):
        values.sort(key=type(values[1]))
    else:
        values = sorted(values)
        
    size = len(values) + 1
    
    integer_to_val = { (i + 1): values[i] for i in range(len(values)) }
    value_to_integer = { integer_to_val[i]: i for i in integer_to_val.keys() } 
    
    return values, size, integer_to_val, value_to_integer

def get_cat_helpers(data, cat_feat):
    
    cat_helpers = {}

    for col in cat_feat:

        values, size, integer_to, to_integer = cat_helper(data, col)
        cat_helpers[col] = {}
        cat_helpers[col]['values'] = values
        cat_helpers[col]['size'] = size
        cat_helpers[col]['integer_to_{}'.format(col)] = integer_to
        cat_helpers[col]['{}_to_integer'.format(col)] = to_integer
        
    return cat_helpers


def get_helpers(data, feat_dic):
    
    helpers = {}
    
    helpers['trace_helper'] = get_trace_helper(data)
    helpers['time_helpers'] = get_time_helpers(data, feat_dic['time_feat'])
    helpers['cat_helpers'] = get_cat_helpers(data, feat_dic['cat_feat'])
    
    return helpers


