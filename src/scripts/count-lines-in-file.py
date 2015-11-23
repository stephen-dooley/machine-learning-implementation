
# coding: utf-8

# In[ ]:




# In[7]:

inputs = [];

def file_length(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            inputs.append(l);
            pass;
    return i + 1;

number_of_files = file_length('/Users/stephen/Documents/college/machine-learning/assignment-3/data/inputs');
print(inputs);
print(number_of_files);


# In[ ]:



