import pickle

with open('./log_data.pkl','rb') as f:
    data = pickle.load(f)

first_row = data[0]
print(first_row['build_status'])
for section in first_row['build_log']:
    print(section)