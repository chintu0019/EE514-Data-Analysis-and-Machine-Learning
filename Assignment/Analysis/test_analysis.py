import os

for filename in os.listdir('../dataset/preprocessed/enron1/spam/'):
    #takes out newlines in the file, each line is put into the list 'lines'
    # lines = [line.rstrip('\n') for line in open('../dataset/preprocessed/enron1/spam/'+filename)]
    
    file_path = '../dataset/preprocessed/enron1/spam/'+filename
    for line in open(file_path, 'r', encoding="utf8", errors='ignore'):
        #print(line)
        lines = line.rstrip('\n')
    
print(lines)