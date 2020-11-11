import pandas as pd
#pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

#Opening the file and splitting into list of lists
f = [i.strip('\n').split(' ') for i in open('browsing-data.txt')]
#removing '' from lists
for i in f:
    i.remove('')

te = TransactionEncoder()
te_ary = te.fit(f).transform(f)
df = pd.DataFrame(te_ary, columns=te.columns_)

#print(len(f))
#size of list is 31101

#we want support to be 100, 100/31101 support
support = 100
size = len(f)
support = apriori(df, min_support=(support/size), use_colnames=True, max_len=3)


#single itemset to help calculate confidence
single = support[support['itemsets'].apply(lambda x: len(x) == 1 )]

#two itemset
two = support[support['itemsets'].apply(lambda x: len(x) == 2 )]

#three itemset
three = support[support['itemsets'].apply(lambda x: len(x) == 3 )]

#make it easy to search the single counts
single['itemsets'] = single['itemsets'].astype(str).str.slice(start=-11,stop=-3)

#calculate confidence of each set
two_confidence = []
for i in two.to_records():
    two_confidence.append((i[1]*size)/(((single[single['itemsets'] == list(i[2])[0]]).iloc[0]['support'])*size))
#calculate confidence of each set
three_confidence = []
for i in three.to_records():
    three_confidence.append((i[1]*size)/(((single[single['itemsets'] == list(i[2])[0]]).iloc[0]['support'])*size))


#combining the with confidence
two['confidence'] = two_confidence
three['confidence'] = three_confidence

#resetting the indexes and sorting
two = two.sort_values(by=['confidence'], ascending=False).reset_index(drop=True).drop(['support'],axis=1)
three = three.sort_values(by=['confidence'], ascending=False).reset_index(drop=True).drop(['support'],axis=1)

#generate txt file with output
output = "OUTPUT A\n"
for i in range(0,5):
    output = output + " ".join(map(str,list(two['itemsets'][i]))) + " " +str(round(two['confidence'][i],4)) +"\n"
    
output = output + "OUTPUT B\n"
for i in range(0,5):
    output = output + " ".join(map(str,list(three['itemsets'][i]))) + " " +str(round(three['confidence'][i],4)) +"\n"

#write to file    
file=open("output.txt", "w")
file.write(output)
file.close()
