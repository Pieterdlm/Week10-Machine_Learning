import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv('frequenttabel.csv')

transactions = data.groupby('ORDER_NUMBER')['PRODUCT_NUMBER'].apply(list).values.tolist()

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.024, use_colnames=True)
frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]

frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(str(i) for i in x))
print(frequent_itemsets)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(frequent_itemsets['itemsets'], frequent_itemsets['support'])
ax.set_xlabel('Itemsets')
ax.set_ylabel('Support')
ax.set_title('Frequent Itemsets')
plt.xticks(rotation=90)
plt.show()