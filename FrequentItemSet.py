import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Read data from CSV file
data = pd.read_csv('frequenttabel.csv')

# Convert the data into a list of lists, where each inner list represents the product_numbers for each order_number
transactions = data.groupby('ORDER_NUMBER')['PRODUCT_NUMBER'].apply(list).values.tolist()

# Initialize TransactionEncoder
te = TransactionEncoder()
# Transform the data into one-hot encoded format
te_array = te.fit_transform(transactions)
# Convert the one-hot encoded data into a DataFrame
df = pd.DataFrame(te_array, columns=te.columns_)

# Generate frequent itemsets using Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.024, use_colnames=True)
frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]

# Convert sets of integers in itemsets column to strings
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(str(i) for i in x))
print(frequent_itemsets)

# Visualize frequent itemsets in a bar graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(frequent_itemsets['itemsets'], frequent_itemsets['support'])
ax.set_xlabel('Itemsets')
ax.set_ylabel('Support')
ax.set_title('Frequent Itemsets')
plt.xticks(rotation=90)
plt.show()