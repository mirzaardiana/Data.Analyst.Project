#using pandas library to view dataset
import pandas as pd
iris = pd.read_csv('iris.csv')
print(iris.head())

#Matplotlib
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10,10,30)
y = x**2
z = x**3
plt.plot(x,z,color='red')
plt.plot(x,y,'*')
plt.xlabel('x-values')
plt.ylabel('y-values')
plt.title('line plot')
plt.grid(True)
plt.show()
#line chart y=x^2 and z=x^2

#Plot Mathematicak Function
import numpy as np
plt.plot(np.sin(x),label='sin(x)',color='orange')
plt.plot(np.cos(x),label='sin(x)',color='green')
plt.xlim(10,100)
#plt.xlim(lower_limit,upper_limit)
#plt.xlim(lower_limit,upper_limit)
plt.legend()
plt.title('math functions')
plt.show()
#line chart of sin(x) and cos(x)

#scatter plot
iris.plot(kind='scatter', x='petal-length', y='petal-width')
plt.show()
#irish data scatter plot

#iriss data with different colour
colours = {'Iris-setosa':'orange', 'Iris-versicolor':'lightgreen', 'Iris-virginica':'lightblue'}
for i in range(len(iris['sepal-length'])):
    plt.scatter(iris['petal-length'][i],iris['petal-width'][i], color = colours[iris['species'][i]])
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Iris')
plt.grid(True)
plt.show()
#Iris Dataset Colored Scatter Plot

#Bar Plot : Compare Categorical Data
a = iris['species'].value_counts()
species = a.index
count = a.values
plt.bar(species,count,color = 'lightgreen')
plt.xlabel('species')
plt.ylabel('count')
plt.show()

#Box Plots
length_width=iris[['petal-length','petal-width','sepal-length','sepal-width']]
length_width.boxplot()
plt.xlabel('Flower Measurements')
plt.ylabel('Values')
plt.title('Irish Dataset Analysis')
plt.show()

#Histogram
import numpy as np
data_ = np.random.randn(1000)
plt.hist(data_,bins = 40,color='red')
plt.xlabel('points')
plt.title('Histogram')
plt.grid(True)
plt.show()

#Heat Maps
correlation = iris.corr()
fig ,ax = plt.subplots()
k = ax.imshow(correlation, cmap = 'magma_r')

ax.set_xticks(np.arange(len(correlation.columns)))
ax.set_yticks(np.arange(len(correlation.columns)))
ax.set_xticklabels(correlation.columns)
ax.set_yticklabels(correlation.columns)

cbar = ax.figure.colorbar(k, ax=ax)
cbar.ax.set_ylabel('color bar', rotation=-90, va="bottom")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

for i in range(len(correlation.columns)):
  for j in range(len(correlation.columns)):
    text = ax.text(j, i, np.around(correlation.iloc[i, j], 
                decimals=2),ha="center", va="center", color="lightgreen")
plt.show()

#Pie Chart
a= iris['species'].value_counts()
species = a.index
count = a.values
colors= ['lightblue','lightgreen','gold']
explode = (0,0.2,0)
plt.pie(count, labels=species,shadow=True,
        colors=colors,explode = explode, autopct='%1.1f%%')
plt.xlabel('species')
plt.axis('equal')
plt.show()

#Seaborn LIBARARY Seaborn LIBARARY Seaborn LIBARARY

#Line Plot
import seaborn as sns
sns.set_style('darkgrid')
sns.lineplot(data=iris.drop(['species'], axis=1))
plt.show()

#Scatter Plot
sns.FacetGrid(iris, hue="species", height=4).map(plt.scatter, "sepal-length", "sepal-width").add_legend()
plt.show()

#Bar Plots
a = iris['species'].value_counts()
species = a.index
count = a.values
sns.barplot(x=species,y=count)
plt.xlabel('species')
plt.ylabel('count')
plt.show()

#Histogram
sns.FacetGrid(iris, hue="species", height=5)\
        .map(sns.histplot, "petal-length")\
        .add_legend()
plt.show()

#Heat Maps
sns.heatmap(iris.corr(), annot=True)
plt.show()

#Pair Plot
sns.pairplot(iris, hue="species", height=5);
plt.show()

#PANDAS LIBRATY PANDAS LIBRATY PANDAS LIBRATY

#Histogram
iris['petal-length'].plot.hist()
plt.show()
iris.plot.hist(subplots=True, layout=(2,2), figsize=(8,8))
plt.show()

#Line plot
iris.drop(['species'], axis=1).plot.line()
plt.show()

##4D-Plot (Iris Dataset)
#import plotly.express as px
#df=px.data.iris()
#fig=px.scatter_3d(df,x='sepal_length', y='sepal-width', z='petal-width', color='petal-length', symbol='species')
#fig.show()









