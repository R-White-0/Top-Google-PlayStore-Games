# %% [markdown]
# ## Dataset Provided by : Dhruvil Dave - https://www.kaggle.com/dhruvildave/top-play-store-games

# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import seaborn as sns
import nbformat 

# %%
pwd = os.getcwd() 
filepath = pwd + "/android-games.csv"
df = pd.read_csv(filepath)

# %% [markdown]
# ## Viewing the Dataset
#  lets read and check our dataset

# %%
print ('Data shape is: ', df.shape)

# %% [markdown]
# * so we have 1730 rows and 15 columns working with
# * lets take a look at the columns and general summary info.

# %%
pd.DataFrame({"Columns": df.columns})

# %%
df.isnull().sum()

# %% [markdown]
# Great there are no missing vlaues within this dataset. That saves us some time from dealing with NAN's.
# * Lets sample our dataset to explore what it contains.
# * lets sample the first 8 rows.

# %%
df.sample(n=8)

# %%
df.info()

# %% [markdown]
# The install column can be useful for analysis. However note that it's a object type .
# It may be more useful if it were an *integer (int)* or *float*.
# *  May need to reformat this later on.

# %% [markdown]
# # Summary so far: 
# *   The dataset consists of games from different categories, ratings and number of installs.
# *   there are no NAN / null values.
# *   **installs** can be useful for analysis. It would be good to reformat it to a numerical variable.
# * **category** is a categorical variable, making it useful for groupings.
# * numerical varibles should be paid close attentions to.
# * should some columns be dropped ?

# %%
df['price'].value_counts()

# %%
df['paid'].value_counts()

# %% [markdown]
# ## Dropping Price column
# we can drop the price column since most games are free and only 7 aren't, also the price of a game isn't relevant to the EDA.

# %%
df.drop('price', axis=1, inplace=True)

# %%
df.info()

# %% [markdown]
# ## Categories Column

# %%
df['category'].value_counts()

# %%
df['category'].value_counts(ascending=True).plot(kind='barh', title='Game Categories')

# %%
plt.figure(figsize=(25,5))
sns.set_theme(style="darkgrid")
sns.countplot(df['category'])
plt.title('Number of Games by Category')
plt.xlabel('game category')
plt.ylabel('# of games')

# %%
# using the plotly.express library
pio.renderers.default = "notebook" #sets the plotly default render.

fig = px.histogram(df, x="category", title="Game Categories", labels={"category": "Categories"})
fig.update_layout(xaxis={"categoryorder":"total descending"},)
fig.update_yaxes(automargin=True)
fig.show()

# %% [markdown]
# ## Paid vs Free games

# %%
df['paid'].value_counts()


# %%
df['paid'].value_counts().plot(kind='pie', title='Free vs Paid games', legend=True,
    autopct='%1.1f%%', fontsize=14, figsize=[7, 7], labels=["free", "paid"], ylabel='', colors=["g", "r"])


# %%
free_games = df['paid'].value_counts()
label = ['Free', 'Paid']
values = df['paid'].value_counts().values

fig = px.pie(free_games, values=values, names=label, title='Free vs Paid games', color_discrete_sequence=["springgreen", "red"])
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.show()


# %% [markdown]
# ## Relationship Analysis

# %%
df.head(n=5)

# %%
df.describe()

# %% [markdown]
# ## Grouping

# %%
total_ratings_by_cateogry = df.groupby('category')['total ratings'].mean()
total_ratings_by_cateogry 


# %% [markdown]
# N.B. Think of groupby() splits the dataset data into buckets by 'category', and then splitting
#     the records inside  each category bucket by the 'total ratings'.

# %% [markdown]
# ### Total Ratings grouped by Category

# %%
fig = px.bar(total_ratings_by_cateogry, x=total_ratings_by_cateogry.index, y=total_ratings_by_cateogry.values,
             labels={'y': 'Ratings'})
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

# %%
df['installs']

# %% [markdown]
# need to convert this column to a numerical type.

# %%
def numbers(df):
    if df.split(".")[1].split(" ")[1] == "M":
        return int(df.split(".")[0])
    else:
        return int(df.split(".")[0])/1000
df["installs"] = df.installs.apply(numbers)
df = df.rename(columns={'installs': 'installs_in_million'})

# %%
df["installs_in_million"].head()

# %% [markdown]
# it's now a type float.

# %% [markdown]
# ### Installs in Millions grouped by Category

# %%
intalls_in_million_by_category = df.groupby('category')['installs_in_million'].mean()
intalls_in_million_by_category

# %%
fig = px.bar(intalls_in_million_by_category, x=intalls_in_million_by_category.index, y=intalls_in_million_by_category.values,
            width=700, height=400,labels={"y":"Total Installs in Million"})
fig.update_layout(xaxis={"categoryorder":"total descending"}),
margin=dict(l=10, r=10, t=10, b=10)
fig.show()


# %% [markdown]
# ## Growth (30 days) grouped by Category

# %%
df.sample(n=4)

# %%
growth = df.groupby('category')[['growth (30 days)',"growth (60 days)"]].mean()
growth

# %%
fig = px.bar(growth, y="growth (30 days)",
labels={"y":"Growth (30 Days)", "category": "Category", "value":"Total Growth"})
fig.update_layout(xaxis={"categoryorder":"total descending"})

# %%
fig = px.bar(growth,y="growth (60 days)",
labels={"y":"Growth (60 Days)", "category":"Category", "value":"Total Growth"})
fig.update_layout(xaxis={"categoryorder":"total descending"})

# %%
fig = px.line(growth,  y=["growth (30 days)", "growth (60 days)"],title="Games Growth",
labels={"category":"Category", "values":"Total Growth"})
fig.show()


