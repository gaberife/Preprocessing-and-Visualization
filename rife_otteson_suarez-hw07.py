import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.scorer import make_scorer
from sklearn import model_selection
from scipy.spatial import distance 
import seaborn as sns


# -----------------------------------------------------------------------------
# From hw06
    
class OneNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.inputsDF = None
        self.outputSeries = None
        self.scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)    
    def fit(self, inputsDF, outputSeries):
        self.inputsDF = inputsDF
        self.outputSeries = outputSeries
        return self
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.outputSeries.loc[findNearestHOF(self.inputsDF, testInput)]
        else:
            series = testInput.apply(lambda r: findNearestHOF(self.inputsDF, r),axis = 1)
            newSeries = series.map(lambda r: self.outputSeries.loc[r])
            return newSeries
            
            
def findNearestHOF(df,testRow):
    nearestHOF = df.apply(lambda row: distance.euclidean(row, testRow),axis = 1)
    return nearestHOF.idxmin()  

def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0
    return accuracy

# -----------------------------------------------------------------------------
# Problem 1
    
def standardize(df, listCol):
    df.loc[:, listCol] = (df.loc[:, listCol] - df.loc[:, listCol].mean()) / df.loc[:, listCol].std()
    return df.loc[:, listCol]

# Given
def operationsOnDataFrames():
    d = {'x' : pd.Series([1, 2], index=['a', 'b']),
         'y' : pd.Series([10, 11], index=['a', 'b']),
         'z' : pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')
    
    cols = ['x', 'z']
    
    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')
    
    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')
    
# Given
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    
    # Need to mix this up before doing CV
    # wineDF = wineDF.sample(frac=1)  # "sample" 100% randomly.
    wineDF = wineDF.sample(frac=1, random_state=99).reset_index(drop=True)
    return wineDF, inputCols, outputCol

# Given
def testStandardize():
    df, inputCols, outputCol = readData()
    someCols = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:,someCols].head(5), sep='\n', end='\n\n')
    standardize(df, someCols)
    print("After standardization, first 5 rows:", df.loc[:,someCols].head(5), sep='\n', end='\n\n')
    
    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, someCols].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, someCols].std(), sep='\n', end='\n\n')

# -----------------------------------------------------------------------------
# Problem 2
def normalize(df, listCol):
    df.loc[:, listCol] = (df.loc[:, listCol]-df.loc[:, listCol].min()) / (df.loc[:, listCol].max()-df.loc[:, listCol].min())
    return df.loc[:, listCol]	

# Given
def testNormalize():
    df, inputCols, outputCol = readData()
    someCols = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:,someCols].head(5), sep='\n', end='\n\n')
    normalize(df, someCols)
    print("After normalization, first 5 rows:", df.loc[:,someCols].head(5), sep='\n', end='\n\n')
    
    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, someCols].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, someCols].min(), sep='\n', end='\n\n')
    
# -----------------------------------------------------------------------------
# Problem 3
def comparePreprocessing(k = 10):
    df, inputCols, outputCol = readData()
    alg = OneNNClassifier()    
        
    results = model_selection.cross_val_score(alg, df.loc[:, inputCols], df.loc[:, outputCol], cv=k, scoring=alg.scorer)
    print("Original Values: ", results.mean())
    
    dfCopy = df.copy()
    standardize(dfCopy, inputCols)
    results = model_selection.cross_val_score(alg, dfCopy.loc[:, inputCols], dfCopy.loc[:, outputCol], cv=k, scoring=alg.scorer)
    print("After standardization: ", results.mean())
    
    dfCopy = df.copy()
    normalize(dfCopy, inputCols)
    results = model_selection.cross_val_score(alg, dfCopy.loc[:, inputCols], dfCopy.loc[:, outputCol], cv=k, scoring=alg.scorer)
    print("After normalization: ", results.mean())
      
'''
a)  Original Values:  0.7467169762641899
    After standardization:  0.954499914000688
    After normalization:  0.9548267113863089

    In both standardization and normalization the data is rescaled 
    Standardization is scaled between the mean = 0 and the standard deviation = 1
    Normalization is scaled between min = 0 and max = 1
    This allows all features to contribute equally 
    
b) z-transformed data is linarily transformed with a mean of zero and a stdev 
    of 1, aka Standardization 
    
c) k-fold where youre computing k-1 folds k times. If theyre using 
    leave-one-out then they're not incorporating the entire dataset 
    hence the discrepancy in values 
'''
    
# -----------------------------------------------------------------------------
# Problem 4
'''
a) Negatively skewed
b) Ash: 0 & Magnesium: -1   
c) classification 1
d) Significant difference beasue there is signifacntly less data to be tested 
    against, there are 11 columns of data missing so theres definately going 
    to be some descrepincy when you begin to exclude that dramatically. 
e) still pretty significant for the same reason. 
f) 

Standardized Diluted & Proline:  0.8815079979360165
Standardized Nonflavanoid Phenols & Ash:   0.4915763673890609
'''
def testSubsets(k=10):
    fullDF, inputCols, outputCol = readData()
    alg = OneNNClassifier()    
    inputColsNew = ["Diluted","Proline"]
    standardize(fullDF,inputColsNew)    
    results = model_selection.cross_val_score(alg, fullDF.loc[:,inputColsNew], fullDF.loc[:, outputCol], cv=k, scoring=alg.scorer)
    print("Standardized Diluted & Proline: ", results.mean())

    alg = OneNNClassifier()    
    inputColsNew2 = ["Nonflavanoid Phenols", "Ash"]
    standardize(fullDF,inputColsNew2)      
    results = model_selection.cross_val_score(alg, fullDF.loc[:,inputColsNew2], fullDF.loc[:, outputCol], cv=k, scoring=alg.scorer)
    print("Standardized Nonflavanoid Phenols & Ash: ", results.mean())
   
# --------------------------------------------------------------------------------------
# Given
def test07():
    df, inputCols, outputCol = readData()    
    dfCopy = df.copy()
    standardize(dfCopy, [inputCols[0]])
    print(dfCopy.loc[:, inputCols[0]].head(2))
    
    dfCopy = df.copy()
    normalize(dfCopy, [inputCols[0]])
    print(dfCopy.loc[:, inputCols[0]].head(2))
    testSubsets()
