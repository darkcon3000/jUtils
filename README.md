# jUtils
A series of functions I use for analyzing data

## plotD
__jUtils.plotD(df, v, color = "blue", scale = None, title = "", xLabel = "", yLabel = "density", width= 600, height = 600)__

plotD uses Altair to create a density plot of one column/variable from a pandas dataframe. 


### Parameters:
* df = Pandas dataframe
* v = column name (string) in the df dataframe that you wish to visualize. Column must have numeric data. 
* color = color (string) for the visualization
* scale = True, False or None. 
  * Setting scale = True will result in an x-axis that ranges from -1 to 1. 
  * Setting scale = False will result in a scale that ranges from 0 to 1. 
  * Setting scale = None will result in Altair automatically setting the scale. 
* title = string of the plot title
* xLabel = string of the label for the x-axis
* yLabel = string of the label for the y-axis
* width = int for the width of the plot
* height = int for the height of the plot
  
### Example:
```
from jUtils import plotD
chart = plotD(df, 'Compound_x', 'red', scale = True, title = 'VADER Compound Score', xLabel='Compound Score')
chart
```
![example1](https://github.com/darkcon3000/jUtils/blob/master/example1.jpg?raw=true)

## multiPlotD
__jUtils.multiPlotD(df,cols,varNames='Variables',color='set1',opacity=0.75,title="",scale=None,width=600,height=600)__
multiPlotD uses Altair to create density plots for multiple variables/columns within the same pandas dataframe. 


### Parameters:
* df = Pandas dataframe
* cols = list of columns/variables to visualize. Column names must be a list of string that match the column name in the df dataframe.
* varNames = string for what the different variables/columns should be collectively called. This is what will appear on the legend. 
* color = string for an Altair/Vega color scheme. https://vega.github.io/vega/docs/schemes/ 
* opacity = float between 0 and 1 that sets the opacity for each of the distributions. 
* title = string for the title of the plot
* scale = a tuple for the scale of the y-axis. If not specified, Altair will automatically set the y-axis. 
* width = int for the width of the plot.
* height = int for the height of the plot. 

### Example:
```
from jUtils import multiPlotD as mpd
chart = mpd(df,cols,varNames='Syuzhet Metrics',title='Syuzhet Post Metrics')
chart
```
![example12](https://github.com/darkcon3000/jUtils/blob/master/example2.jpg?raw=true)

## ttest
__jUtils.ttest(columns,df1,df2,n=100000,pvalue=0.05)__

Returns a dataframe with the variables being tested, the t statistic, the raw p value of each test, and a boolean for whether or not the test passed significance with the Bonferroni Correction applied. 

### Parameters:
* columns = list of shared columns(str) between two dataframes for the ttests to be run on. 
* df1 = 1st pandas dataframe
* df2 = 2nd pandas dataframe
* n = int for the number of permuations for the t-tests to run. Set n = None for no permutation testing. 
* pvalue = float for the p-value of the t-tests to determine significance. 

### Example:

```
from jUtils import ttest
df1 = pd.read_csv('randomData.csv')
df1 = pd.read_csv('exampleData.csv')
columns = ['variable1','variable2','variable3']
resultsDF = ttest(columns,df1,df2)
print(resultsDF)
```


