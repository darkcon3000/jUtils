from scipy import stats
import pandas as pd
import numpy as np
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analyzer import Analyzer
#from bs4 import BeautifulSoup
#from nltk.corpus import stopwords
#import re, os
# import nltk
# from tqdm import tqdm
# import progressbar
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
# __author__ = 'whatknows'

def plotD(df, v, color = "blue", scale = None, title = "", xLabel = "", yLabel = "density", width= 600, height = 600):
    '''Function for plotting density distributions of a single variable in Altair.
    df parameter takes the pandas dataframe for the data you wish to plot.
    v parameter is the column name (string) of the variable you wish to plot. 
    color parameter takes a string of the color of the plot.
    scale parameter takes three options: True, False, or None.
        If True (bool), the x axis scale will range from -1 to 1. 
        If False (bool), the x axis scale will range from 0 to 1.
        If None (none type), the x axis scale will not be specified and Altair will decide for you.
    title parameter takes the title of the plot as a string. 
    xLabel parameter takes a string for the x-axis label. Will default to the variable/column name specified in the v parameter.
    yLabel parameter takes a string for the x-axis label. Will default to "density".
    width parameter takes an integer for the width of the plot.
    height parameter takes an integer for the height of the plot. 
    Returns altair chart object.
    '''
    if len(xLabel) == 0:
        xLabel = v
    if scale:
        Xscale = (-1,1)
        scaled = True
    elif scale == False:
        Xscale = (0,1)
        scaled = True
    else:
        scaled = False
#         pass
    vQ = v+':Q'
    # vT = v+' Score'
    if scaled:
        chart = alt.Chart(df).transform_density(
            v,
            as_=[v, 'density'],
        ).mark_area(line={'color':color},color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0),
                       alt.GradientStop(color=color, offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0)).encode(
            alt.X(vQ,
            scale=alt.Scale(domain=Xscale),
            title=xLabel
            ),
            y=alt.Y('density:Q',
            title=yLabel)
        ).properties(
            title=title,
            width= width,
            height = height
        )

        return chart
    else:
        chart = alt.Chart(df).transform_density(
            v,
            as_=[v, 'density'],
        ).mark_area(line={'color':color},color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='white', offset=0),
                       alt.GradientStop(color=color, offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0)).encode(
            alt.X(vQ, title=xLabel),
            y=alt.Y('density:Q',
            title=yLabel)
        ).properties(
            title=title,
            width= width,
            height = height
        )

        return chart

def multiPlotD(df,cols,varNames='Variables',color='set1',opacity=0.75,title="",scale=None,width=600,height=600):
    '''
    Creates a multivariable density/distribution plot.
    df parameter takes in a pandas dataframe.
    cols parameter takes a list of columns/variables within the aforementioned dataframe that you wish to visualize.
    varNames parameter takes a string that will descibe the legend. IE what the variables should collectively be called.
    opacity parameter takes a float (0.0 - 1.0) to set the transparency of each distribution. 
    color parameter takes an altair/vega color scheme in the form of a string.
    title parameter takes a string for the title of the plot.
    scale parameter sets the y scale in the form of a tuple. If unspecified, altair will automatically set the y-scale.
    width parameter takes an int for the width of the plot.
    height parameter takes an int for the height of the plot.
    Returns an Altair chart object.
    '''
    alt.data_transformers.disable_max_rows()
    vN = varNames+":N"
    if scale is None:
        chart = alt.Chart(df).transform_fold(
            cols,
            as_ = [varNames, 'value']
        ).transform_density(
            density='value',
        #     bandwidth=0.3,
            groupby=[varNames],
            extent= [0, 1],
        #     counts = True,
        #     steps=200
        ).mark_area(opacity=opacity).encode(
            alt.X('value:Q'),
            alt.Y('density:Q'),
            alt.Color(vN,scale=alt.Scale(scheme=color)),
            tooltip = ['value:Q',vN,'density:Q']
        ).properties(title=title,width=600, height=600).interactive()

        return chart
        
    else:
        chart = alt.Chart(df).transform_fold(
            cols,
            as_ = [varNames, 'value']
        ).transform_density(
            density='value',
        #     bandwidth=0.3,
            groupby=[varNames],
            extent= [0, 1],
        #     counts = True,
        #     steps=200
        ).mark_area(opacity=opacity).encode(
            alt.X('value:Q'),
            alt.Y('density:Q',scale=alt.Scale(domain=scale)),
            alt.Color(vN,scale=alt.Scale(scheme=color)),
            tooltip = ['value:Q',vN,'density:Q']
        ).properties(title=title,width=width, height=height).interactive()

        return chart

def sigThres(x,n,p=0.05):
    '''Returns the Bonferroni correction used in the following ttest function'''
    a = p/n
    if x < a:
        return 1
    else:
        return 0
        
def ttest(columns,df1,df2,n=100000,pvalue=0.05):
    '''Performs a t-test on two different dataframes with a list of shared columns.
    columns parameter takes a list of shared columns that the t-tests will be run on between the two pandas dataframes.
    df1 parameter is the first pandas dataframe and the df2 parameter is the second pandas dataframe.
    n parameter is the number of permuations for the t-test to run. Put "None" without quotation marks to run without permutations. 
    pvalue parameter is the p-value.
    This returns a dataframe with each shared varible as the rows and the p-value and test statistic for the values.
    Additionally, the Bonferroni correction is applied and returned in a binary format in the "significant" column with 1 being significant by the Bonferroni standard and 0 being insignificant.
    '''
    rng = np.random.default_rng()
    rdf = pd.DataFrame(columns=['variable','test-statistic','p-value'])
    for i in columns:
    #     result = {}
#         print(i)
        test = stats.ttest_ind(df1[i], df2[i],permutations=n, random_state=rng)
        p = test[1]
        t = test[0]
    #     result['variable']=i
    #     result['p-value'] = p
    #     result['test statistic'] = t
        rdf = rdf.append({'variable':i,'test-statistic':t,'p-value':p},ignore_index=True)
    rdf['significant'] = rdf['p-value'].apply(lambda x: sigThres(x,len(columns),pvalue))
    #     results.append(result)
    return rdf

def classThres(Y,X=None,p=None,model=None,title='Classification Thresholds',width=600,height=600):
    '''
    Returns a plot showing the accuracy, precision, and recall at different classification thresholds.
    Y parameter takes a vector (numpy or pandas) of the labels for the classifier.
    X takes the feature vector (numpy or pandas) for the aforementioned classifier. If X is blank, p must be specified.
    p parameter takes a vector (numpy of pandas) of probability estimates. Use this parameter if the model's predictions are already calculated.
    model parameter takes a trained classifier with the .predict_proba method. Use this parameter if the model's predictions are not yet calculated, will calculate p for you. 
    title parameter is a string of the title for the plot.
    width parameter takes an int for the width of the plot.
    height parameter takes an int for the height of the plot.
    Returns an altair chart object multi-line graph. 

    If the parameter X is passed, the p parameter does not need to be, but the model parameter must be passed.
    Basically, use the X and model parameters or use the p parameter.
    '''

    accuracy = []
    recall = []
    precision = []
    threshold = []
    f1 = []
    roc = []
    if p is None:
        p = model.predict_proba(X)[:,1:]

    for i in range(50,100,5):
        i = i/100
        pred = np.where(p > i, 1, 0)
        accuracy.append(accuracy_score(Y, pred))
        recall.append(recall_score(Y, pred))
        precision.append(precision_score(Y, pred))
        f1.append(f1_score(Y, pred))
        roc.append(roc_auc_score(Y,pred))
        threshold.append(i)

    decision = pd.DataFrame({'Threshold':threshold,'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1 Score':f1,'ROC AUC Score':roc})
    source = pd.melt(decision, id_vars=['Threshold'], value_vars=['Accuracy','Precision','Recall','F1 Score','ROC AUC Score'])

    lines = (
    alt.Chart(source)
    .mark_line(point=alt.OverlayMarkDef())
    .encode(
        x=alt.X("Threshold",scale=alt.Scale(domain=(0.5,1.0))),
        y=alt.Y("value",scale=alt.Scale(domain=(0.0,1.0))),
        color="variable",
        strokeDash = 'variable',
        tooltip = ['variable','Threshold','value']
    )

    ).properties(
        title = title,
        width = width,
        height = height
    )

    return lines

def sentimentAnalysis(df, comment, key):
    '''
    Performs VADER and LIWC sentiment analysis on a corpus. 
    df parameter takes a pandas dataframe.
    comment parameter takes the column name (string) of the text.
    key parameter takes the ID column so the dataframe retuned from this function can be merged back to original dataframe. Index can be passed here.
    Returns a dataframe with the text, ID, and sentiment analysis features.
    '''
    analyzer = Analyzer('LIWC')
    headers = analyzer.headers
    if type(key) == str:
        idParam = True
        keyCol = key
    else:
        keyCol = "index"
        idParam = False
    cols = [keyCol,comment, "Negative", "Neutral", 'Positive','Compound']
    columns = cols+headers
    ndf = pd.DataFrame(columns=columns)
    sid_obj = SentimentIntensityAnalyzer()
    for index, row in df.iterrows():
        if idParam:
            ID = row[keyCol]
        else:
            ID = index
        sentence = row[comment]
        sDict = sid_obj.polarity_scores(sentence)
        metrics = analyzer.analyze(sentence)
        output = {keyCol:ID,comment:sentence,"Negative":sDict['neg'],"Neutral":sDict['neu'],
                          "Positive":sDict['pos'],"Compound":sDict['compound']}
        output.update(metrics)
        ndf = ndf.append(output,ignore_index=True)

    return(ndf)