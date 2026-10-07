import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from (videos.csv and category_id.json) and answer the questions.
"""

def Q1():
    """
        1. How many rows are there in the videos.csv after removing duplications?
        - To access 'videos.csv', use the path '/data/videos.csv'.
    """
    # TODO: Paste your code here
    df = pd.read_csv('/data/videos.csv').drop_duplicates()
    return df['video_id'].count()

def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO: Paste your code here
    return vdo_df['title'].loc[vdo_df['dislikes']>vdo_df['likes']].nunique()

def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    # TODO: Paste your code here
    return vdo_df.query("trending_date == '18.22.01' and comment_count > 10000")['video_id'].count()

def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO:  Paste your code here
    avg_count = vdo_df.groupby('trending_date')['comment_count'].mean()
    return avg_count.idxmin() 

def Q5(vdo_df):
    '''
        5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - You must load the additional data from 'category_id.json' into memory before executing any operations.
            - To access 'category_id.json', use the path '/data/category_id.json'.
    '''
    # TODO:  Paste your code here
    category_id = pd.read_json('/data/category_id.json')
    cat_map = {str(int(item['id'])): item['snippet']['title'] for item in category_id['items']}
    
    vdo_df['cat_str'] = vdo_df['category_id'].apply(str)
    vdo_df['category_title'] = vdo_df['cat_str'].map(cat_map)

    sc_df = vdo_df[vdo_df['category_title'].isin(['Sports', 'Comedy'])]
    
    if sc_df.empty:
        return 0

    daily_views = sc_df.pivot_table(index='trending_date', columns='category_title', values='views', aggfunc='sum', fill_value=0)
    
    sports = daily_views['Sports'] if 'Sports' in daily_views.columns else pd.Series(0, index=daily_views.index)
    comedy = daily_views['Comedy'] if 'Comedy' in daily_views.columns else pd.Series(0, index=daily_views.index)
    
    return (sports > comedy).sum()