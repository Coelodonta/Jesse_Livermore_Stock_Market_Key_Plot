import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_csv("data/SPY_max_1d.csv",index_col="Date", parse_dates=True)
data['Price'] = data.Close

# Just using 1 years worth for testing....
data=data.iloc[-365:]

initial_set = "downtrend"
key_figure = 2

dic_data = {'price': np.nan,
            'state': np.nan,
            'prev_state': np.nan,
            'latest_uptrend': np.nan,
            'latest_downtrend': np.nan,
            'latest_rally': np.nan,
            'latest_reaction': np.nan,
            'latest_second_rally': np.nan,
            'latest_second_reaction': np.nan,
            'redline_uptrend': np.nan,
            'redline_reaction': np.nan,
            'blackline_downtrend': np.nan,
            'blackline_rally': np.nan}


def from_uptrend(df_data, k_figure=key_figure):
    ## to downturn
    if np.isnan(df_data['latest_downtrend']) == False and df_data['price'] < df_data['latest_downtrend']:
        df_data['state'] = "downtrend"
        df_data['latest_downtrend'] = df_data['price']
    ## to reaction or secondary reaction
    elif df_data['price'] < df_data['latest_uptrend'] - k_figure:
        df_data['state'] = "reaction"
        df_data['latest_reaction'] = df_data['price']
    ## continue
    else:
        df_data['state'] = "uptrend"
        if df_data['price'] > df_data['latest_uptrend'] or np.isnan(df_data['latest_uptrend']) == True:
            df_data['latest_uptrend'] = df_data['price']

    return (df_data)


def from_downtrend(df_data, k_figure=key_figure):
    ## to uptrend
    if np.isnan(df_data['latest_uptrend']) == False and df_data['price'] > df_data['latest_uptrend']:
        df_data['state'] = "uptrend"
        df_data['latest_uptrend'] = df_data['price']
    ## to rally or secondary rally
    elif df_data['price'] > df_data['latest_downtrend'] + k_figure:
        df_data['state'] = "rally"
        df_data['latest_rally'] = df_data['price']
    ## continue
    else:
        df_data['state'] = "downtrend"
        if df_data['price'] < df_data['latest_downtrend'] or np.isnan(df_data['latest_downtrend']) == True:
            df_data['latest_downtrend'] = df_data['price']

    return (df_data)


def from_rally(df_data, k_figure=key_figure):
    ## to uptrend
    if np.isnan(df_data['latest_uptrend']) == False and df_data['price'] > df_data['latest_uptrend']:
        df_data['state'] = "uptrend"
        df_data['latest_uptrend'] = df_data['price']
    elif df_data['price'] > df_data['blackline_rally'] + k_figure / 2:
        df_data['state'] = "uptrend"
        df_data['latest_uptrend'] = df_data['price']
    ## to reaction
    elif df_data['price'] < df_data['latest_rally'] - k_figure:
        if np.isnan(df_data['latest_reaction']) == False and df_data['price'] > df_data['latest_reaction']:
            df_data['state'] = "second_reaction"
            df_data['latest_second_reaction'] = df_data['price']
        else:
            df_data['state'] = "reaction"
            df_data['latest_reaction'] = df_data['price']
    ## continue
    else:
        df_data['state'] = "rally"
        if df_data['price'] > df_data['latest_rally'] or np.isnan(df_data['latest_rally']) == True:
            df_data['latest_rally'] = df_data['price']

    return (df_data)


def from_reaction(df_data, k_figure=key_figure):
    ## to downtrend
    if np.isnan(df_data['latest_downtrend']) == False and df_data['price'] < df_data['latest_downtrend']:
        df_data['state'] = "downtrend"
        df_data['latest_downtrend'] = df_data['price']
    elif df_data['price'] < df_data['redline_reaction'] - k_figure / 2:
        df_data['state'] = "downtrend"
        df_data['latest_downtrend'] = df_data['price']
    ## to rally
    elif df_data['price'] > df_data['latest_reaction'] + k_figure:
        if np.isnan(df_data['latest_rally']) == False and df_data['price'] < df_data['latest_rally']:
            df_data['state'] = "second_rally"
            df_data['latest_second_rally'] = df_data['price']
        else:
            df_data['state'] = "rally"
            df_data['latest_rally'] = df_data['price']
    ## continue
    else:
        df_data['state'] = "reaction"
        if df_data['price'] < df_data['latest_reaction'] or np.isnan(df_data['latest_reaction']) == True:
            df_data['latest_reaction'] = df_data['price']

    return (df_data)


def from_second_rally(df_data, k_figure=key_figure):
    ## to rally
    if df_data['price'] > df_data['latest_rally']:
        df_data['state'] = "rally"
        df_data['latest_rally'] = df_data['price']
    ## to reaction
    elif df_data['price'] < df_data['latest_reaction']:
        df_data['state'] = "reaction"
        df_data['latest_reaction'] = df_data['price']
    ## continue
    else:
        df_data['state'] = "second_rally"
        df_data['latest_second_rally'] = df_data['price']
    return(df_data)


def from_second_reaction(df_data, k_figure=key_figure):
    ## to reaction
    if df_data['price'] < df_data['latest_reaction']:
        df_data['state'] = "reaction"
        df_data['latest_reaction'] = df_data['price']
    ## to rally
    elif df_data['price'] > df_data['latest_rally']:
        df_data['state'] = "rally"
        df_data['latest_rally'] = df_data['price']
    ## continue
    else:
        df_data['state'] = "second_reaction"
        df_data['latest_second_reaction'] = df_data['price']
    return (df_data)


def liner(df_data, k_figure=key_figure):
    if df_data['state'] == "reaction" and df_data['prev_state'] != "reaction":
        if np.isnan(df_data['latest_uptrend']) == False:
            df_data['redline_uptrend'] = df_data['latest_uptrend']

    if df_data['state'] == "rally" and df_data['prev_state'] != "rally":
        if np.isnan(df_data['latest_reaction']) == False:
            df_data['redline_reaction'] = df_data['latest_reaction']

    if df_data['state'] == "uptrend" and df_data['prev_state'] != "uptrend":
        if np.isnan(df_data['latest_reaction']) == False:
            df_data['redline_reaction'] = df_data['latest_reaction']

    if df_data['state'] == "rally" and df_data['prev_state'] != "rally":
        if np.isnan(df_data['latest_downtrend']) == False:
            df_data['blackline_downtrend'] = df_data['latest_downtrend']

    if df_data['state'] == "reaction" and df_data['prev_state'] != "reaction":
        if np.isnan(df_data['latest_reaction']) == False:
            df_data['blackline_rally'] = df_data['latest_rally']

    if df_data['state'] == "downtrend" and df_data['prev_state'] != "downtrend":
        if np.isnan(df_data['latest_reaction']) == False:
            df_data['blackline_rally'] = df_data['latest_rally']

    return (df_data)


## Initial Setting
prev_data = dic_data.copy()
prev_data['price'] = data['Price'][0]
prev_data['Date'] = data.index[0] # This is so we can index x-axis on Date
prev_data['prev_state'] = initial_set

df = pd.DataFrame(prev_data, index=[0])

## Iteration
idx=1
for i in data['Price'][1:]:
    cur_data = prev_data.copy()
    cur_data['price'] = i
    cur_data['state'] = np.nan

    cur_data['Date'] = data.index[idx]  # This is so we can index x-axis on Date
    idx+=1

    if cur_data['prev_state'] == "downtrend":
        cur_data = from_downtrend(cur_data)
    elif cur_data['prev_state'] == "uptrend":
        cur_data = from_uptrend(cur_data)
    elif cur_data['prev_state'] == "rally":
        cur_data = from_rally(cur_data)
    elif cur_data['prev_state'] == "reaction":
        cur_data = from_reaction(cur_data)
    elif cur_data['prev_state'] == "second_rally":
        cur_data = from_second_rally(cur_data)
    elif cur_data['prev_state'] == "second_reaction":
        cur_data = from_second_reaction(cur_data)

    cur_data = liner(cur_data)

    prev_data = cur_data.copy()
    prev_data['prev_state'] = prev_data['state']

    df = df.append(cur_data, ignore_index=True)
    df.to_csv("data/df.csv")

## plotting

days = [d for d in df['Date']]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))

plt.plot(days,df['price'],color="black",linestyle='solid',lw=0.7)
plt.plot(days,df['blackline_downtrend'], color="red",linestyle='solid',lw=0.5)
plt.plot(days,df['blackline_rally'], color="red",linestyle='dashed',lw=0.5)
plt.plot(days,df['redline_uptrend'], color="green",linestyle='solid',lw=0.5)
plt.plot(days,df['redline_reaction'], color="green",linestyle='dashed',lw=0.5)

# Add a column for indexing the background color
df['bgcolor'] = range(1,len(df)+1)

background = df[df['state']== 'uptrend']['bgcolor']
for x in background:
    plt.gca().axvline(df.Date.iloc[x-1], color='green',linewidth=10,alpha=0.05)

background = df[df['state']== 'downtrend']['bgcolor']
for x in background:
    plt.gca().axvline(df.Date.iloc[x-1], color='red',linewidth=10,alpha=0.05)

background = df[df['state']== 'rally']['bgcolor']
for x in background:
    plt.gca().axvline(df.Date.iloc[x-1], color='green',linewidth=10,alpha=0.01)

background = df[df['state']== 'reaction']['bgcolor']
for x in background:
    plt.gca().axvline(df.Date.iloc[x-1], color='red',linewidth=10,alpha=0.01)

plt.gcf().autofmt_xdate()
plt.savefig("image/SPY_Market_Key.png")
plt.show()
