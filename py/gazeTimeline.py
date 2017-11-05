
# coding: utf-8

# In[1316]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib.patches as mpatches
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

plt.ioff() # Turn interactive plotting off

#url = "https://raw.githubusercontent.com/jamcro/shootGaze/master/data/shootingNoviceGaze.csv"
#gaze_df = pd.read_csv(url, sep=',', skipinitialspace=True)
#url = "https://raw.githubusercontent.com/jamcro/shootGaze/master/data/shootingNoviceKinematics.csv"
#kin_df = pd.read_csv(url, sep=',', skipinitialspace=True)

# shot_df 0=miss, 1=limb, 2=torso, missing=no shot
gaze_df = pd.read_csv('/Users/jc/Documents/GitHub/shootGaze/data/shootingNoviceGaze.csv', sep=',', skipinitialspace=True)
kin_df = pd.read_csv('/Users/jc/Documents/GitHub/shootGaze/data/shootingNoviceKinematics.csv', sep=',', skipinitialspace=True)
shot_df = pd.read_csv('/Users/jc/Documents/GitHub/shootGaze/data/shootingNoviceShots.csv', sep=',', skipinitialspace=True)


# In[1272]:


# remove trailing nans
first_idx = gaze_df.first_valid_index()
last_idx = gaze_df.last_valid_index()
print(first_idx, last_idx)
gaze_df = gaze_df.loc[first_idx:last_idx]

#gaze_df['fixOn'] = (gaze_df.startTime - gaze_df.Timer)
#gaze_df['fixOff'] = (gaze_df.finishTime - gaze_df.Timer)
#gaze_df['fixLen'] = (gaze_df.fixOff - gaze_df.fixOn)

# put them in order to make table easier to view
gaze_df.insert(7, 'tFixOn', (gaze_df.startTime - gaze_df.Timer))
gaze_df.insert(8, 'tFixOff', (gaze_df.finishTime - gaze_df.Timer))
gaze_df.insert(9, 'fixDur', (gaze_df.tFixOff - gaze_df.tFixOn))
gaze_df['shotFired'] = gaze_df['shotFired'].astype('int64')
gaze_df['fixID'] = pd.Categorical(gaze_df['fixID'])
gaze_df['fixIDcode'] = gaze_df.fixID.cat.codes
#gaze_df['fixID'].value_counts()
gaze_df.head(10)


# In[1274]:


shot_df['threat1'] = pd.Categorical(shot_df['threat1'], categories=['A','B','C','D'])
shot_df['threat2'] = pd.Categorical(shot_df['threat2'], categories=['A','B','C','D'])
shot_df['targetShot1'] = pd.Categorical(shot_df['targetShot1'], categories=['A','B','C','D'])
shot_df['targetShot2'] = pd.Categorical(shot_df['targetShot2'], categories=['A','B','C','D'])
shot_df['outcomeShot1'] = pd.Categorical(shot_df['outcomeShot1'])
shot_df['outcomeShot2'] = pd.Categorical(shot_df['outcomeShot2'])


# In[1275]:


# remove trailing nans
first_idx = kin_df.first_valid_index()
last_idx = kin_df.last_valid_index()
print(first_idx, last_idx)
kin_df = kin_df.loc[first_idx:last_idx]

# convert dtypes
kin_df['Subject'] = kin_df['Subject'].astype('int64')
kin_df['Intervention'] = kin_df['Intervention'].astype('int64')
kin_df['Day'] = kin_df['Day'].astype('int64')
kin_df['Trial'] = kin_df['Trial'].astype('int64')

#subtract Timer (start of trial) from all time values
kin_df.loc[:, 'Timer':'tShot2'] = kin_df.loc[:, 'Timer':'tShot2'].sub(kin_df['Timer'], axis=0)
kin_df.head(3)


# In[1276]:


# combine data that has one row per trial
events_df = pd.merge(kin_df, shot_df, on=['Subject', 'Intervention', 'Day', 'Trial'])

# combine with gaze data (i.e. repeat trial based data for each fixation)
dat_df = pd.merge(gaze_df,events_df, on=['Subject', 'Intervention', 'Day', 'Trial'])
dat_df.tail(10)


# In[1279]:


onthreat_df = dat_df.loc[((dat_df['fixID']==dat_df['threat1']) | (dat_df['fixID']==dat_df['threat2'])), 
           ['Subject', 'Intervention', 'Day', 'Trial', 'shotFired','tFixOn', 'fixDur', 'fixID','Position','fixIDcode']]
onthreat_df.head()

#['Subject', 'Intervention', 'Day', 'Trial', 'shotFired','tFixOn', 'fixID', 'threat1', 'threat2',
#            'tOnTrigger', 'tShot1', 'tShot2', 'targetShot1', 'targetShot2', 'outcomeShot1', 'outcomeShot2']]


# ### define function to plot gaze for each subject

# In[1280]:


def plotGazeTimeline(df, subj, day, tr, axId):
    # identify this trial
    x = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].tFixOn
    width = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].fixDur
    bottom = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].Position
    height = [0.5] * len(x)

    catColor = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].fixIDcode
    mycolors = ['#d7191c','#fdae61','#abdda4','#2b83ba']
    colorlist = [mycolors[x] for x in catColor-1]
    
    if axId.ndim == 1:
        axId[tr-1].bar(x, height, width, bottom, align='edge', color=colorlist, label=catColor)
    else:
        axId[tr-1, day-1].bar(x, height, width, bottom, align='edge', color=colorlist, label=catColor)


# In[1290]:


def plotThreat(df, subj, day, tr, axId):
    # identify this trial
    x = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].tFixOn
    width = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].fixDur
    bottom = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)].Position
    height = [0.5] * len(x)
    
    if axId.ndim == 1:
        axId[tr-1].bar(x, height, width, bottom, align='edge', color='none', edgecolor = 'k', lw=1.5)
    else:
        axId[tr-1, day-1].bar(x, height, width, bottom, align='edge', color=colorlist, label=catColor)


# In[1291]:


def plotKinEvents(df, subj, day, tr, axId):
    # identify this trial
    thisTr = df.loc[(df.Subject == subj) & (df.Day == day) & (df.Trial == tr)]

    from itertools import cycle
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)

    #labels=df.loc[:,'handGun':'shot2'].columns.tolist()
    if thisTr.size > 0:
        xcoords = thisTr.loc[:,'tHandGun':'tShot2'].values
        for xc in xcoords[0]:
            if axId.ndim == 1:
                axId[tr-1].axvline(x=xc, linestyle=next(linecycler), color='k', lw=1)
            else:
                axId[tr-1, day-1].axvline(x=xc, linestyle=next(linecycler), color='k', lw=1)


# ### plot all trials for each day

# In[1305]:


# plotGazeTimeline(gaze_df, 1, 1, 2, 1)
width = 8
height = 12
nsubj = gaze_df.Subject.unique().size

for sbj in range(1,nsubj+1):
    for dy in range(1,4):
        fig, ax = plt.subplots(nrows=10, ncols=1, sharex=True, sharey=True, figsize=(width, height))
        figName = '/Users/jc/Documents/GitHub/shootGaze/figs/s' + str(sbj) + 'day' + str(dy) + 'gaze.png'
        for tr in range(1,11):
            plotGazeTimeline(gaze_df, sbj, dy, tr, ax)
            plotThreat(onthreat_df, sbj, dy, tr, ax)
            plotKinEvents(kin_df, sbj, dy, tr, ax)

        ax[0].set_title('Day ' + str(dy))
        ax[9].set_xlabel('Time (s)')

        axc=plt.gca()                            # get the axis
        axc.set_ylim([0.75,4.75])
        axc.set_ylim(axc.get_ylim()[::-1])        # invert the axis
        axc.yaxis.set_ticks([1.25,2.25,3.25,4.25]) # set y-ticks
        axc.yaxis.tick_left()                    # remove right y-Ticks
        labels = ['Far Left','Left','Right','Far Right']    
        axc.set_yticklabels(labels, fontdict=None, minor=False)

        xlims = axc.get_xlim()
        axc.set_xlim([0,xlims[1]])

        plt.tick_params(top='off', right='off', labelbottom='on')     # remove all the ticks

        #for spine in plt.gca().spines.values():
         #   spine.set_visible(False)


        patch1 = mpatches.Patch(color='#d7191c', label='A')
        patch2 = mpatches.Patch(color='#fdae61', label='B')
        patch3 = mpatches.Patch(color='#abdda4', label='C')
        patch4 = mpatches.Patch(color='#2b83ba', label='D')

        plt.legend(handles=[patch1,patch2,patch3,patch4],
                   bbox_to_anchor=(0.95, 0.95),
                   loc=1, ncol=4,
                   bbox_transform=plt.gcf().transFigure,
                   fontsize=8)

        # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        #plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
        #plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)

        # Tight layout often produces nice results but requires the title to be spaced accordingly
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        plt.savefig(figName)
        # plt.show()
        plt.close(fig)


# ### Find gaze and events when fixation on valid threat and shot fired

# In[1277]:


fired_df = dat_df.loc[(((dat_df['fixID']==dat_df['threat1']) | (dat_df['fixID']==dat_df['threat2'])) & 
           (dat_df['shotFired']==1)), 
           ['Subject', 'Intervention', 'Day', 'Trial', 'shotFired','tFixOn', 'fixID', 'threat1', 'threat2',
            'tOnTrigger', 'tShot1', 'tShot2', 'targetShot1', 'targetShot2', 'outcomeShot1', 'outcomeShot2']]
fired_df.head(10)


# ### Times of interest
# Time to ID target from trial onset (tOnTrigger)  
# Time to ID target after fixation (tFixOn - tOnTrigger)  
# Time to shoot (tOnTrigger - tShot)  

# In[1317]:


sns.countplot(y='threat1', data=fired_df, palette="Greens_d")


# In[1315]:





# In[1307]:


plt.bar(fired_df.threat1)

