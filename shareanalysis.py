import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import time


filepath='E:\\19.BSH\\'

st.set_page_config(page_title='Product Information Inquiry', page_icon=':bar_chart:', layout='wide')


st.write('-----------------------------------')
col1, col2 = st.columns([2,2])
with col1:
    month = st.radio('请选择月份', ['val_jan_23','val_feb_23','val_mar_23','val_apr_23','val_may_23','val_jun_23','val_jul_23','val_aug_23','val_sep_23','val_oct_23','val_nov_23','val_dec_23'],key='month', horizontal=True)
with col2:
    upload_models= st.file_uploader('请上传你需要的产品 xlsx格式')    
st.write('-----------------------------------')

df=pd.read_excel(upload_models,index_col=False)
df.fillna(0,inplace=True)
df1=df


#需要input 单月的哪个月
valmonth23='val_mar_23'
valmonth22=valmonth23.replace('23','22')

volmonth23=valmonth23.replace('val','vol')
volmonth22=volmonth23.replace('23','22')


df1=df
df1['ytdvol22']=df1.loc[:,'vol_jan_22':volmonth22].sum(axis=1)
df1['ytdval22']=df1.loc[:,'val_jan_22':valmonth22].sum(axis=1)
df1['ytdvol23']=df1.loc[:,'vol_jan_23':volmonth23].sum(axis=1)
df1['ytdval23']=df1.loc[:,'val_jan_23':valmonth23].sum(axis=1)


df1['mprice22']=df1[valmonth22]/df1[volmonth22]
df1['mprice23']=df1[valmonth23]/df1[volmonth23]

df1['ytdprice22']=df1['ytdval22']/df1['ytdvol22']
df1.fillna(0,inplace=True)

df1['ytdprice23']=df1['ytdval23']/df1['ytdvol23']
df1.fillna(0,inplace=True)

chose = st.radio('是否需要选择如何对价格带分组（默认:1)<3000, 2)3000 < 5000, 3)5000 < 7000, 4)7000 < 10000, 5)10000 < 15000, 6)>= 15000)？', ['Yes','No'], index=1,key='unique_key'+'price')
if chose=='No':
    df1['ytd_price_label23']=df1['ytdprice23'].apply(lambda x : '1)<3000' if x<=3000 else
                                                             ('2)3000 < 5000' if x>3000 and x<=5000 else
                                                             ('3)5000 < 7000' if x>5000 and x<=7000 else
                                                             ('4)7000 < 10000' if x>7000 and x<=10000 else
                                                             ('5)10000 < 15000' if x>10000 and x<=15000 else
                                                             ('6)>= 15000' if x>15000 else ''))))))
else:
    number=st.number_input('请输入你要分成几个价格段:',key='number price')

    df1['ytd_price_label23']=df1['ytdprice23']

    for num in range(1,int(number)+1):
        col1, col2 = st.columns([2,2])
        with col1: 
            num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值:')
            st.write(num_min)
        with col2:
            num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值:')
            st.write(num_max)        
        if num_max==0.00:
            time.sleep(20)
        else:
            df1['ytd_price_label23']=df1['ytd_price_label23'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)
    
st.dataframe(df1[['ytd_price_label23','ytdprice23']])
st.dataframe(df1, use_container_width=True)

st.write('-----------------------------------')

periods=['singlemonth', 'ytdval23']
mainchannels=df1['channel'].drop_duplicates().values.tolist()
channels=df1['sub_channel'].drop_duplicates().values.tolist()
# st.write(channels)
channels=['Total'] + mainchannels + channels

col1, col2 = st.columns(2)
with col1:
    period=st.multiselect('请选择一个你要分析的时间段:',
                    (periods), 
                    (periods))
with col2:
    channel=st.multiselect('请选择一个你要分析的渠道:',
                    (channels), 
                    (channels))


# st.write(period)
if period[0]=='singlemonth':
    period23=month
else:
    period23=period[0]

st.write(period, period23)

st.write('-----------------------------------')


# period23='ytdval23'
period22=period23.replace('23','22')

price23='mprice23'
price22=price23.replace('23','22')

var=locals()
# @st.cache_data
def table(df1,sub_channel,period23,period22,price23,price22):
    if sub_channel=='Total':
        temp=df1
    elif sub_channel=='offline' or sub_channel=='online':
        df1=df1[df1['channel']==sub_channel]
    else:
        df1=df1[df1['sub_channel']==sub_channel]

    df2=df1.groupby('Brand').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfshr=df2.apply(lambda x:x/(x.sum()),axis=0)
    dfshr.reset_index(inplace=True)
    dfshr.loc['Total']=dfshr.sum(axis=0)
    dfshr.loc['Total','Brand']='Total'
    dfshr.fillna(0,inplace=True)
    dfshr['+/-shr23']=dfshr[period23] - dfshr[period22] 
    dfshr=dfshr[['Brand',period23,'+/-shr23']]
    dfshr.round(4)

    df2=df1.groupby('Brand').agg({period22: 'sum', period23: 'sum',price22:'mean',price23:'mean'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfgr=df2
    dfgr.reset_index(inplace=True)
    dfgr.loc['Total']=dfgr.sum(axis=0)
    dfgr.loc['Total','Brand']='Total'  #选取某个单元格
    dfgr.fillna(0,inplace=True)
    dfgr['gr_23']=dfgr[period23]/dfgr[period22]-1
    dfgr['pricegr_23']=dfgr[price23]/dfgr[price22]-1
    dfgr=dfgr[['Brand','gr_23','pricegr_23']] #选取多个列 period22,period23: sales value
    dfgr.round(4)
  
    output=pd.merge(dfgr,dfshr,on='Brand',how='outer')

    mid=output['pricegr_23']   #取备注列的值
    output.pop('pricegr_23')  #删除备注列
    output.insert(4,'pricegr_23',mid) #插入备注列

    # cols=['_'.join(sub_channel)+'_'+i for i in output.columns.tolist()]
    cols=['Brand']+[sub_channel+'_'+i for i in output.columns.tolist() if i!='Brand']
    # st.write(cols)
    output.columns=cols
    output=output.round(2)
    # st.dataframe(output, use_container_width=True)

    chose = st.radio('需要选择出示哪些列吗？', ['Yes','No'],index=1, key='unique_key'+sub_channel)

    if chose=='Yes':
        fact=st.multiselect('请选择一个你要显示的列(Brand and offline value must chose):',
                        (output.columns.tolist()))
        output=output[fact]
    else:
        pass

    output.sort_values(by=[sub_channel+'_'+ period23],ascending=False,inplace=True)

    return output


sub_channels=channel  #['NKA'] #,'RKA'
for sub_channel in sub_channels:
    var[sub_channel] = table(df1,sub_channel,period23,period22,price23,price22)
    st.dataframe(var[sub_channel],use_container_width=True)

tables=var[sub_channels[0]]['Brand']
for sub_channel in sub_channels:
    tables=pd.merge(tables,var[sub_channel],on='Brand',how='outer')

tables_percent=tables
tables_percent.set_index('Brand',inplace=True)
tables_percent=tables_percent.style.format("{:.1%}")

st.dataframe(tables_percent,use_container_width=True)

st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
csv = convert_df(tables)
st.download_button(
    label="Download tables as CSV",
    data=csv,
    file_name='tables_percent.csv',
    mime='text/csv',)



def pricetable(df1,chose,variable,period23,period22,price23,price22):
    if variable=='Total':
        temp=df1
    else:
        temp=df1[df1[chose]==variable] 

    # st.dataframe(temp)

    df2=temp.groupby('ytd_price_label23').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfshr=df2.apply(lambda x:x/(x.sum()),axis=0)
    dfshr.columns=[period22+'_imp',period23+'_imp']
    dfshr[period23+'_size']=df2[period23]             #保留market size 算segment importance
    dfshr.reset_index(inplace=True)
    dfshr.loc[variable]=dfshr.sum(axis=0)             #variable 就是Total
    dfshr.loc[variable,'ytd_price_label23']=variable
    dfshr.fillna(0,inplace=True)
    dfshr['+/-imp23']=dfshr[period23+'_imp'] - dfshr[period22+'_imp'] 
    dfshr=dfshr[['ytd_price_label23',period23+'_size',period23+'_imp','+/-imp23']]
    # st.dataframe(dfshr.round(3))

    df2=temp.groupby('ytd_price_label23').agg({period22: 'sum', period23: 'sum',price22:'mean',price23:'mean'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfgr=df2
    dfgr.reset_index(inplace=True)
    dfgr.loc[variable]=dfgr.sum(axis=0)
    dfgr.loc[variable,'Brand']=variable  #选取某个单元格
    dfgr.fillna(0,inplace=True)
    dfgr['gr_23']=dfgr[period23]/dfgr[period22]-1
    dfgr['pricegr_23']=dfgr[price23]/dfgr[price22]-1
    dfgr=dfgr[['ytd_price_label23','gr_23',price23,'pricegr_23']] #选取多个列 period22,period23: sales value
    dfgr.loc[variable,'ytd_price_label23']=variable
    # st.dataframe(dfgr.round(3))

    # st.dataframe(dfgr['ytd_price_label23'].values.tolist()[:-1])
    bshshare=pd.DataFrame({})
    for aaa in dfgr['ytd_price_label23'].values.tolist()[:-1]:
        temp=df1
        temp=temp[temp['ytd_price_label23']==aaa]

        df2=temp.groupby('Brand').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
        dfshrbsh=df2.apply(lambda x:x/(x.sum()),axis=0)
        dfshrbsh.columns=[period22+'_bshshr',period23+'_bshshr']
        dfshrbsh.reset_index(inplace=True)
        dfshrbsh.loc[variable]=dfshrbsh.sum(axis=0)
        dfshrbsh.loc[variable,'Brand']=variable
        dfshrbsh.fillna(0,inplace=True)
        dfshrbsh['+/-bshshr23']=dfshrbsh[period23+'_bshshr'] - dfshrbsh[period22+'_bshshr'] 
        dfshrbsh=dfshrbsh[['Brand',period23+'_bshshr','+/-bshshr23']]

        # st.dataframe(dfshrbsh[(dfshrbsh['Brand']=='SIEMENS') | (dfshrbsh['Brand']=='BOSCH')].round(3))
        bsh=dfshrbsh[(dfshrbsh['Brand']=='SIEMENS') | (dfshrbsh['Brand']=='BOSCH')].round(3)
        bsh.loc[variable]=bsh.sum(axis=0)
        bsh.loc[variable,'Brand']=aaa  #选取某个单元格

        # st.dataframe(bsh[bsh.index==variable])
        bshshare=pd.concat([bshshare,bsh[bsh.index==variable]],axis=0)

    bshshare.reset_index(drop=True,inplace=True)
    bshshare.loc[variable]=bshshare.sum(axis=0)
    bshshare.loc[variable,'Brand']=variable  #选取某个单元格
    bshshare=bshshare.rename(columns={'Brand':'ytd_price_label23'})

    output1=pd.merge(dfshr,dfgr,on='ytd_price_label23',how='outer')
    output=pd.merge(output1,bshshare,on='ytd_price_label23',how='outer')
    st.dataframe(output)
    # break
    return(output)


chose = st.radio('请选择需要分析的变量？', ['Type','Capacity(L) /Loading KG','Width(cm)','Depth(cm)','Height(cm)'],index=0,horizontal=True,key='variables')
submark=['Total']+df1[chose].drop_duplicates(inplace=False).values.tolist()
variables=st.multiselect('请选择一个你要显示的细分市场:',
                        (submark),
                        (['Total']))

pricetables=pd.DataFrame({})

for variable in variables:
    temp=pricetable(df1,chose,variable,period23,period22,price23,price22)
    pricetables=pd.concat([pricetables,temp],axis=0)

mid=pricetables['mprice23']   #取备注列的值
pricetables.pop('mprice23')  #删除备注列
pricetables.insert(8,'mprice23',mid) #插入备注列

mid=pricetables['pricegr_23']   #取备注列的值
pricetables.pop('pricegr_23')  #删除备注列
pricetables.insert(7,'pricegr_23',mid) #插入备注列
st.dataframe(pricetables,use_container_width=True)

st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(pricetables)

st.download_button(
    label="Download pricetables as CSV",
    data=csv,
    file_name='pricetable.csv',
    mime='text/csv',
)

st.write('-----------------------------------')

import plotly.express as px

for temp, col in zip(['dfcap','dfwidth','dfdepth'],['Capacity(L) /Loading KG','Width(cm)','Depth(cm)']):
    var[temp]=df1.groupby(col).agg({period23: 'sum'})
    var[temp]['cumsum']=var[temp][period23].apply(lambda x:x/var[temp][period23].sum()).cumsum()
    # st.dataframe(temp)

dfcap1,dfwidth1,dfdepth1=st.tabs(['Capacity(L) /Loading KG','Width(cm)','Depth(cm)'])

with dfcap1:
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfcap, use_container_width=True)
    with plotlychart:
        fig = px.bar(dfcap,x=dfcap.index.values.tolist(),y='cumsum',) # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    chose = st.radio('是否需要选择如何分组（默认:<200, 200<300, 300<400, 400<500, 500<600, >600)？', ['Yes','No'], index=1,key='unique_key'+'Capacity(L) /Loading KG')

    if chose=='No':
        df1['caplabel']=df1['Capacity(L) /Loading KG'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                       '1)0 < 200' if x<200 else(
                                                                       '2)200 < 300' if x>=200 and x<300 else(
                                                                       '3)300 < 400' if x>=300 and x<400 else(
                                                                       '4)400 < 500' if x>=400 and x<500 else(
                                                                       '5)500 < 600' if x>=500 and x<600 else(
                                                                       '6)>=600' if x>=600 else np.nan)))))))
    else:
        number=st.number_input('请输入你要分成几个Capacity(L) /Loading KG段:',key='number df1cap')

        df1['caplabel']=df1['Capacity(L) /Loading KG']
        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值:', key='df1cap'+str(num))
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值:',key='df1cap2'+str(num))
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(20)
            else:
                df1['caplabel']=df1['caplabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)
            
    group=df1.groupby('caplabel').agg({period23: 'sum',period22: 'sum'})
    group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
    group['gr_23']=(group[period23]/group[period22]-1)*100
    group=group.round(2)

    fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        # fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        st.dataframe(group)   #1)0 < 200   2)200 < 300   3)300 < 400    4)400 < 500   5)500 < 600  6)600 < 600000  unknown
    with col3:
        st.dataframe(df1[['caplabel','Capacity(L) /Loading KG']])

    groupshr=pd.pivot_table(df1,index='Brand',columns='caplabel',values=period23,aggfunc="sum",margins=True)
    groupshr=groupshr.iloc[:-1,:]
    groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
    groupshr=groupshr.iloc[:6,:]
    groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
    groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

    # st.dataframe(groupshr.columns.values.tolist())


    import plotly.graph_objects as go
    import numpy as np

    importance=groupshr.columns.values.tolist()
    growth=group['gr_23'].round(2).values.tolist()
    # st.write(growth)

    widths = np.array([i*100 for i in group['cumsum'].values.tolist()])
    # st.write(widths,np.cumsum(widths)-widths)

    fig = go.Figure()
    for key in groupshr.index.values.tolist():
        # st.write(groupshr[groupshr.index==key].values.tolist()[0])
        fig.add_trace(go.Bar(
            name=key,
            y=groupshr[groupshr.index==key].values.tolist()[0],
            x=np.cumsum(widths)-widths,
            width=widths,
            offset=0,
            # customdata=np.transpose([labels, widths*data[key]]),
            texttemplate="%{y}",
            textposition='outside',
            textangle=0,
            textfont_color="white",
            marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
            width=1.0  # 设置边框线的宽度            
        ))))

    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
    )

    fig.update_xaxes(range=[0,100])
    fig.update_yaxes(range=[0,100])

    fig.update_layout(
        title_text="Marimekko Chart",
        barmode="stack",
        height=600
        # uniformtext=dict(mode="hide", minsize=10),
    )

    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    col1, col2 =st.columns([3,1])
    with col1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        # st.dataframe(groupshr2, use_container_width=True)
        table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                              cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                         height=45, align=['left', 'center'],
                                        #  fill_color='lightgrey'
                                         ))
        layout = go.Layout(title='DataFrame Table',height=700)
        fig = go.Figure(data=[table_data],layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    st.cache_data 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(groupshr2)
    st.download_button(
        label="Download groupshr2_Capacity(L) as CSV",
        data=csv,
        file_name='groupshr2_Capacity(L).csv',
        mime='text/csv',)


with dfwidth1:
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfwidth, use_container_width=True)
    with plotlychart:
        fig = px.bar(dfwidth,x=dfwidth.index.values.tolist(),y='cumsum',) # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    chose = st.radio('是否需要选择如何分组（默认:<200, 200<300, 300<400, 400<500, 500<600, >600)？', ['Yes','No'], index=1,key='unique_key'+'Width(cm)')

    if chose=='No':
        df1['caplabel']=df1['Width(cm)'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                       '1)<50' if x<50 else(
                                                                       '2)50<60' if x>=50 and x<60 else(
                                                                       '3)60<70' if x>=60 and x<70 else(
                                                                       '4)70<80' if x>=70 and x<80 else(
                                                                       '5)80<90' if x>=80 and x<90 else(
                                                                       '6)>=90' if x>=90 else np.nan)))))))
    else:
        number=st.number_input('请输入你要分成几个Capacity(L) /Loading KG段:',key='number Width(cm)')

        df1['caplabel']=df1['Width(cm)']

        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值:', key='Width'+str(num))
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值:',key='Width2'+str(num))
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(20)
            else:
                df1['caplabel']=df1['caplabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)
            
    group=df1.groupby('caplabel').agg({period23: 'sum',period22: 'sum'})
    st.write(group)    
    group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
    group['gr_23']=(group[period23]/group[period22]-1)*100
    group=group.round(2)

    fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        # fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        st.dataframe(group)   #1)0 < 200   2)200 < 300   3)300 < 400    4)400 < 500   5)500 < 600  6)600 < 600000  unknown
    with col3:
        st.dataframe(df1[['caplabel','Width(cm)']])

    groupshr=pd.pivot_table(df1,index='Brand',columns='caplabel',values=period23,aggfunc="sum",margins=True)
    groupshr=groupshr.iloc[:-1,:]
    groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
    groupshr=groupshr.iloc[:6,:]
    groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
    groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

    # st.dataframe(groupshr.columns.values.tolist())


    import plotly.graph_objects as go
    import numpy as np

    importance=groupshr.columns.values.tolist()
    growth=group['gr_23'].round(2).values.tolist()
    # st.write(growth)

    widths = np.array([i*100 for i in group['cumsum'].values.tolist()])
    # st.write(widths,np.cumsum(widths)-widths)

    fig = go.Figure()
    for key in groupshr.index.values.tolist():
        # st.write(groupshr[groupshr.index==key].values.tolist()[0])
        fig.add_trace(go.Bar(
            name=key,
            y=groupshr[groupshr.index==key].values.tolist()[0],
            x=np.cumsum(widths)-widths,
            width=widths,
            offset=0,
            # customdata=np.transpose([labels, widths*data[key]]),
            texttemplate="%{y}",
            textposition='outside',
            textangle=0,
            textfont_color="white",
            marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
            width=1.0  # 设置边框线的宽度
        ))))

    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
    )

    fig.update_xaxes(range=[0,100])
    fig.update_yaxes(range=[0,100])

    fig.update_layout(
        title_text="Marimekko Chart",
        barmode="stack",
        height=600
        # uniformtext=dict(mode="hide", minsize=10),
    )

    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    col1, col2 =st.columns([3,1])
    with col1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        # st.dataframe(groupshr2, use_container_width=True)
        table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                              cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                         height=45, align=['left', 'center'],
                                        #  fill_color='lightgrey'
                                         ))
        layout = go.Layout(title='DataFrame Table',height=700)
        fig = go.Figure(data=[table_data],layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    st.cache_data 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(groupshr2)
    st.download_button(
        label="Download groupshr2_Width(cm) as CSV",
        data=csv,
        file_name='groupshr2_Width(cm).csv',
        mime='text/csv',)


with dfdepth1: # 改这里
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfdepth, use_container_width=True) # 改这里
    with plotlychart:
        fig = px.bar(dfdepth,x=dfdepth.index.values.tolist(),y='cumsum',) # color="continent", # 改这里
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    chose = st.radio('是否需要选择如何分组Depth(cm)（默认:<55, 55<60, 60<65, 65<70, 70<75, >75)？', ['Yes','No'], index=1,key='unique_key'+'Depth(cm)')

    if chose=='No':
        df1['caplabel']=df1['Depth(cm)'].apply(lambda x: 'na' if x=='unknown' else(
                                                                       '1)<60' if x<60 else(
                                                                    #    '2)50<60' if x>=50 and x<60 else(
                                                                       '3)60<65' if x>=60 and x<65 else(
                                                                       '4)65<70' if x>=65 and x<70 else(
                                                                       '5)70<75' if x>=70 and x<75 else(
                                                                       '6)>=75' if x>=75 else np.nan))))))
    else:
        number=st.number_input('请输入你要分成几个Depth(cm)段:',key='number Depth(cm)')

        df1['caplabel']=df1['Depth(cm)'] # 改这里

        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值:', key='Depth(cm)'+str(num))
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值:',key='Depth(cm)2'+str(num))
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(20)
            else:
                df1['caplabel']=df1['caplabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)
            
    group=df1.groupby('caplabel').agg({period23: 'sum',period22: 'sum'})
    group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
    group['gr_23']=(group[period23]/group[period22]-1)*100
    group=group.round(2)

    fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        # fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        st.dataframe(group)   #1)0 < 200   2)200 < 300   3)300 < 400    4)400 < 500   5)500 < 600  6)600 < 600000  unknown
    with col3:
        st.dataframe(df1[['caplabel','Depth(cm)']])

    groupshr=pd.pivot_table(df1,index='Brand',columns='caplabel',values=period23,aggfunc="sum",margins=True)
    groupshr=groupshr.iloc[:-1,:]
    groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
    groupshr=groupshr.iloc[:6,:]
    groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
    groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

    # st.dataframe(groupshr.columns.values.tolist())


    importance=groupshr.columns.values.tolist()
    growth=group['gr_23'].round(2).values.tolist()
    # st.write(growth)

    widths = np.array([i*100 for i in group['cumsum'].values.tolist()])
    # st.write(widths,np.cumsum(widths)-widths)

    fig = go.Figure()
    for key in groupshr.index.values.tolist():
        # st.write(groupshr[groupshr.index==key].values.tolist()[0])
        fig.add_trace(go.Bar(
            name=key,
            y=groupshr[groupshr.index==key].values.tolist()[0],
            x=np.cumsum(widths)-widths,
            width=widths,
            offset=0,
            # customdata=np.transpose([labels, widths*data[key]]),
            texttemplate="%{y}",
            textposition='outside',
            textangle=0,
            textfont_color="white",
            marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
            width=1.0  # 设置边框线的宽度
        ))))

    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
    )

    fig.update_xaxes(range=[0,100],tickangle=0)
    fig.update_yaxes(range=[0,100])

    fig.update_layout(
        # title_text="Marimekko Chart",
        barmode="stack",
        height=600
        # uniformtext=dict(mode="hide", minsize=10),
    )


    col1, col2 =st.columns([3,1])
    with col1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        # st.dataframe(groupshr2, use_container_width=True)
        table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                              cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                         height=45, align=['left', 'center'],
                                        #  fill_color='lightgrey'
                                         ))
        layout = go.Layout(title='DataFrame Table',height=700)
        fig = go.Figure(data=[table_data],layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    st.cache_data 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(groupshr2)
    st.download_button(
        label="Download groupshr2_Depth(cm) as CSV",
        data=csv,
        file_name='groupshr2_Depth(cm).csv',
        mime='text/csv',)