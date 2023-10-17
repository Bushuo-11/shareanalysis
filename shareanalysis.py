import os
import io
import pandas as pd
import numpy as np
import streamlit as st


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
df1['ytd_price_label23']=df1['ytdprice23'].apply(lambda x : '1)<3000' if x<=3000 else
                                                         ('2)3000 < 5000' if x>3000 and x<=5000 else
                                                         ('3)5000 < 7000' if x>5000 and x<=7000 else
                                                         ('4)7000 < 10000' if x>7000 and x<=10000 else
                                                         ('5)10000 < 15000' if x>10000 and x<=15000 else
                                                         ('6)>= 15000' if x>15000 else ''))))))
# df1=df1.drop(['ytd_price_label22'],axis=1)

# st.dataframe(df1, use_container_width=True)

st.write('-----------------------------------')

periods=['singlemonth', 'ytdval23']
mainchannels=['offline','online']
channels=['offline','online','NKA','RKA','INDEPENDENTS','Online Vertical','Online Others']

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
    if sub_channel=='offline' or sub_channel=='online':
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

    chose = st.radio('需要选择出示哪些列吗？', ['Yes','No'], key='unique_key'+sub_channel)

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
    # st.dataframe(output)
    # break
    return(output)

chose = st.radio('请选择需要分析的细分市场？', ['Type','Capacity(L) /Loading KG','Width(cm)','Depth(cm)','Height(cm)'], key='variables')

variables=['Total']+df1[chose].drop_duplicates(inplace=False).values.tolist()
st.write(variables)

pricetables=pd.DataFrame({})

for variable in variables:
    temp=pricetable(df1,chose,variable,period23,period22,price23,price22)
    pricetables=pd.concat([pricetables,temp],axis=0)

mid=pricetables['mprice23']   #取备注列的值
pricetables.pop('mprice23')  #删除备注列
pricetables.insert(7,'mprice23',mid) #插入备注列

mid=pricetables['pricegr_23']   #取备注列的值
pricetables.pop('pricegr_23')  #删除备注列
pricetables.insert(7,'pricegr_23',mid) #插入备注列
st.dataframe(pricetables)