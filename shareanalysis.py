import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import logging
logging.basicConfig(level=logging.WARNING, filename='warnings.log')


filepath='E:\\19.BSH\\'
var=locals()

st.set_page_config(page_title='Product Information Inquiry', page_icon=':bar_chart:', layout='wide')

# st.write('-----------------------------------')
with st.sidebar:
    # col1, col2 = st.columns([2,2])
    # with col1:
    upload_models= st.file_uploader('请上传你需要的产品 xlsx格式') 
    st.write('-----------------------------------')
    month = st.radio('请选择月份', ['val_jan_23','val_feb_23','val_mar_23','val_apr_23','val_may_23','val_jun_23','val_jul_23','val_aug_23','val_sep_23','val_oct_23','val_nov_23','val_dec_23'],key='month', horizontal=True)
    # with col2:
       
# upload_models= st.file_uploader('请上传你需要的产品 xlsx格式')
df=pd.read_excel(upload_models,index_col=False)
df.fillna(0,inplace=True)

df=df.apply(lambda x:x.replace('unknown',0))

colstemp=df.columns.values.tolist()
# st.write(colstemp)

if 'DW Subcategory' not in colstemp:
    df['DW Subcategory']=0
    mid=df['DW Subcategory']
    df.pop('DW Subcategory')  #删除备注列
    df.insert(4,'DW Subcategory',mid) #插入备注列

if 'Water consumption' not in colstemp:
    df['Water consumption']=0
    mid=df['Water consumption']
    df.pop('Water consumption')  #删除备注列
    df.insert(4,'Water consumption',mid) #插入备注列

if 'Sets' not in colstemp:
    df['Sets']=0
    mid=df['Sets']
    df.pop('Sets')  #删除备注列
    df.insert(4,'Sets',mid) #插入备注列   

if ('Capacity(L)/Loading KG' not in colstemp) and ('Capacity(L)' not in colstemp):
    df['Capacity(L)/Loading KG']=0
    mid=df['Capacity(L)/Loading KG']
    df.pop('Capacity(L)/Loading KG')  #删除备注列
    df.insert(4,'Capacity(L)/Loading KG',mid) #插入备注列   

colstemp=df.columns.values.tolist()
colstemp=[i.replace("Channel",'channel') for i in colstemp]
colstemp=[i.replace("Sub_channel",'sub_channel') for i in colstemp]
colstemp=[i.replace("Groupbrand",'Brand Group') for i in colstemp]
colstemp=[i.replace("Width(mm)",'Width(cm)') for i in colstemp]
colstemp=[i.replace("Depth(mm)",'Depth(cm)') for i in colstemp]
colstemp=[i.replace("Height(mm)",'Height(cm)') for i in colstemp]
colstemp = [i.replace("Capacity(L)", "Capacity(L)/Loading KG") if i.count("Capacity(L)/Loading KG") == False else i for i in colstemp]
colstemp = [i.replace("Type of Washer", "Type") if i.count("Type of Washer") == True else i for i in colstemp]
# colstemp=[i.replace("VOL_Jan",'Height(cm)') for i in colstemp]



df.columns=colstemp

df1=df
#需要input 单月的哪个月
valmonth23=month
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


# st.write('-----------------------------------')

periods=['singlemonth', 'ytdval23']
mainchannels=df1['channel'].drop_duplicates().values.tolist()
channels=df1['sub_channel'].drop_duplicates().values.tolist()
channels=['Total'] + mainchannels + channels

# st.write(channels)

col1, col2 = st.columns(2)
with col1:
    period=st.multiselect('请选择一个你要分析的时间段:',
                    (periods), 
                    (periods))
with col2:
    channel=st.multiselect('请选择一个你要分析的渠道:',
                    (channels), 
                    (['Total']))

if period[0]=='singlemonth':
    period23=month
    price23='mprice23'
else:
    period23=period[0]
    price23='ytdprice23'

# period23='ytdval23'
period22=period23.replace('23','22')

price22=price23.replace('23','22')  #设置选择后的Price period 对应到main table 里面的mprice 列

periodvol23=period23.replace('val','vol')
periodvol22=period22.replace('val','vol')

st.write(period22, period23,  periodvol22, periodvol23,price22,price23)

st.header(':blue[     查看主表的内容]')
st.write('(time period: ', period23,' 主表df中已经包括 ytdvol22,23, ytdval 22,23, mprice22,23, ytdprice 22,23)')
st.dataframe(df1, use_container_width=True)


df1['price_label23']=df1[price23].apply(lambda x : '   1]<=3000' if x<=3000 else
                                                        ('   2]3000 <= 5000' if x>3000 and x<=5000 else
                                                        ('   3]5000 <= 7000' if x>5000 and x<=7000 else
                                                        ('   4]7000 <= 10000' if x>7000 and x<=10000 else
                                                        ('   5]10000 <= 15000' if x>10000 and x<=15000 else
                                                        ('   6]>= 15000' if x>15000 else ''))))))

df1['caplabel']=df1['Capacity(L)/Loading KG'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                '1)0 < 200' if x<200 else(
                                                                '2)200 < 300' if x>=200 and x<300 else(
                                                                '3)300 < 400' if x>=300 and x<400 else(
                                                                '4)400 < 500' if x>=400 and x<500 else(
                                                                '5)500 < 600' if x>=500 and x<600 else(
                                                                '6)>=600' if x>=600 else np.nan)))))))

df1['widthlabel']=df1['Width(cm)'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                '1)<=58' if x<=58 else(
                                                                '2)58<=68' if x>58 and x<=68 else(
                                                                '3)68<=78' if x>68 and x<=78 else(
                                                                '4)78<=88' if x>78 and x<=88 else(
                                                                '5)88<=98' if x>88 and x<=98 else(
                                                                '6)>98' if x>98 else np.nan)))))))

df1['depthlabel']=df1['Depth(cm)'].apply(lambda x: 'na' if x=='unknown' else(   # 改这里 depthlabel 'Depth(cm)'
                                                                '1)<60' if x<60 else(
                                                            #    '2)50<60' if x>=50 and x<60 else(
                                                                '3)60<65' if x>=60 and x<65 else(
                                                                '4)65<70' if x>=65 and x<70 else(
                                                                '5)70<75' if x>=70 and x<75 else(
                                                                '6)>=75' if x>=75 else np.nan))))))

df1['Waterlabel']=df1['Water consumption'].apply(lambda x: 'na' if x=='unknown' else(   # 改这里 depthlabel 'Depth(cm)'
                                                                '1)<60' if x<60 else(
                                                            #    '2)50<60' if x>=50 and x<60 else(
                                                                '3)60<65' if x>=60 and x<65 else(
                                                                '4)65<70' if x>=65 and x<70 else(
                                                                '5)70<75' if x>=70 and x<75 else(
                                                                '6)>=75' if x>=75 else np.nan))))))

df1['Setslabel']=df1['Sets'].apply(lambda x: 'na' if x=='unknown' else(   # 改这里 depthlabel 'Depth(cm)'
                                                                '1)<12' if x<12 else(
                                                                '2)12=<13' if x>=12 and x<13 else(
                                                                '3)13=<14' if x>=13 and x<14 else(
                                                                '4)14=<15' if x>=14 and x<15 else(
                                                                '5)15=<16' if x>=15 and x<16 else(
                                                                '6)>=16' if x>=16 else np.nan)))))))


st.write('--------------------------------------------------------------------')
st.header(':blue[市场增长，份额，价格变化总览 及 SKU level 分析]')  #, divider='rainbow'

# @st.cache_data
def table(df1,sub_channel,period23,period22,price23,price22):
    if sub_channel=='Total':
        temp=df1
    elif sub_channel=='offline' or sub_channel=='online'or sub_channel=='Online'or sub_channel=='Offline':
        df1=df1[df1['channel']==sub_channel]
    else:
        df1=df1[df1['sub_channel']==sub_channel]

    dfmain = df1['Brand Group']
    dfshrtrend22 = df1.loc[:,'val_jan_22':valmonth22]
    dfshrtrend22.columns=[i + '_shr' for i in dfshrtrend22.columns.values.tolist()]
    dfshrtrend22 = dfshrtrend22.apply(lambda x:x/x.sum()*100)
    dfshrtrend23 = df1.loc[:,'val_jan_23':valmonth23]
    dfshrtrend23.columns=[i + '_shr' for i in dfshrtrend23.columns.values.tolist()]
    shrchgtrend23=[i + '_shrchg' for i in dfshrtrend23.columns.values.tolist()]
    dfshrtrend23 = dfshrtrend23.apply(lambda x:x/x.sum()*100)

    dfshrall=pd.concat([dfmain, dfshrtrend22,dfshrtrend23],axis=1)
    # st.write(dfshrall)

    colmunsall=dfshrtrend22.columns+dfshrtrend23.columns
    dftrend_table=pd.pivot_table(dfshrall,index='Brand Group',aggfunc=sum)
    # st.write(dftrend_table)

    dftrendchg_table=pd.DataFrame({})
    for shr22, shr23, shrchg23 in zip(dfshrtrend22.columns.values.tolist(),dfshrtrend23.columns.values.tolist(),shrchgtrend23):
        dftrendchg_table[shrchg23]=dftrend_table[shr23] - dftrend_table[shr22]

    dftrendchg_table=dftrendchg_table.round(2)
    st.write(dftrendchg_table,dftrendchg_table.index.values.tolist())

    fig1 = go.Figure()
    for key in dftrendchg_table.index.values.tolist():
        # st.write(key)
        # st.write(dftrendchg_table[dftrendchg_table.index==key].values.tolist()[0])
        # st.write(dftrendchg_table.columns.values.tolist())

        fig1.add_trace(go.Bar(
            name=key,
            y=dftrendchg_table[dftrendchg_table.index==key].values.tolist()[0],
            x=dftrendchg_table.columns.values.tolist(),
            # offset=0,
            # customdata=np.transpose([labels, widths*data[key]]),
            texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
            textposition='outside',
            textangle=0,
            textfont_color="white",
            marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
            width=1.0  # 设置边框线的宽度            
        ))))


    # fig.update_xaxes(range=[0,100])
    # fig1.update_yaxes(range=[-3,3])

    fig1.update_layout(
        title_text="share change by month",
        barmode='relative',
        plot_bgcolor='white',  # 设置图表背景颜色为白色
        paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
        height=600
        # uniformtext=dict(mode="hide", minsize=10),
    )

    st.plotly_chart(fig1, use_container_width=True)

    # df1['Brand Group'].fillna('na',inplace=True)
    df1['Brand Group']=df1['Brand Group'].apply(lambda x: 'zzzz' if x==0 else x)
    # st.write(df1)
    df2=df1.groupby('Brand Group').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfshr=df2.apply(lambda x:x/(x.sum()),axis=0)
    dfshr.reset_index(inplace=True)
    dfshr.loc['Total']=dfshr.sum(axis=0)
    dfshr.loc['Total','Brand Group']='Total'
    dfshr.fillna(0,inplace=True)
    dfshr['+/-shr23']=dfshr[period23] - dfshr[period22] 
    dfshr=dfshr[['Brand Group',period23,'+/-shr23']]
    dfshr.round(4)

    df2=df1.groupby('Brand Group').agg({period22: 'sum', period23: 'sum',periodvol22:'sum',periodvol23:'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfgr=df2
    dfgr.reset_index(inplace=True)
    dfgr.loc['Total']=dfgr.sum(axis=0)
    dfgr.loc['Total','Brand Group']='Total'  #选取某个单元格
    dfgr.fillna(0,inplace=True)
    dfgr['gr_23']=dfgr[period23]/dfgr[period22]-1
    dfgr['pricegr_23']=(dfgr[period23]/dfgr[periodvol23])/(dfgr[period22]/dfgr[periodvol22])-1
    dfgr[price23]=dfgr[period23]/dfgr[periodvol23]
    dfgr=dfgr[['Brand Group','gr_23','pricegr_23']] #选取多个列 period22,period23: sales value
    dfgr.round(4)
  
    outputgroup=pd.merge(dfgr,dfshr,on='Brand Group',how='outer')
    outputgroup.set_index('Brand Group',inplace=True)
    st.caption(':blue[集团公司销售金额和份额变化]')  #, divider='rainbow'
    st.dataframe(outputgroup.style.format("{:.1%}"),use_container_width=True)

    @st.cache_data 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(outputgroup)
    st.download_button(
        label="Download BrandGoup share as CSV",
        data=csv,
        file_name="Download BrandGoup share as CSV"+'.csv',
        mime='text/csv',)


    # df1['Brand'].fillna('na',inplace=True)
    df1['Brand']=df1['Brand'].apply(lambda x: 'zzzz' if x==0 else x)
    # st.write(df1)
    df2=df1.groupby('Brand').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfshr=df2.apply(lambda x:x/(x.sum()),axis=0)
    dfshr.reset_index(inplace=True)
    dfshr.loc['Total']=dfshr.sum(axis=0)
    dfshr.loc['Total','Brand']='Total'
    dfshr.fillna(0,inplace=True)
    dfshr['+/-shr23']=dfshr[period23] - dfshr[period22] 
    dfshr=dfshr[['Brand',period23,'+/-shr23']]
    dfshr.round(4)

    df2=df1.groupby('Brand').agg({period22: 'sum', period23: 'sum',periodvol22:'sum',periodvol23:'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfgr=df2
    dfgr.reset_index(inplace=True)
    dfgr.loc['Total']=dfgr.sum(axis=0)
    dfgr.loc['Total','Brand']='Total'  #选取某个单元格
    dfgr.fillna(0,inplace=True)
    dfgr['gr_23']=dfgr[period23]/dfgr[period22]-1
    dfgr['pricegr_23']=(dfgr[period23]/dfgr[periodvol23])/(dfgr[period22]/dfgr[periodvol22])-1
    dfgr[price23]=dfgr[period23]/dfgr[periodvol23]
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
    output=output.round(4)
    # st.dataframe(output, use_container_width=True)

    # chose = st.radio('需要选择出示哪些列吗(growth, share, share & price change)？', ['Yes','No'],index=1,horizontal=True, key='unique_key'+sub_channel)

    # if chose=='Yes':
    #     fact=st.multiselect('请选择一个你要显示的列(Brand and offline value must chose):',
    #                     (output.columns.tolist()))
    #     output=output[fact]
    # else:
    #     pass

    output.sort_values(by=[sub_channel+'_'+ period23],ascending=False,inplace=True)
    st.dataframe(output,use_container_width=True)

    @st.cache_data 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(outputgroup)
    st.download_button(
        label="Download Brand share as CSV",
        data=csv,
        file_name="Download Brand share as CSV"+'.csv',
        mime='text/csv',)


    st.subheader(':blue[SKU share By brand 的分析]')
    chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗 (默认+/- share >0.2%？)', ['Yes','No'],index=1, horizontal=True,key='sku share'+sub_channel)

    if chose=='Yes':
        fact=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                        (output.Brand.tolist()))
    else:
        fact=output[output[sub_channel+'_'+ '+/-shr23']>0.002].Brand.values.tolist()

    st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌 (默认+/- share >0.2%？)]')
    sumval23=df1[period23].sum()
    sumval22=df1[period22].sum()

    for brand in fact:
        st.write(brand)
        # st.write(df1[df1['Brand']==brand])
            
        dfbrand=df1[df1['Brand']==brand][['Brand','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Launch date',period23,period22]]
        dfbrand['shr_change']=( dfbrand[period23]/sumval23 - dfbrand[period22]/sumval22)
        dfbrand.loc['Total']=dfbrand.apply(lambda x:x.sum())
        dfbrand.loc['Total','Brand']='Total'
        dfbrand.loc['Total','Model']='Total'
        dfbrand.loc['Total','Launch date']='Total'

        dfbrand.sort_values(by=['shr_change'],ascending=False,inplace=True)
        dfbrand['url']='https://www.baidu.com/s?wd=' + dfbrand['Model']
        st.write(dfbrand)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(dfbrand)
        st.download_button(
            label="Download tables as CSV"+'_'+brand,
            data=csv,
            file_name="Download tables as CSV"+'_'+brand+'.csv',
            mime='text/csv',)

    return output

var=globals()
sub_channels=channel  #['NKA'] #,'RKA'
for sub_channel in sub_channels:
    var[sub_channel] = table(df1,sub_channel,period23,period22,price23,price22)
    # st.dataframe(var[sub_channel],use_container_width=True)

st.subheader(':blue[显示所有渠道不同品牌的增长，份额，份额及价格变化]')
tables=var[sub_channels[0]]['Brand']
for sub_channel in sub_channels:
    tables=pd.merge(tables,var[sub_channel],on='Brand',how='outer')

tables_percent=tables
tables_percent.set_index('Brand',inplace=True)
tables_percent=tables_percent.style.format("{:.1%}")

st.dataframe(tables_percent,use_container_width=True)

@st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
csv = convert_df(tables)
st.download_button(
    label="Download tables as CSV",
    data=csv,
    file_name='tables_percent.csv',
    mime='text/csv',)


st.write('--------------------------------------------------------------------')
st.header(':blue[根据市场特征，制作各个变量对应的标签]')  #, divider='rainbow'


for temp, col in zip(['dftype','dfcap','dfwidth','dfdepth','dfSubcate','dfWater','dfSets'],['Type','Capacity(L)/Loading KG','Width(cm)','Depth(cm)','DW Subcategory','Water consumption','Sets']):

    var[temp]=df1.groupby(col).agg({period23: 'sum'})
    var[temp]['cumsum']=var[temp][period23].apply(lambda x:x/var[temp][period23].sum()).cumsum()
    # st.dataframe(temp)

dftype1,dfprice1,dfcap1,dfwidth1,dfdepth1,dfSubcate1,dfWater1,dfSets1=st.tabs(['Type','price','Capacity(L)/Loading KG','Width(cm)','Depth(cm)','DW Subcategory(DC only)','Water consumption(DC only)','Sets(DC only)'])

with dftype1:
    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dftype1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write('Type中要分析的渠道: ',chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        dftype=dftemp.groupby('Type').agg({period23: 'sum'})
        dftype['cumsum']=dftype[period23].apply(lambda x:x/dftype[period23].sum()).cumsum()

        plotlychart,col2,col1 = st.columns([4,1,2])
        with plotlychart:
            fig = px.bar(dftype,x=dftype.index.values.tolist(),y='cumsum',) # color="continent",
            st.plotly_chart(fig,  use_container_width=True)
        with col1:
            st.dataframe(dftype, use_container_width=True) 
        # with col2:
        #     st.dataframe(dftype, use_container_width=True)                        

        group=dftemp.groupby('Type').agg({period23: 'sum',period22: 'sum'})
        group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
        group['gr_23']=(group[period23]/group[period22]-1)*100
        group=group.round(3)

        fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            # fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with col2:
            st.dataframe(group)   #1)0 < 200   2)200 < 300   3)300 < 400    4)400 < 500   5)500 < 600  6)600 < 600000  unknown
        with col3:
            st.dataframe(dftemp[['Type']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='Type',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(41)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='Type',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(3)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_Type as CSV",
            data=csv,
            file_name='groupshr2_Type.csv',
            mime='text/csv',
            key='unique_key'+'Type groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_Type as CSV",
            data=csv,
            file_name='groupshrchg2_Type.csv',
            mime='text/csv',
            key='unique_key'+'Type'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x caplabel'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的细分市场段:',
                                (dftemp['Type'].drop_duplicates().tolist()),
                                (dftemp['Type'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['Type']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['Type']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets','price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel'],index=0,horizontal=True, key='dfbrandpriceformat0caplabel'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x Type2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',period23,period22,'caplabel','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

with dfprice1:

    dfprice=df1.groupby(price23).agg({period23: 'sum'})
    dfprice['cumsum']=dfprice[period23].apply(lambda x:x/dfprice[period23].sum()).cumsum()
    dfprice.reset_index(drop=False,inplace=True)
    dfprice[price23]=dfprice[price23].round(0)
    # st.write(dfprice)

    plotlychart, col1= st.columns([2,2])
    with plotlychart:
        fig = px.scatter(dfprice,x=price23,y='cumsum',) # color="continent",
        st.plotly_chart(fig,  use_container_width=True)
    with col1:
        st.dataframe(dfprice, use_container_width=True)

    #制作 YTD 价格标签
    chose = st.radio('是否需要选择如何对价格带分组（默认:1)<3000, 2)3000 < 5000, 3)5000 < 7000, 4)7000 < 10000, 5)10000 < 15000, 6)>= 15000)？', ['Yes','No'], index=1,key='unique_key'+'dfprice')
    if chose=='No':
        df1['price_label23']=df1[price23].apply(lambda x : '   1]<=3000' if x<=3000 else
                                                                ('   2]3000 <= 5000' if x>3000 and x<=5000 else
                                                                ('   3]5000 <= 7000' if x>5000 and x<=7000 else
                                                                ('   4]7000 <= 10000' if x>7000 and x<=10000 else
                                                                ('   5]10000 <= 15000' if x>10000 and x<=15000 else
                                                                ('   6]>= 15000' if x>15000 else ''))))))
    else:
        number=st.number_input('请输入你要分成几个价格段:',key='number price')

        df1['price_label23']=df1[price23]

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
                df1['price_label23']=df1['price_label23'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)

    st.write('检查价格带分类是否正确')   
    st.dataframe(df1[['price_label23',price23]].head(5)) 

    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfprice1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]






        group=dftemp.groupby('price_label23').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['price_label23','ytdprice23']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='price_label23',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='price_label23',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        # st.write('£££££££££££££££££',groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_dfprice as CSV",
            data=csv,
            file_name='groupshr2_dfprice.csv',
            mime='text/csv',
            key='unique_key'+'dfprice groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_dfprice as CSV",
            data=csv,
            file_name='groupshrchg2_dfprice.csv',
            mime='text/csv',
            key='unique_key'+'dfprice'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x price'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的价格段:',
                                (dftemp['price_label23'].drop_duplicates().tolist()))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['price_label23']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',period23,period22,'price_label23']]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['price_label23']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',period23,period22,'price_label23']]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel'],index=0,horizontal=True, key='dfbrandpriceformat0'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write(pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']))

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',period23,period22,'price_label23','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

with dfcap1:
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfcap, use_container_width=True)
    with plotlychart:
        fig = px.bar(dfcap,x=dfcap.index.values.tolist(),y='cumsum',) # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    chose = st.radio('是否需要选择如何分组（默认:<200, 200<300, 300<400, 400<500, 500<600, >600)？', ['Yes','No'], index=1,key='unique_key'+'Capacity(L)/Loading KG')

    if chose=='No':
        df1['caplabel']=df1['Capacity(L)/Loading KG'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                       '1)0 < 200' if x<200 else(
                                                                       '2)200 < 300' if x>=200 and x<300 else(
                                                                       '3)300 < 400' if x>=300 and x<400 else(
                                                                       '4)400 < 500' if x>=400 and x<500 else(
                                                                       '5)500 < 600' if x>=500 and x<600 else(
                                                                       '6)>=600' if x>=600 else np.nan)))))))
    else:
        number=st.number_input('请输入你要分成几个Capacity(L)/Loading KG段:',key='number df1cap')

        df1['caplabel']=df1['Capacity(L)/Loading KG']
        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入容量第 '+ str(num) +' 段容量范围_最小值:', key='df1cap'+str(num))
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入容量第 '+ str(num) +' 段容量范围_最大值:',key='df1cap2'+str(num))
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(20)
            else:
                df1['caplabel']=df1['caplabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)


    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfcap1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]


        group=dftemp.groupby('caplabel').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['caplabel','Capacity(L)/Loading KG']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='caplabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='caplabel',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_caplabel as CSV",
            data=csv,
            file_name='groupshr2_caplabel.csv',
            mime='text/csv',
            key='unique_key'+'caplabel groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_caplabel as CSV",
            data=csv,
            file_name='groupshrchg2_caplabel.csv',
            mime='text/csv',
            key='unique_key'+'caplabel'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x caplabel2'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的Capacity段:',
                                (dftemp['caplabel'].drop_duplicates().tolist()),
                                (dftemp['caplabel'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['caplabel']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['caplabel']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets','price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel'],index=0,horizontal=True, key='dfbrandpriceformat0pricelabel'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x caplabel2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',period23,period22,'caplabel','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

with dfwidth1:
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfwidth, use_container_width=True)
    with plotlychart:
        fig = px.bar(dfwidth,x=dfwidth.index.values.tolist(),y='cumsum',) # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    chose = st.radio('是否需要选择如何分组（默认:<=58, 58<=68, 68<=78, 78<=88, 88<=98, >98)？', ['Yes','No'], index=1,key='unique_key'+'Width(cm)')

    if chose=='No':
        df1['widthlabel']=df1['Width(cm)'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                       '1)<=58' if x<=58 else(
                                                                       '2)58<=68' if x>58 and x<=68 else(
                                                                       '3)68<=78' if x>68 and x<=78 else(
                                                                       '4)78<=88' if x>78 and x<=88 else(
                                                                       '5)88<=98' if x>88 and x<=98 else(
                                                                       '6)>98' if x>98 else np.nan)))))))
    else:
        number=st.number_input('请输入你要分成几个Capacity(L)/Loading KG段:',key='number Width(cm)')

        df1['widthlabel']=df1['Width(cm)']

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
                df1['widthlabel']=df1['widthlabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)
            
    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfwidth1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        group=dftemp.groupby('widthlabel').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['widthlabel','Capacity(L)/Loading KG']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='widthlabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='widthlabel',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2widthlabel as CSV",
            data=csv,
            file_name='groupshr2_widthlabel.csv',
            mime='text/csv',
            key='unique_key'+'widthlabel groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_widthlabel as CSV",
            data=csv,
            file_name='groupshrchg2_widthlabel.csv',
            mime='text/csv',
            key='unique_key'+'widthlabel'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x widthlabel'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的Capacity段:',
                                (dftemp['widthlabel'].drop_duplicates().tolist()),
                                (dftemp['widthlabel'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['widthlabel']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['widthlabel']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets'],index=0,horizontal=True, key='dfbrandpriceformat0widthlabel'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x widthlabel2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                        "Capacity(L)/Loading KG",'Sets',period23,period22,'widthlabel','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass
 
with dfdepth1: # 改这里
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfdepth, use_container_width=True) # 改这里 dfdepth
    with plotlychart:
        fig = px.bar(dfdepth,x=dfdepth.index.values.tolist(),y='cumsum',) # color="continent", # 改这里 dfdepth
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                                                                                                               # 改这里 'Depth(cm)'
    chose = st.radio('是否需要选择如何分组Depth(cm)（默认:<55, 55<60, 60<65, 65<70, 70<75, >75)？', ['Yes','No'], index=1,key='unique_key'+'Depth(cm)')

    if chose=='No':
        df1['depthlabel']=df1['Depth(cm)'].apply(lambda x: 'na' if x=='unknown' else(   # 改这里 depthlabel 'Depth(cm)'
                                                                       '1)<60' if x<60 else(
                                                                    #    '2)50<60' if x>=50 and x<60 else(
                                                                       '3)60<65' if x>=60 and x<65 else(
                                                                       '4)65<70' if x>=65 and x<70 else(
                                                                       '5)70<75' if x>=70 and x<75 else(
                                                                       '6)>=75' if x>=75 else np.nan))))))
    else:
        number=st.number_input('请输入你要分成几个Depth(cm)段:',key='number Depth(cm)') # 改这里 'Depth(cm)'

        df1['depthlabel']=df1['Depth(cm)'] # 改这里

        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值:', key='Depth(cm)'+str(num)) # 改这里 'Depth(cm)'
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值:',key='Depth(cm)2'+str(num)) # 改这里 'Depth(cm)'
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(20)
            else:                                              # 改这里 'depthlabel'
                df1['depthlabel']=df1['depthlabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)

    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfdepth1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]


        group=dftemp.groupby('depthlabel').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['depthlabel','Capacity(L)/Loading KG']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='depthlabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='depthlabel',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2depthlabel as CSV",
            data=csv,
            file_name='groupshr2_depthlabel.csv',
            mime='text/csv',
            key='unique_key'+'depthlabel groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_depthlabel as CSV",
            data=csv,
            file_name='groupshrchg2_depthlabel.csv',
            mime='text/csv',
            key='unique_key'+'depthlabel'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x depthlabel'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的Capacity段:',
                                (dftemp['depthlabel'].drop_duplicates().tolist()),
                                (dftemp['depthlabel'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['depthlabel']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['depthlabel']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets'],index=0,horizontal=True, key='dfbrandpriceformat0depthlabel'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x depthlabel2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                        "Capacity(L)/Loading KG",'Sets',period23,period22,'depthlabel','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

with dfWater1: # 改这里
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfWater, use_container_width=True) # 改这里 dfdepth
    with plotlychart:
        fig = px.bar(dfWater,x=dfWater.index.values.tolist(),y='cumsum',) # color="continent", # 改这里 dfdepth
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                                                                                                               # 改这里 'Depth(cm)'
    chose = st.radio('是否需要选择如何分组Water consumption（默认:<55, 55<60, 60<65, 65<70, 70<75, >75)？', ['Yes','No'], index=1,key='unique_key'+'Water consumption')

    if chose=='No':
        df1['Waterlabel']=df1['Water consumption'].apply(lambda x: 'na' if x=='unknown' else(   # 改这里 depthlabel 'Depth(cm)'
                                                                       '1)<60' if x<60 else(
                                                                    #    '2)50<60' if x>=50 and x<60 else(
                                                                       '3)60<65' if x>=60 and x<65 else(
                                                                       '4)65<70' if x>=65 and x<70 else(
                                                                       '5)70<75' if x>=70 and x<75 else(
                                                                       '6)>=75' if x>=75 else np.nan))))))
    else:
        number=st.number_input('请输入你要分成几个Water consumption段:',key='number Water consumption') # 改这里 'Depth(cm)'

        df1['Waterlabel']=df1['Water consumption'] # 改这里

        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值:', key='Water consumption'+str(num)) # 改这里 'Depth(cm)'
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值:',key='Water consumption'+str(num)) # 改这里 'Depth(cm)'
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(20)
            else:                                              # 改这里 'depthlabel'
                df1['Waterlabel']=df1['Waterlabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)

    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfSubcate1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        group=dftemp.groupby('Waterlabel').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['Waterlabel','Capacity(L)/Loading KG']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='Waterlabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='Waterlabel',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2Waterlabel as CSV",
            data=csv,
            file_name='groupshr2_Waterlabel.csv',
            mime='text/csv',
            key='unique_key'+'Waterlabel groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_Waterlabel as CSV",
            data=csv,
            file_name='groupshrchg2_Waterlabel.csv',
            mime='text/csv',
            key='unique_key'+'Waterlabel'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x Waterlabel'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的Capacity段:',
                                (dftemp['Waterlabel'].drop_duplicates().tolist()),
                                (dftemp['Waterlabel'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['Waterlabel']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['Waterlabel']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets'],index=0,horizontal=True, key='dfbrandpriceformat0Waterlabel'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x Waterlabel2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                        "Capacity(L)/Loading KG",'Sets',period23,period22,'Waterlabel','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

with dfSets1: # 改这里
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfSets, use_container_width=True) # 改这里 dfdepth
    with plotlychart:
        fig = px.bar(dfSets,x=dfSets.index.values.tolist(),y='cumsum',) # color="continent", # 改这里 dfdepth
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                                                                                                               # 改这里 'Depth(cm)'
    chose = st.radio('是否需要选择如何分组Sets（默认:<12, 12=<13, 13=<14, 14=<15, 15=<16, >=16)？', ['Yes','No'], index=1,key='unique_key'+'Sets')

    if chose=='No':
        df1['Setslabel']=df1['Sets'].apply(lambda x: 'na' if x=='unknown' else(   # 改这里 depthlabel 'Depth(cm)'
                                                                       '1)<12' if x<12 else(
                                                                       '2)12=<13' if x>=12 and x<13 else(
                                                                       '3)13=<14' if x>=13 and x<14 else(
                                                                       '4)14=<15' if x>=14 and x<15 else(
                                                                       '5)15=<16' if x>=15 and x<16 else(
                                                                       '6)>=16' if x>=16 else np.nan)))))))
    else:
        number=st.number_input('请输入你要分成几个Sets段:',key='number Sets') # 改这里 'Depth(cm)'

        df1['Setslabel']=df1['Sets'] # 改这里

        for num in range(1,int(number)+1):
            col1, col2 = st.columns([2,2])
            with col1: 
                num_min=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最小值(>=):', key='Sets1'+str(num)) # 改这里 'Depth(cm)'
                st.write(num_min)
            with col2:
                num_max=st.number_input('请输入价格第 '+ str(num) +' 段价格范围_最大值(<):',key='Sets2'+str(num)) # 改这里 'Depth(cm)'
                st.write(num_max)        
            if num_max==0.00:
                time.sleep(10)
            else:                                              # 改这里 'depthlabel'
                df1['Setslabel']=df1['Setslabel'].apply(lambda x : str(num)+')'+str(int(num_min))+' < '+ str(int(num_max)) if not isinstance(x, str) and x>=num_min and x<num_max else x)

    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfSets1') # 改这里 dfSets1: 
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        group=dftemp.groupby('Setslabel').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['Setslabel','Sets']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='Setslabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='Setslabel',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[i + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[i + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2Setslabel as CSV",
            data=csv,
            file_name='groupshr2_Setslabel.csv',
            mime='text/csv',
            key='unique_key'+'Setslabel groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_Setslabel as CSV",
            data=csv,
            file_name='groupshrchg2_Setslabel.csv',
            mime='text/csv',
            key='unique_key'+'Setslabel'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x Setslabel'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的Capacity段:',
                                (dftemp['Setslabel'].drop_duplicates().tolist()),
                                (dftemp['Setslabel'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['Setslabel']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['Setslabel']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22]]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets'],index=0,horizontal=True, key='dfbrandpriceformat0Setslabel'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x Setslabel2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(pricetier + ' ------- ' + brand)
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                        "Capacity(L)/Loading KG",'Sets',period23,period22,'Setslabel','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

with dfSubcate1:
    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfSets1: ') #改这里 dfSets1: 
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        dfSubcate=dftemp.groupby('DW Subcategory').agg({period23: 'sum'})   #改这里 dftype   #改这里 'Type'
        dfSubcate['cumsum']=dfSubcate[period23].apply(lambda x:x/dfSubcate[period23].sum()).cumsum() #改这里 dftype

        col1, plotlychart = st.columns([2,2])
        with col1:
            st.dataframe(dfSubcate, use_container_width=True) #改这里 dftype
        with plotlychart:
            fig = px.bar(dfSubcate,x=dfSubcate.index.values.tolist(),y='cumsum',) # color="continent",   #改这里 dftype
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                

        group=dftemp.groupby('DW Subcategory').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['DW Subcategory']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='DW Subcategory',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='DW Subcategory',values=period22,aggfunc="sum",margins=True)
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        st.write(groupshr22)

        cols=[str(i) + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[str(i) + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        st.write('£££££££££££££££££',group,groupshr,groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化
        groupshr_chg_short=groupshr_chg_short.iloc[:,:-1].fillna(0).round(1)
        groupshr_chg_short.columns=groupshr.columns

        # st.write(groupshr,groupshr_chg_short)

        import plotly.graph_objects as go
        import numpy as np

        for coltest in groupshr.columns.values.tolist():
            if groupshr[coltest][-1]==100:
                groupshr.pop(coltest)
                group.drop(group[group.index.str.contains(coltest)].index,inplace=True)

        # st.write(groupshr,group)

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(2).values.tolist()
        # st.write(groupshr,group['gr_23'])

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])

        # st.write(widths,np.cumsum(widths)-widths)

        growth=[0 if pd.isnull(i)==True else i for i in growth]
        # st.write(importance, widths,growth)

        from plotly.subplots import make_subplots
        import plotly.subplots as sp

        # theme_plotly = None
        # fig = go.Figure()
        fig = sp.make_subplots(specs=[[{'secondary_y': True}]])
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr.index.values.tolist():
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig.add_trace(go.Bar(
                name=key,
                y=groupshr[groupshr.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
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
            title_text="Mekko_"+ chose,
            barmode="stack",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                                cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)



        importance=groupshr_chg_short.columns.values.tolist()
        # st.write(groupshr_chg_short)
        # st.write(importance)
        fig2 = go.Figure()
        # fig = make_subplots(rows=1, cols=1,print_grid=True)
        for key in groupshr_chg_short.index.values.tolist():
            # st.write(key)
            # st.write(groupshr[groupshr.index==key].values.tolist()[0])
            fig2.add_trace(go.Bar(
                name=key,
                y=groupshr_chg_short[groupshr_chg_short.index==key].values.tolist()[0],
                x=np.cumsum(widths)-widths,
                width=widths,
                offset=0,
                # customdata=np.transpose([labels, widths*data[key]]),
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig2.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        # fig2.update_xaxes(range=[-10,10])
        fig2.update_yaxes(range=[-10,10])

        fig2.update_layout(
            title_text="Mekko_"+ chose,
            barmode="relative",
            plot_bgcolor='white',  # 设置图表背景颜色为白色
            paper_bgcolor='rgb(250, 250, 250)',  # 设置打印纸背景颜色为浅灰色
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.write(groupshr_chg_short)

        group=group[['cumsum','gr_23']]
        groupshrchg2=pd.concat([group.T,groupshr_chg_short],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        # st.dataframe(groupshrchg2, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            table_data = go.Table(header=dict(values=['Total']+list(groupshrchg2.columns)),
                                cells=dict(values=[groupshrchg2.index] + [groupshrchg2[col] for col in groupshrchg2.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2DW Subcategory as CSV",
            data=csv,
            file_name='groupshr2_DW Subcategory.csv',
            mime='text/csv',
            key='unique_key'+'DW Subcategory groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshrchg2)
        st.download_button(
            label="Download groupshrchg2_DW Subcategory as CSV",
            data=csv,
            file_name='groupshrchg2_DW Subcategory.csv',
            mime='text/csv',
            key='unique_key'+'DW Subcategory'+chose)
        

        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x DW Subcategory'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的Capacity段:',
                                (dftemp['DW Subcategory'].drop_duplicates().tolist()),
                                (dftemp['DW Subcategory'].drop_duplicates().tolist()[0]))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            skutemp0=dftemp[(dftemp['DW Subcategory']==pricetiers[0])][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22,'DW Subcategory']]
            skutemp0['share22']=skutemp0[period22].apply(lambda x:x/skutemp0[period22].sum(axis=0)*100)
            skutemp0['share23']=skutemp0[period23].apply(lambda x:x/skutemp0[period23].sum(axis=0)*100)
            skutemp0['shrchg23']=(skutemp0['share23']-skutemp0['share22']).round(2)
            st.write(skutemp0)

            for pricetier in pricetiers:
                # dfbrandprice=pd.DataFrame({})
                skutemp=skutemp0[(skutemp0['DW Subcategory']==pricetier)][['Brand','Type','Model','Width(cm)','Depth(cm)',"Capacity(L)/Loading KG",'Sets',
            'price_label23','caplabel','widthlabel','depthlabel','Waterlabel','Setslabel',period23,period22,'DW Subcategory']]
                skutemp['share22']=skutemp[period22].apply(lambda x:x/skutemp[period22].sum(axis=0)*100)
                skutemp['share23']=skutemp[period23].apply(lambda x:x/skutemp[period23].sum(axis=0)*100)
                skutemp['shrchg23']=(skutemp['share23']-skutemp['share22']).round(2)
                brandprice=pd.pivot_table(skutemp,index='Brand',values='shrchg23',aggfunc=sum,margins=True)
                st.write(brandprice)   

                chose2 = st.radio('需要看这个价格段下，哪个维度的信息?', ['No | ','Type','Width(cm)','Depth(cm)',
                "Capacity(L)/Loading KG",'Sets'],index=0,horizontal=True, key='dfbrandpriceformat0DW Subcategory'+ chose)
                if chose2!='No | ':
                    dfbrandpriceformat=pd.pivot_table(skutemp,index='Brand',columns=[chose2],values='shrchg23',aggfunc=sum,margins=True)     
                    st.write('分析的是: ',pricetier, dfbrandpriceformat)   

                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (['No'] + dftemp.Brand.tolist()), (['No']),key='sku share x DW Subcategory2'+ chose)

                if brands[0]!='No':
                    for brand in brands:                
                        st.write(str(pricetier) + ' ------- ' + str(brand))
                        # st.write(df1[df1['Brand']==brand])                        
                        dfbrandprice=skutemp[(skutemp['Brand']==brand)][['Brand','Type','Model','Width(cm)','Depth(cm)',
                        "Capacity(L)/Loading KG",'Sets',period23,period22,'DW Subcategory','shrchg23']]
                        dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                        dfbrandprice.loc['Total','Brand']='Total'
                        dfbrandprice.loc['Total','Model']='Total'
                        dfbrandprice.loc['Total','Type']='Total'
                        dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                        dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                        st.write(dfbrandprice)   

                else:
                    pass
        else:
            pass

st.write('--------------------------------------------------------------')
st.header(':blue[下面是当月/YTD份额表现的细化分析 (include where to play)]')
st.write('--------------------------------------------------------------')
def pricetable(df1,sub_channel,chose,variable,period23,period22,price23,price22):
    if sub_channel=='Total':
        df1=df1
    elif sub_channel=='offline' or sub_channel=='online'or sub_channel=='Online'or sub_channel=='Offline':
        df1=df1[df1['channel']==sub_channel]
    else:
        df1=df1[df1['sub_channel']==sub_channel]

    if variable=='Total':
        temp=df1
    else:
        temp=df1[df1[chose]==variable] 

    # st.dataframe(temp)
    df2=temp.groupby('price_label23').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    dfshr=df2.apply(lambda x:x/(x.sum()),axis=0)
    dfshr.columns=[period22+'_imp',period23+'_imp']
    dfshr[period23+'_size']=df2[period23]             #保留market size 算segment importance
    dfshr.reset_index(inplace=True)
    dfshr.loc[variable]=dfshr.sum(axis=0)             #variable 就是Total
    dfshr.loc[variable,'price_label23']=variable
    dfshr.fillna(0,inplace=True)
    dfshr['+/-imp23']=dfshr[period23+'_imp'] - dfshr[period22+'_imp'] 
    dfshr=dfshr[['price_label23',period23+'_size',period23+'_imp','+/-imp23']]
    # st.dataframe(dfshr.round(3))

    df2=temp.groupby('price_label23').agg({period22: 'sum', period23: 'sum',periodvol22:'sum',periodvol23:'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
    # st.write(df2)
    dfgr=df2
    dfgr.reset_index(inplace=True)
    dfgr.loc[variable]=dfgr.sum(axis=0)
    dfgr.loc[variable,'Brand']=variable  #选取某个单元格
    dfgr.fillna(0,inplace=True)
    dfgr['gr_23']=dfgr[period23]/dfgr[period22]-1
    dfgr['pricegr_23']=(dfgr[period23]/dfgr[periodvol23])/(dfgr[period22]/dfgr[periodvol22])-1
    dfgr[price23]=dfgr[period23]/dfgr[periodvol23]
    dfgr=dfgr[['price_label23','gr_23',price23,'pricegr_23']] #选取多个列 period22,period23: sales value
    dfgr.loc[variable,'price_label23']=variable
    # st.write(df2)
    # st.dataframe(dfgr.round(3))

    # st.dataframe(dfgr['ytd_price_label23'].values.tolist()[:-1])
    bshshare=pd.DataFrame({})
    for aaa in dfgr['price_label23'].values.tolist():
        # st.write(aaa)
        if aaa=='Total':
            df3=temp
        else:
            df3=temp
            df3=df3[df3['price_label23']==aaa]

        # st.write(period23)
        df2=df3.groupby('Brand').agg({period22: 'sum', period23: 'sum'}) #,'ytdval22': 'sum', 'ytdval23': 'sum'
        # st.write(df2)
        dfshrbsh=df2.apply(lambda x:x/(x.sum()),axis=0)
        dfshrbsh.columns=[period22+'_bshshr',period23+'_bshshr']
        dfshrbsh.reset_index(inplace=True)
        # dfshrbsh.loc[variable]=dfshrbsh.sum(axis=0)
        # dfshrbsh.loc[variable,'Brand']=variable
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
    # bshshare.loc[variable]=bshshare.sum(axis=0)
    # bshshare.loc[variable,'Brand']=variable  #选取某个单元格
    bshshare=bshshare.rename(columns={'Brand':'price_label23'})

    output1=pd.merge(dfshr,dfgr,on='price_label23',how='outer')
    output=pd.merge(output1,bshshare,on='price_label23',how='outer')
    # st.dataframe(output)
    # break
    return(output)

st.caption(':blue[不同变量 x 价格的 where to play table]')
chose = st.radio('请选择需要分析的变量？', ['Type','caplabel','widthlabel','depthlabel','DW Subcategory','Setslabel','Waterlabel'],index=0,horizontal=True,key='variables')
submark=['Total']+df1[chose].drop_duplicates(inplace=False).values.tolist()
variables=st.multiselect('请选择一个你要显示的细分市场:',
                        (submark),
                        (['Total']))

pricetables=pd.DataFrame({})

for variable in variables:
    temp=pricetable(df1,sub_channel,chose,variable,period23,period22,price23,price22)
    pricetables=pd.concat([pricetables,temp],axis=0)

# st.write(pricetables)
mid=pricetables['mprice23']   #取备注列的值
pricetables.pop('mprice23')  #删除备注列
pricetables.insert(8,'mprice23',mid) #插入备注列

mid=pricetables['pricegr_23']   #取备注列的值
pricetables.pop('pricegr_23')  #删除备注列
pricetables.insert(8,'pricegr_23',mid) #插入备注列
st.dataframe(pricetables,use_container_width=True)

@st.cache_data 
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
st.header(':blue[不同变量之间的交叉分析]')

chosex1 = st.radio('请选择交叉分析的变量1（行）？', ['Type','price_label23','caplabel','widthlabel','depthlabel','DW Subcategory','Setslabel','Waterlabel'],index=0,horizontal=True,key='variablesx1')
chosex2 = st.radio('请选择交叉分析的变量2（列）？', ['Type','price_label23','caplabel','widthlabel','depthlabel','DW Subcategory','Setslabel','Waterlabel'],index=1,horizontal=True,key='variablesx2')

st.write('选择的行是:',chosex1,'  选择的列是:',chosex2)

tablex23=pd.pivot_table (df1,index=chosex1,columns=chosex2,values=period23,aggfunc='sum',margins=True)
impx23=tablex23.tail(1)
impx23=impx23.apply(lambda x: x/impx23.iloc[-1,-1])
impx23.rename(index={'All':'impx23'},inplace=True)
# st.write(impx23)

tablex22=pd.pivot_table (df1,index=chosex1,columns=chosex2,values=period22,aggfunc='sum',margins=True)
impx_gr23=((tablex23.tail(1)/tablex22.tail(1)-1)*100).round(1)
impx_gr23.rename(index={'All':'imp_gr23'},inplace=True)
# st.write(impx_gr23)

tablex23=tablex23.iloc[:-1,:]
tablex23=tablex23.apply(lambda x: x/x.sum(axis=0))

tablex22=tablex22.iloc[:-1,:]
tablex22=tablex22.apply(lambda x: x/x.sum(axis=0))

tablex=pd.concat([impx23,impx_gr23,tablex23],axis=0)
st.caption(':blue[带有importance,growth rate的交叉表]')
st.write(tablex)

@st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(tablex)

st.download_button(
    label="Download 交叉表 as CSV",
    data=csv,
    file_name='交叉表.csv',
    mime='text/csv',
)

tablex23.columns=[i+'_23'for i in tablex23.columns.values.tolist()]
tablex22.columns=[i+'_22'for i in tablex22.columns.values.tolist()]

tablex_impx=pd.concat([tablex23,tablex22],axis=1)
# st.write(tablex_impx)

colimpx=[]
for col22, col23 in zip(tablex22.columns.values.tolist(),tablex23.columns.values.tolist()):
    colimpx.append(col22)
    colimpx.append(col23)
tablex_impx=tablex_impx[colimpx]
st.caption(':blue[根据列显示出来的两年的对比数据]')
st.write(tablex_impx.round(3))

@st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(tablex_impx.round(3))

st.download_button(
    label="Download 交叉表两年的对比数据 as CSV",
    data=csv,
    file_name='交叉表两年的对比数据.csv',
    mime='text/csv',
)


st.write('-----------------------------------')
st.header(':blue[含有所有标签的底表]')
st.write(df1)


@st.cache_data 
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df1)

st.download_button(
    label="Download 含有所有标签的底表 as CSV",
    data=csv,
    file_name='含有所有标签的底表.csv',
    mime='text/csv',
)


st.write('--------------------------------------------------------------------')
st.header(':blue[品牌价格分析 - 每个品牌，每个价格带的占比及总份额]')
for chose in channel:
    st.write('原始品牌表-',chose)
    dftemp=df1
    if chose=='Total':
        dftemp=df1
    elif chose=='offline' or chose=='online'or chose=='Online'or chose=='Offline':
        dftemp=df1[df1['channel']==chose]
    else:
        dftemp=df1[df1['sub_channel']==chose]

    dftemp['share22']=dftemp['ytdval22'].apply(lambda x:x/dftemp['ytdval22'].sum(axis=0)*100)
    dftemp['share23']=dftemp['ytdval23'].apply(lambda x:x/dftemp['ytdval23'].sum(axis=0)*100)
    dftemp=dftemp[['channel','sub_channel','Brand','Model','Type','Launch date','Capacity(L)/Loading KG','Width(cm)','Depth(cm)','Height(cm)','ytdvol22','ytdvol23','ytdval22','ytdval23','mprice22','mprice23','ytdprice22','ytdprice23','share22','share23']]

    dftemp['ytd_price_label23']=dftemp['ytdprice23'].apply(lambda x : '1]<=3k' if x<=3000 else
                                                            ('2]3k <= 4k' if x>3000 and x<=4000 else
                                                            ('3]4k <= 5k' if x>4000 and x<=5000 else                                                         
                                                            ('4]5k <= 6k' if x>5000 and x<=6000 else
                                                            ('5]6k <= 7k' if x>6000 and x<=7000 else
                                                            ('6]7k <= 8k' if x>7000 and x<=8000 else                                                         
                                                            ('7]8k <= 9k' if x>8000 and x<=9000 else
                                                            ('8]9k <= 10k' if x>9000 and x<=10000 else
                                                            ('90]10k <= 11k' if x>10000 and x<=11000 else                                                        
                                                            ('91]11k <= 12k' if x>11000 and x<=12000 else
                                                            ('92]12k <= 13k' if x>12000 and x<=13000 else
                                                            ('93]13k <= 14k' if x>13000 and x<=14000 else
                                                            ('94]14k <= 15k' if x>14000 and x<=15000 else
                                                            ('95]15k <= 16k' if x>15000 and x<=16000 else                                                       
                                                            ('96]16k <= 17k' if x>16000 and x<=17000 else
                                                            ('97]17k <= 18k' if x>17000 and x<=18000 else
                                                            ('98]18k <= 19k' if x>18000 and x<=19000 else
                                                            ('99]19k <= 20k' if x>19000 and x<=20000 else
                                                            ('999]>=20k' if x>20000 else '')))))))))))))))))))    
    # dftemp['importance']=dftemp['share23']=dftemp['ytdval23'].apply(lambda x:x/dftemp['ytdval23'].sum(axis=0)*100)
    st.write(dftemp)
    # st.write(dftemp['Brand'])

    # st.cache_data  #@st.cache_data 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(dftemp)
    st.download_button(
        label="Download masterfile_KPI as CSV",
        data=csv,
        file_name='masterfile_KPI.csv',
        mime='text/csv',
        key='masterfile_KPI'+chose)


    brands=st.multiselect('请选择一个你要做价格对比分析的品牌:',
                    (dftemp['Brand']), 
                    (dftemp['Brand'].iloc[:1]))

    dfbrand=pd.DataFrame({})
    aveprice=[]
    for brand in brands:
        temp=dftemp[dftemp['Brand']==brand]
        temp['importance']=temp['ytdval23'].apply(lambda x:x/temp['ytdval23'].sum(axis=0)*100).round(2)
        temp['aveprice']=(temp['ytdval23'].sum(axis=0)/temp['ytdvol23'].sum(axis=0)).round(2)
        dfbrand=pd.concat([dfbrand,temp],axis=0)
    
    st.subheader(':blue[品牌和价格标签的SKU底表]')
    st.write(dfbrand)


    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(dfbrand)
    st.download_button(
        label="Download masterfile_keybrand as CSV",
        data=csv,
        file_name='masterfile_keybrand.csv',
        mime='text/csv',
        key='masterfile_keybrand'+chose)


    tablebrand =pd.pivot_table(dfbrand,index='ytd_price_label23',columns='Brand',values='importance',aggfunc=sum)
    tableprice =pd.pivot_table(dfbrand,columns='Brand',values='aveprice',aggfunc='mean')
    tableshare=pd.pivot_table(dfbrand,columns='Brand',values='share23',aggfunc=sum)

    tablebrand=pd.concat([tablebrand,tableprice,tableshare],axis=0)
    st.subheader(':blue[品牌x价格带(列相加100%)]')
    st.write(tablebrand)


    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    csv = convert_df(tablebrand)
    st.download_button(
        label="Download masterfile_tablebrand as CSV",
        data=csv,
        file_name='masterfile_tablebrand.csv',
        mime='text/csv',
        key='masterfile_tablebrand'+chose)