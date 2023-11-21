import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import time
import plotly.express as px

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

if ('Capacity(L) /Loading KG' not in colstemp) and ('Capacity(L)' not in colstemp):
    df['Capacity(L) /Loading KG']=0
    mid=df['Capacity(L) /Loading KG']
    df.pop('Capacity(L) /Loading KG')  #删除备注列
    df.insert(4,'Capacity(L) /Loading KG',mid) #插入备注列   

colstemp=df.columns.values.tolist()
colstemp=[i.replace("Channel",'channel') for i in colstemp]
colstemp=[i.replace("Sub_channel",'sub_channel') for i in colstemp]
colstemp=[i.replace("Groupbrand",'Brand Group') for i in colstemp]
colstemp=[i.replace("Width(mm)",'Width(cm)') for i in colstemp]
colstemp=[i.replace("Depth(mm)",'Depth(cm)') for i in colstemp]
colstemp=[i.replace("Height(mm)",'Height(cm)') for i in colstemp]
colstemp = [i.replace("Capacity(L)", "Capacity(L) /Loading KG") if i.count("Capacity(L) /Loading KG") == False else i for i in colstemp]
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
else:
    period23=period[0]

# period23='ytdval23'
period22=period23.replace('23','22')

price23='mprice23'
price22=price23.replace('23','22')  #设置选择后的Price period 对应到main table 里面的mprice 列



st.header(':blue[     查看主表的内容]')
st.write('(time period: ', period23,' 主表df中已经包括 ytdvol22,23, ytdval 22,23, mprice22,23, ytdprice 22,23)')
st.dataframe(df1, use_container_width=True)

st.write('--------------------------------------------------------------------')
st.header(':blue[品牌价格分析 - 每个品牌，每个价格带的占比及总份额]')
for chose in channel:
    st.write('原始品牌表-',chose)
    dftemp=df1
    if chose=='Total':
        dftemp=df1
    elif chose=='Offline' or chose=='Online':
        dftemp=df1[df1['channel']==chose]
    else:
        dftemp=df1[df1['sub_channel']==chose]

    dftemp['share22']=dftemp['ytdval22'].apply(lambda x:x/dftemp['ytdval22'].sum(axis=0)*100)
    dftemp['share23']=dftemp['ytdval23'].apply(lambda x:x/dftemp['ytdval23'].sum(axis=0)*100)
    dftemp=dftemp[['channel','sub_channel','Brand','Model','Type','Launch date','Capacity(L) /Loading KG','Width(cm)','Depth(cm)','Height(cm)','ytdvol22','ytdvol23','ytdval22','ytdval23','mprice22','mprice23','ytdprice22','ytdprice23','share22','share23']]

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


st.write('--------------------------------------------------------------------')
st.header(':blue[根据市场特征，制作各个变量对应的标签]')  #, divider='rainbow'


for temp, col in zip(['dftype','dfcap','dfwidth','dfdepth','dfSubcate','dfWater','dfSets'],['Type','Capacity(L) /Loading KG','Width(cm)','Depth(cm)','DW Subcategory','Water consumption','Sets']):

    var[temp]=df1.groupby(col).agg({period23: 'sum'})
    var[temp]['cumsum']=var[temp][period23].apply(lambda x:x/var[temp][period23].sum()).cumsum()
    # st.dataframe(temp)

dftype1,dfprice1,dfcap1,dfwidth1,dfdepth1,dfSubcate1,dfWater1,dfSets1=st.tabs(['Type','price','Capacity(L) /Loading KG','Width(cm)','Depth(cm)','DW Subcategory(DC only)','Water consumption(DC only)','Sets(DC only)'])

with dftype1:
    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dftype1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write('Type中要分析的渠道: ',chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online':
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
        
        col1, col2, col3 = st.columns([4,1,2])
        with col1:
            # fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        # with col2:
        with col3:
            st.dataframe(group)   #1)0 < 200   2)200 < 300   3)300 < 400    4)400 < 500   5)500 < 600  6)600 < 600000  unknown


        groupshr=pd.pivot_table(dftemp,index='Brand',columns='Type',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


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
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        # st.write(groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化

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

        import plotly.graph_objs as go
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

        # st.plotly_chart(fig, use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr_chg_short.columns)),
                                cells=dict(values=[groupshr_chg_short.index] + [groupshr_chg_short[col] for col in groupshr_chg_short.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        st.subheader(':blue[这个维度下的 size, growth & share]')
        st.write('channel= ', chose)
        st.dataframe(groupshr2, use_container_width=True)


        # table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
        #                     cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
        #                                 height=35, align=['left', 'center'],
        #                                 #  fill_color='lightgrey'
        #                                 ))
        # layout = go.Layout(title="size, growth & share_"+ chose,height=500)
        # fig = go.Figure(data=[table_data],layout=layout)
        # st.plotly_chart(fig)


        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr_chg_short)
        st.download_button(
            label="Download groupshr_chg_short_Capacity(L) as CSV",
            data=csv,
            file_name='groupshr_chg_short_Capacity(L).csv',
            mime='text/csv',
            key='unique_key'+'dftype groupshr_chg_short'+chose)
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_Capacity(L) as CSV",
            data=csv,
            file_name='groupshr2_Capacity(L).csv',
            mime='text/csv',
            key='unique_key'+'dftype'+chose)

with dfprice1:
    df1['mprice22']=df1[valmonth22]/df1[volmonth22]  #单月价格
    df1['mprice23']=df1[valmonth23]/df1[volmonth23]  #单月价格

    df1['ytdprice22']=df1['ytdval22']/df1['ytdvol22'] #单月价格
    df1['ytdprice23']=df1['ytdval23']/df1['ytdvol23'] #单月价格
    df1.fillna(0,inplace=True)


    dfprice=df1.groupby('ytdprice23').agg({period23: 'sum'})
    dfprice['cumsum']=dfprice[period23].apply(lambda x:x/dfprice[period23].sum()).cumsum()
    dfprice.reset_index(drop=False,inplace=True)
    dfprice['ytdprice23']=dfprice['ytdprice23'].round(0)
    # st.write(dfprice)

    plotlychart, col1= st.columns([2,2])
    with plotlychart:
        fig = px.scatter(dfprice,x='ytdprice23',y='cumsum',) # color="continent",
        st.plotly_chart(fig,  use_container_width=True)
    with col1:
        st.dataframe(dfprice, use_container_width=True)

    
    #制作 YTD 价格标签
    chose = st.radio('是否需要选择如何对价格带分组（默认:1)<3000, 2)3000 < 5000, 3)5000 < 7000, 4)7000 < 10000, 5)10000 < 15000, 6)>= 15000)？', ['Yes','No'], index=1,key='unique_key'+'dfprice')
    if chose=='No':
        df1['ytd_price_label23']=df1['ytdprice23'].apply(lambda x : '   1]<=3000' if x<=3000 else
                                                                ('   2]3000 <= 5000' if x>3000 and x<=5000 else
                                                                ('   3]5000 <= 7000' if x>5000 and x<=7000 else
                                                                ('   4]7000 <= 10000' if x>7000 and x<=10000 else
                                                                ('   5]10000 <= 15000' if x>10000 and x<=15000 else
                                                                ('   6]>= 15000' if x>15000 else ''))))))
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

    st.write('检查价格带分类是否正确')   
    st.dataframe(df1[['ytd_price_label23','ytdprice23']].head(5)) 

    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfprice1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        group=dftemp.groupby('ytd_price_label23').agg({period23: 'sum',period22: 'sum'})
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
            st.dataframe(dftemp[['ytd_price_label23','ytdprice23']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='ytd_price_label23',values=period23,aggfunc="sum",margins=True)
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

        growth=[0 if i==np.inf else i for i in growth]
        # st.write(importance, widths,growth)


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
            st.plotly_chart(fig, theme="streamlit",use_container_width=True)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr mekko table as CSV",
            data=csv,
            file_name='groupshr mekko table.csv',
            mime='text/csv',
            key='unique_key'+'groupshr mekko table'+chose)
        
        dftemp['share22']=dftemp[period22].apply(lambda x:x/dftemp[period22].sum(axis=0)*100)
        dftemp['share23']=dftemp[period23].apply(lambda x:x/dftemp[period23].sum(axis=0)*100)
        dftemp['shrchg23']=(dftemp['share23']-dftemp['share22']).round(2)
        shrchgtable=pd.pivot_table(dftemp,index='Brand',columns='ytd_price_label23',values='shrchg23',aggfunc=sum,margins=True)
        st.caption(':blue[份额变化 x 价格(乘过100)]')
        st.write(shrchgtable)

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(shrchgtable)
        st.download_button(
            label="Download shrchgtable as CSV",
            data=csv,
            file_name='shrchgtable.csv',
            mime='text/csv',
            key='unique_key'+'shrchgtable'+chose)
        
        st.subheader(':blue[SKU share x Brand x Pricetier 的分析]')
        chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗?', ['Yes','No'],index=0,horizontal=True, key='sku share x price'+ chose)

        if chose=='Yes':
            col1, col2 =st.columns([1,1])
            with col1:
                pricetiers=st.multiselect('请选择一个你要分析的价格段:',
                                (dftemp['ytd_price_label23'].drop_duplicates().tolist()))
            with col2:
                brands=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                                (dftemp.Brand.tolist()))

            st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')

            for pricetier in pricetiers:
                for brand in brands:                
                    st.write(pricetier + ' ------- ' + brand)
                    # st.write(df1[df1['Brand']==brand])
                        
                    dfbrandprice=dftemp[(dftemp['Brand']==brand) & (dftemp['ytd_price_label23']==pricetier)][['Brand','Model','Width(cm)','Depth(cm)',"Capacity(L) /Loading KG",period23,period22,'ytd_price_label23','shrchg23']]
                    dfbrandprice.loc['Total']=dfbrandprice.apply(lambda x:x.sum())
                    dfbrandprice.loc['Total','Brand']='Total'
                    dfbrandprice.loc['Total','Model']='Total'
                    dfbrandprice.sort_values(by=['shrchg23'],ascending=False,inplace=True)
                    dfbrandprice['url']='https://www.baidu.com/s?wd=' + dfbrandprice['Model']
                    st.write(dfbrandprice)        
        else:
            pass

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

    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfcap1')
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online':
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
            st.dataframe(dftemp[['caplabel','Capacity(L) /Loading KG']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='caplabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        # st.dataframe(groupshr.columns.values.tolist())


        import plotly.graph_objects as go
        import numpy as np

        importance=groupshr.columns.values.tolist()
        # st.write(group.index=='unknown')
        group=group.drop(group[group.index=='unknown'].index)
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

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_Capacity(L) as CSV",
            data=csv,
            file_name='groupshr2_Capacity(L).csv',
            mime='text/csv',
            key='unique_key'+'dfcap1'+chose)

with dfwidth1:
    col1, plotlychart = st.columns([2,2])
    with col1:
        st.dataframe(dfwidth, use_container_width=True)
    with plotlychart:
        fig = px.bar(dfwidth,x=dfwidth.index.values.tolist(),y='cumsum',) # color="continent",
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    chose = st.radio('是否需要选择如何分组（默认:<200, 200<300, 300<400, 400<500, 500<600, >600)？', ['Yes','No'], index=1,key='unique_key'+'Width(cm)')

    if chose=='No':
        df1['widthlabel']=df1['Width(cm)'].apply(lambda x: 'unknown' if x=='unknown' else(
                                                                       '1)<50' if x<50 else(
                                                                       '2)50<60' if x>=50 and x<60 else(
                                                                       '3)60<70' if x>=60 and x<70 else(
                                                                       '4)70<80' if x>=70 and x<80 else(
                                                                       '5)80<90' if x>=80 and x<90 else(
                                                                       '6)>=90' if x>=90 else np.nan)))))))
    else:
        number=st.number_input('请输入你要分成几个Capacity(L) /Loading KG段:',key='number Width(cm)')

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
        elif chose=='offline' or chose=='online':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]

        
        group=dftemp.groupby('widthlabel').agg({period23: 'sum',period22: 'sum'})
        group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum()).round(3)
        st.write(group['cumsum'])
        group['gr_23']=(group[period23]/group[period22]-1)*100
        group=group.drop(group[group.index=='unknown'].index)
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
            st.dataframe(dftemp[['widthlabel','Width(cm)']])

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='widthlabel',values=period23,aggfunc="sum",margins=True)
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)
        st.write(groupshr)
        # st.dataframe(groupshr.columns.values.tolist())


        import plotly.graph_objects as go
        import numpy as np

        importance=groupshr.columns.values.tolist()
        growth=group['gr_23'].round(3).values.tolist()
        # st.write(growth)

        widths = np.array([i*100 for i in group['cumsum'].values.tolist()])
        # st.write(widths,np.cumsum(widths)-widths)
        growth=[0 if i==np.inf else i for i in growth]
        st.write(importance, widths,growth)


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

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_Width(cm) as CSV",
            data=csv,
            file_name='groupshr2_Width(cm).csv',
            mime='text/csv',
            key='unique_key'+'dfwidth1'+chose)

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
        elif chose=='offline' or chose=='online':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]


        group= dftemp.groupby('depthlabel').agg({period23: 'sum',period22: 'sum'}) # 改这里'depthlabel'
        group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
        group['gr_23']=(group[period23]/group[period22]-1)*100
        group=group.drop(group[group.index=='na'].index)
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
            st.dataframe( dftemp[['depthlabel','Depth(cm)']]) # 改这里'depthlabel'   'Depth(cm)'

        groupshr=pd.pivot_table( dftemp,index='Brand',columns='depthlabel',values=period23,aggfunc="sum",margins=True) # 改这里'depthlabel'
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
            # ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
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

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_Depth(cm) as CSV",
            data=csv,
            file_name='groupshr2_Depth(cm).csv',
            mime='text/csv',
            key='unique_key'+'dfdepth1'+chose)

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
        elif chose=='offline' or chose=='online':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]


        group= dftemp.groupby('Waterlabel').agg({period23: 'sum',period22: 'sum'}) # 改这里'depthlabel'
        group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
        group['gr_23']=(group[period23]/group[period22]-1)*100
        group=group.drop(group[group.index=='na'].index)
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
            st.dataframe( dftemp[['Waterlabel','Water consumption']]) # 改这里'depthlabel'   'Depth(cm)'

        groupshr=pd.pivot_table( dftemp,index='Brand',columns='Waterlabel',values=period23,aggfunc="sum",margins=True) # 改这里'depthlabel'
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
            # ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
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

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_dfWater as CSV",
            data=csv,
            file_name='groupshr2_dfWater.csv',
            mime='text/csv',
            key='unique_key'+'dfWater'+chose)

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
        elif chose=='offline' or chose=='online':
            dftemp=df1[df1['channel']==chose]
        else:
            dftemp=df1[df1['sub_channel']==chose]


        group= dftemp.groupby('Setslabel').agg({period23: 'sum',period22: 'sum'}) # 改这里'depthlabel'
        group['cumsum']=group[period23].apply(lambda x:x/group[period23].sum())
        group['gr_23']=(group[period23]/group[period22]-1)*100
        group=group.drop(group[group.index=='na'].index)
        group=group.round(5) 

        fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            # fig = px.bar(group,x=group.index.values.tolist(),y='cumsum',text='cumsum') # color="continent",
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with col2:
            st.dataframe(group)   #1)0 < 200   2)200 < 300   3)300 < 400    4)400 < 500   5)500 < 600  6)600 < 600000  unknown
        with col3:
            st.dataframe( dftemp[['Setslabel','Sets']]) # 改这里'depthlabel'   'Depth(cm)'

        groupshr=pd.pivot_table( dftemp,index='Brand',columns='Setslabel',values=period23,aggfunc="sum",margins=True) # 改这里'depthlabel'
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

        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_Sets as CSV",
            data=csv,
            file_name='groupshr2_Sets.csv',
            mime='text/csv',
            key='unique_key'+'dfSets1'+chose)

with dfSubcate1:
    choses = st.multiselect('请选择一个你要分析的渠道:',(channels), (['Total']),key='unique_key'+'dfSets1: ') #改这里 dfSets1: 
    # chose = st.radio('请选择你要分析的渠道', channels, index=1,key='unique_key'+'dftype1')   
    for chose in choses:
        st.write(chose)
        dftemp=df1
        if chose=='Total':
            dftemp=df1
        elif chose=='offline' or chose=='online':
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
                
        group=dftemp.groupby('DW Subcategory').agg({period23: 'sum',period22: 'sum'}) #改这里 'Type'
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
            st.dataframe(dftemp[['DW Subcategory']])  #改这里 'Type'

        groupshr=pd.pivot_table(dftemp,index='Brand',columns='DW Subcategory',values=period23,aggfunc="sum",margins=True) #改这里 'Type'
        groupshr=groupshr.iloc[:-1,:]
        groupshr=groupshr.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr23=groupshr
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)


        groupshr22=pd.pivot_table(dftemp,index='Brand',columns='DW Subcategory',values=period22,aggfunc="sum",margins=True) #改这里 'Type'
        groupshr22=groupshr22.iloc[:-1,:]
        groupshr22=groupshr22.apply(lambda x: x/x.sum()*100).sort_values(by='All',ascending=False)
        groupshr22=groupshr22.fillna(0)

        cols=[str(i) + '_22' for i in groupshr22.columns.values.tolist()]
        colshrchg=[str(i) + '_shrchg' for i in groupshr22.columns.values.tolist()]
        groupshr22.columns=cols
        groupshr_chg=pd.merge(groupshr22,groupshr23,left_index=True, right_index=True)

        lens=len(groupshr22.columns.values.tolist())
        for col,col22,col23 in zip(colshrchg,groupshr22.columns.values.tolist(),groupshr23.columns.values.tolist()):
            groupshr_chg[col]=groupshr_chg[col23] - groupshr_chg[col22]
        groupshr_chg_short=groupshr_chg[colshrchg].round(2)
        # st.write(groupshr_chg_short)

        groupshr=groupshr_chg[groupshr23.columns.values.tolist()]
        groupshr=groupshr.iloc[:6,:]
        groupshr.loc['other']=groupshr.apply(lambda x: 100-x.sum())
        groupshr=groupshr.iloc[:,:-1].fillna(0).round(1)

        groupshr_chg_short=groupshr_chg_short.iloc[:6,:]
        groupshr_chg_short.loc['other']=groupshr_chg_short.apply(lambda x: 1-x.sum()).round(2)  #份额的变化

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
                texttemplate="%{y} ",           #x %{width} =<br>%{customdata[1]}
                textposition='outside',
                textangle=0,
                textfont_color="white",
                marker=dict(line=dict(color='rgb(256, 256, 256)',  # 设置边框线的颜色
                width=1.0  # 设置边框线的宽度            
            ))))

        fig.update_xaxes(
            tickvals=np.cumsum(widths)-widths/2,
            # ticktext= ["%s<br>%d<br>%d" % (w,i,g) for w,i,g in zip(importance,widths,growth)]
        )

        fig.update_xaxes(range=[0,100])
        fig.update_yaxes(range=[0,100])

        fig.update_layout(
            title_text="Mekko_"+ chose,
            barmode="stack",
            height=600
            # uniformtext=dict(mode="hide", minsize=10),
        )

        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        col1, col2 =st.columns([2,1])
        with col1:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with col2:

            table_data = go.Table(header=dict(values=['Total']+list(groupshr_chg_short.columns)),
                                cells=dict(values=[groupshr_chg_short.index] + [groupshr_chg_short[col] for col in groupshr_chg_short.columns],
                                            height=45, align=['left', 'center'],
                                            #  fill_color='lightgrey'
                                            ))
            layout = go.Layout(title="share change table_"+ chose,height=700)
            fig = go.Figure(data=[table_data],layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        group=group[['cumsum','gr_23']]
        groupshr2=pd.concat([group.T,groupshr],axis=0)
        # st.dataframe(groupshr2, use_container_width=True)
        table_data = go.Table(header=dict(values=['Total']+list(groupshr2.columns)),
                            cells=dict(values=[groupshr2.index] + [groupshr2[col] for col in groupshr2.columns],
                                        height=35, align=['left', 'center'],
                                        #  fill_color='lightgrey'
                                        ))
        layout = go.Layout(title="size, growth & share_"+ chose,height=600)
        fig = go.Figure(data=[table_data],layout=layout)
        st.plotly_chart(fig, use_container_width=True)


        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr_chg_short)
        st.download_button(
            label="Download groupshr_chg_short_DW Subcategory as CSV",  #改这里 'DW Subcategory'
            data=csv,
            file_name='groupshr_chg_short_DW Subcategory.csv',  #改这里 'DW Subcategory'
            mime='text/csv',
            key='unique_key'+'dfwater groupshr_chg_short'+chose)  #改这里 'DW Subcategory'
        
        @st.cache_data 
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(groupshr2)
        st.download_button(
            label="Download groupshr2_DW Subcategory as CSV",  #改这里 'DW Subcategory'
            data=csv,
            file_name='groupshr2_DW Subcategory.csv',  #改这里 'DW Subcategory'
            mime='text/csv',
            key='unique_key'+'DW Subcategory'+chose)  #改这里 'DW Subcategory'



st.write('--------------------------------------------------------------')
st.header(':blue[下面是where to play cross table]')

# @st.cache_data
def table(df1,sub_channel,period23,period22,price23,price22):
    if sub_channel=='Total':
        temp=df1
    elif sub_channel=='offline' or sub_channel=='online'or sub_channel=='Online'or sub_channel=='Offline':
        df1=df1[df1['channel']==sub_channel]
    else:
        df1=df1[df1['sub_channel']==sub_channel]

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
    output=output.round(4)
    # st.dataframe(output, use_container_width=True)

    chose = st.radio('需要选择出示哪些列吗(growth, share, share & price change)？', ['Yes','No'],index=1, key='unique_key'+sub_channel)

    if chose=='Yes':
        fact=st.multiselect('请选择一个你要显示的列(Brand and offline value must chose):',
                        (output.columns.tolist()))
        output=output[fact]
    else:
        pass

    output.sort_values(by=[sub_channel+'_'+ period23],ascending=False,inplace=True)
    st.dataframe(output,use_container_width=True)

    st.subheader(':blue[SKU share By brand 的分析]')
    chose = st.radio('需要选择哪些品牌进行SKU leve的计算吗 (默认+/- share >0.1%？', ['Yes','No'],index=1, key='sku share'+sub_channel)

    if chose=='Yes':
        fact=st.multiselect('请选择一个你要显示SKU SHARE 的品牌:',
                        (output.Brand.tolist()))
    else:
        fact=output[output[sub_channel+'_'+ '+/-shr23']>0.01].Brand.values.tolist()

    st.subheader(':blue[显示哪个品牌的哪个SKU的share在涨/跌]')
    sumval23=df1[period23].sum()
    sumval22=df1[period22].sum()

    for brand in fact:
        st.write(brand)
        # st.write(df1[df1['Brand']==brand])
            
        dfbrand=df1[df1['Brand']==brand][['Brand','Model','Width(cm)','Depth(cm)',"Capacity(L) /Loading KG",period23,period22]]
        dfbrand['shr_change']=( dfbrand[period23]/sumval23 - dfbrand[period22]/sumval22)
        dfbrand.loc['Total']=dfbrand.apply(lambda x:x.sum())
        dfbrand.loc['Total','Brand']='Total'
        dfbrand.loc['Total','Model']='Total'
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

st.header(':blue[不同变量 x 价格的 where to play table]')
chose = st.radio('请选择需要分析的变量？', ['Type','caplabel','widthlabel','depthlabel','DW Subcategory','Setslabel','Waterlabel'],index=0,horizontal=True,key='variables')
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

chosex1 = st.radio('请选择交叉分析的变量1（行）？', ['Type','caplabel','widthlabel','depthlabel','DW Subcategory','Setslabel','Waterlabel'],index=0,horizontal=True,key='variablesx1')
chosex2 = st.radio('请选择交叉分析的变量2（列）？', ['Type','caplabel','widthlabel','depthlabel','DW Subcategory','Setslabel','Waterlabel'],index=1,horizontal=True,key='variablesx2')

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