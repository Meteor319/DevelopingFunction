
### 當日融券賣出與借券賣出成交量值 
### Short Sales Volume and Value
"""
save_path = path
date = '20171030'
ShortSalesVolumnValue(save_path, date)
"""
def ShortSalesVolumnValue(save_path, date):
    import requests
    import pandas as pd
    # Path def
    #date = '20171027'
    url = 'http://www.tse.com.tw/exchangeReport/TWTASU?response=csv&date=' + date
    #print(url)
    
    # Request web
    head = {
      'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
      'Accept-Encoding':'gzip, deflate',
      'Accept-Language':'en-US,en;q=0.8',
      'Connection':'keep-alive',           
    	'Host':'www.tse.com.tw',
    	'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
      'Referer':'http://www.tse.com.tw/zh/page/trading/exchange/TWTASU.html',
      'Upgrade-Insecure-Requests':'1'}

    r = requests.get(url, headers = head)
    
    # Information check
    """
    print(r.headers)
    print(r.content)
    r.encoding
    r.cookies
    r.content
    """
    #print(r.content.decode('MS950'))
    k = r.content.decode('MS950')
    s = k.strip().split('\r')
    m = pd.DataFrame(s[2:(len(s)-3)])
    
    #
    year = s[0].replace('"','').split('年')[0]
    month = s[0].replace('"','').split('年')[1].split('月')[0]
    day = s[0].replace('"','').split('年')[1].split('月')[1].split('日')[0]
    File_date = [year,'-',month,'-',day]
    #
    """
    print(m.ix[13,0])
    m.ix[13,0].split('"')[1].replace('   ',' ').strip()
    m.ix[13,0].split('"')[3].replace(',','')
    m.ix[13,0].split('"')[5].replace(',','')
    m.ix[13,0].split('"')[7].replace(',','')
    m.ix[13,0].split('"')[9].replace(',','')
    """
    m['證券代號'] = m[0].map(lambda x :x.split('"')[1].replace('   ',' ').strip().split(' ')[0]) 
    m['證券名稱'] = m[0].map(lambda x :x.split('"')[1].replace('   ',' ').strip().split(' ')[1])
    m['融券賣出成交數量'] = m[0].map(lambda x :x.split('"')[3].replace(',',''))
    m['融券賣出成交金額'] = m[0].map(lambda x :x.split('"')[5].replace(',',''))
    m['借券賣出成交數量'] = m[0].map(lambda x :x.split('"')[7].replace(',',''))
    m['借券賣出成交金額'] = m[0].map(lambda x :x.split('"')[9].replace(',',''))
    m['File_date'] = ('').join(File_date)
    del m[0]
    #print(m.head(2))
    
    # Output file
    file_name = 'Short_Sales_Volume_Value' + '_' +('').join(File_date) + '.csv'
    tmp_path = save_path +file_name
    print(tmp_path)     
    m.to_csv(tmp_path, sep=',')
    print('Save 當日融券賣出與借券賣出成交量值 :',('').join(File_date))



#############
ww2 = '20171110'
ShortSalesVolumnValue(r'C:\Users\Meteor\Desktop\Stock\Rawdata\當日融券賣出與借券賣出成交量值\\',ww2)

#
"""
import datetime
import time
import random
ww = datetime.datetime.now() - datetime.timedelta(days = 1)
ww2 = ww.strftime('%Y%m%d')
print(ww2)
#
ww2 = '20171107'
ShortSalesVolumnValue(r'C:\Users\Meteor\Desktop\Stock\Rawdata\當日融券賣出與借券賣出成交量值\\',ww2)
#
for i in range(454, 1000):
    ww = datetime.datetime.now() - datetime.timedelta(days = i)
    query_date = ww.strftime('%Y%m%d')
    #print(query_date)
    try:
        ShortSalesVolumnValue(r'C:\Users\Meteor\Desktop\Stock\Rawdata\當日融券賣出與借券賣出成交量值\\',query_date)
        time.sleep(random.randint(5,9))
    except:
        print(i, '|', query_date)
        pass
"""