                                           #BG-NBD ve Gamma-Gamma ile CLTV Prediction



"""
                                          İŞ PROBLEMİ

FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete 
sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.
"""



                                           # Veri Seti Hikayesi
# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline
# alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


#Öncelikle kütüphaneleri import edelim ve veri setimizi çağıralım.

import seaborn as sns
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt
import matplotlib
from lifetimes.plotting import plot_period_transactions
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

df_ = pd.read_csv(r"C:\Users\sevim\Desktop\MIUUL\HAFTA 3\CASE STUDY-2\FLOCLTVPrediction\flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum() #masterid unique o sebeple gruplandırmamıza ve tekilleştirmemize gerek yok.




"""2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız. """

#outlier_thresholds fonksiyonu tamamlayacağım. Bu fonksiyon ile birlikte aykırı değer baskılama gerçekleştireceğim.
def outlier_thresholds(dataframe, variable):
    quartile1=dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3-quartile1
    up_limit =quartile3 + 1.5*interquantile_range
    low_limit = quartile1-1.5*interquantile_range
    return low_limit,up_limit

#replace fonksiyonu ile aykırı değerleri baskıladım ve belirlediğim alt ve üst limitlere eşitledim.
#Aykırı değerleri tıraşlama fonksiyonu

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

""" "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" 
değişkenlerinin aykırı değerleri varsa baskılayanız."""

columns = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)

df.describe().T


"""Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
alışveriş sayısı ve harcaması için yeni değişkenler oluşturun."""


df["total_order"]= df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
df.head()


""" 5.Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
#date içeren değişkenleri seçmem lazım. """
df.columns
df.index
df.info()
df.dtypes


for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
df.dtypes

                         # GÖREV 2: CLTV Veri Yapısının Oluşturulması


# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()  #Timestamp('2021-05-30 00:00:00')
today_date= dt.datetime(2021,6,2)
type(today_date)  #datetime.datetime


"""2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv 
dataframe'i oluşturunuz.Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık 
cinsten ifade edilecek.buradaki değerler rfm hesaplamaktan daha farklıdır. Burada toplam olarak aldıgımız değerleri 
average olarak alırız."""
#customer_id unique değer olan master_id değişkenine eşit olacak.
#burada recency kavramı RFMden daha farklı. Müşterinin ilk ve son alışveriş tarihi farkına eşit olacak.
#T_weekly müşteri yaşını ifade eder.Bugünden müşterinin sisteme ilk katılıp alışveriş yaptığı tarih farkını alırız.
#frequency direkt olarak müşteri frekansına eşit olacaktır.
#monetary değeri ise total müşteri değerinin ortalaması anlamına gelmektedir. Frekans değerine böleceğiz.



cltv_df=pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency"] = (df["last_order_date"]- df["first_order_date"]).dt.days
cltv_df["T"]= (today_date-df["first_order_date"]).dt.days
cltv_df["frequency"]=df["total_order"]
cltv_df["monetary"]=df["total_price"]
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] ##average
#Frekans değeri 1den büyük olan müşterilerle ilgileniyoruz. Bunlar bizden en az 1 kez alışveriş yapan kitleyi ifade eder.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
#recency ve T değerleri günlük değerlerdir. Haftaya çevireceğiz.
cltv_df["recency_cltv_weekly"] = cltv_df["recency"] / 7

cltv_df["T_weekly"] = cltv_df["T"] / 7
cltv_df.head()

#Şuan cltv_df Dataframeini oluşturduk.
cltv_df.index
cltv_df.columns  #(['customer_id', 'recency', 'T', 'frequency', 'monetary','recency_cltv_weekly', 'T_weekly'])

                       # GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması

# 1. BG/NBD modelini fit ediniz.
#BG-NBD modelinde kullanılacak değişkenler frequency, recency,T değerleridir.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

 # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

#Biz değişkenlerimizi hafta olarak hesaplamıştık. O yüzden ay cinsine çevirmemiz gerekmektedir.
#BG/NBD modelinde predict kullanabiliriz fakat Gamma Gammada bu metodu kullanamıyoruz.


bgf.predict(4*3,cltv_df['frequency'],cltv_df['recency_cltv_weekly'],cltv_df['T_weekly']).sort_values(ascending=False).head(10)

#BG-NBD işlemimizi kalıcı olması için yeni bir değişkene atadık.
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])


cltv_df.sort_values("exp_sales_3_month",ascending=False).head(10)
#head atarak ilk 10 müşteriyi elde ettik.

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])


cltv_df.sort_values("exp_sales_6_month",ascending=False).head(10)

###tahmin değerleri ile gercek değerleri kontrol edelim.

plot_period_transactions(bgf)
plt.show(block=True)


#Gamma Gamma modelini fit ediniz.Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.
#Gamma Gamma modelinde kullanılacak değişkenler frequency, monetary değerleridir.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)


cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

cltv_df.sort_values("exp_average_value", ascending=False).head(10)

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv
cltv_df.head()
cltv.sort_values(by="clv", ascending=False).head(10)
cltv = cltv.reset_index()
cltv.head()


                   # GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması

# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
#qcut küçükten büğüğe sıralama yapar. O sebeple sıralamaya D,C,B,A ile başlarız.
cltv["segment"] = pd.qcut(cltv["clv"], 4, labels=["D", "C", "B", "A"])

cltv.sort_values("clv", ascending=False).head(20)

cltv["segment"].value_counts() ##qcut eşit şekilde segmentlere ayırır.

cltv.groupby("segment").agg({"clv": ["mean", "min", "max"]})




