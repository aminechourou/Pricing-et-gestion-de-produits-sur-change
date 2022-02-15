from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import streamlit as st
import pandas_datareader.data as web
import io
import requests
import requests_cache
import time





def d1(S,K,T,r,re,sigma):
    return(log(S/K)+(r-re+sigma**2/2.)*T)/(sigma*sqrt(T))
def d2(S,K,T,r,re,sigma):
    return d1(S,K,T,r,re,sigma)-sigma*sqrt(T)
def bs_call(S,K,T,r,re,sigma):
    return S*exp(-re*T)*norm.cdf(d1(S,K,T,r,re,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,re,sigma))
  
def bs_put(S,K,T,r,re,sigma):
    return -S*exp(-re*T)*norm.cdf(-d1(S,K,T,r,re,sigma))+K*exp(-r*T)*norm.cdf(-d2(S,K,T,r,re,sigma))

def call_implied_volatility(Price, S, K, T, r,re):
    sigma = 0.0001
    while sigma < 1:
        Price_implied = S*exp(-re*T)*norm.cdf(d1(S,K,T,r,re,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,re,sigma))
        if Price-(Price_implied) < 0.001:
            return sigma
        sigma += 0.001
    return "Not Found"

def put_implied_volatility(Price, S, K, T, r,re):
    sigma = 0.0001
    while sigma < 1:
        Price_implied = -S*exp(-re*T)*norm.cdf(-d1(S,K,T,r,re,sigma))+K*exp(-r*T)*norm.cdf(-d2(S,K,T,r,re,sigma))
        if Price-(Price_implied) < 0.001:
            return sigma
        sigma += 0.001
    return "Not Found"
def call_delta(S,K,T,r,re,sigma):
    return norm.cdf(d1(S,K,T,r,re,sigma))
def call_gamma(S,K,T,r,re,sigma):
    return norm.pdf(d1(S,K,T,r,re,sigma))/(S*sigma*sqrt(T))
def call_vega(S,K,T,r,re,sigma):
    return 0.01*(S*norm.pdf(d1(S,K,T,r,re,sigma))*sqrt(T))
def call_theta(S,K,T,r,re,sigma):
    return 0.01*(-(S*norm.pdf(d1(S,K,T,r,re,sigma))*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2(S,K,T,r,re,sigma)))
def call_rho(S,K,T,r,re,sigma):
    return 0.01*(K*T*exp(-r*T)*norm.cdf(d2(S,K,T,r,re,sigma)))

   
def put_delta(S,K,T,r,re,sigma):
    return -norm.cdf(-d1(S,K,T,r,re,sigma))
def put_gamma(S,K,T,r,re,sigma):
    return norm.pdf(d1(S,K,T,r,re,sigma))/(S*sigma*sqrt(T))
def put_vega(S,K,T,r,re,sigma):
    return 0.01*(S*norm.pdf(d1(S,K,T,r,re,sigma))*sqrt(T))
def put_theta(S,K,T,r,re,sigma):
    return 0.01*(-(S*norm.pdf(d1(S,K,T,r,re,sigma))*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2(S,K,T,r,re,sigma)))
def put_rho(S,K,T,r,re,sigma):
    return 0.01*(-K*T*exp(-r*T)*norm.cdf(-d2(S,K,T,r,re,sigma)))



def home_page() -> None:
    #st.title('Looking into Iris Dataset')
    #st.image('all_three.jpg', caption='Three types of Iris flowers.')
    st.title(' Taux de change')
    #st.write(expiry1)
    #stock = 'EURUSD=X'
    #expiry = '12-10-2023'
    #if(strike_price == None):
     #   strike_price = 1.13


    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite')

    # just add headers to your session and provide it to the reader
    session.headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0',     'Accept': 'application/json;charset=utf-8'     }


    today = datetime.now()
    #st.write( str(today.replace(day=today.day+3)))
    one_year_ago = today.replace(year=today.year-1)

    df = web.DataReader(stock, 'yahoo', one_year_ago, today,  session=session)
    
    with st.expander('Voir Tous les données '):
        st.write(df)
    #st.header('General Information')
    df = df.sort_values(by="Date")
    df = df.dropna()
    df = df.assign(close_day_before=df.Close.shift(1))
    df['returns'] = ((df.Close - df.close_day_before)/df.close_day_before)
    #normal_return.plot(legend = False , figsize=(12,6))
    #selected_sepecies = st.radio('Select species', ['Setosa', 'Versicolor', 'Virginica'])
    #show_description(selected_sepecies)
    
    sigma = np.sqrt(252) * df['returns'].std()
    #uty = (web.DataReader(
     #   "^TNX", 'yahoo', today.replace(day=today.day-1), today,  session=session)['Close'].iloc[-1])/100

    lcp = df['Close'].iloc[-1]
    #st.write(expiry)
    #st.write(datetime.utcnow())
    #st.write(datetime.strptime(expiry, "%m-%d-%Y"))
    
    t = (datetime.strptime(str(expiry.strftime("%m-%d-%Y")), "%m-%d-%Y") - datetime.utcnow()).days / 365
    st.write(t)
    col1, col2 = st.columns(2)
    col1.header('Call')
    col1.write('Le prix de l option call : '+str( bs_call(lcp, strike_price, t, uty,re, sigma) ), unsafe_allow_html=True)
    col2.header('Put')
    col2.write('Le prix de l option put : '+ str( bs_put(lcp, strike_price, t, uty,re, sigma) ), unsafe_allow_html=True)
    st.write('votre strike price ', strike_price)
    
    
    st.header('volatilité')
    st.write('votre volatilité historique  ', 100*sigma , '%')
    col3, col4 = st.columns(2)
    col3.write('Volatilité implicite call '+str( 100 * call_implied_volatility(bs_call(lcp, strike_price, t, uty,re, sigma), lcp, strike_price, t, uty,re) )+'%', unsafe_allow_html=True)
    
    col4.write('Volatlité implicite put  '+ str(100 * put_implied_volatility(bs_put(lcp, strike_price, t, uty,re, sigma), lcp, strike_price, t, uty,re) )+'%', unsafe_allow_html=True)
    
    #####################################################################################################
    st.header("Les grecs")
    st.subheader("Delta")

    st.markdown("**c est la sensibilité de prix de l option par rapport au variation du prix du sous-jacent**")

    col_call_delta, col_put_delta = st.columns(2)
    st.subheader("Gamma")

    st.markdown('**Le gamma mesure la variation du delta qui est engendrée par une variation du cours du sous-jacent**')

    col_call_gamma, col_put_gamma = st.columns(2)
    st.subheader("Vega")

    st.markdown("**Le véga d’une option correspond au taux de variation d’une option consécutive à une variation de la volatilité**")

    col_call_vega, col_put_vega = st.columns(2)
    st.subheader("Theta")

    st.markdown("**Le thêta représente donc combien de valeur l’option perd par jour**")

    col_call_theta, col_put_theta = st.columns(2)
    st.subheader("Rho")

    st.markdown("**Le rho mesure la sensibilité par rapport aux taux**")

    col_call_rho, col_put_rho = st.columns(2)

    
    col_call_delta.write(str(call_delta(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    col_put_delta.write(str(put_delta(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    
    col_call_gamma.write(str(call_gamma(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    col_put_gamma.write(str(put_gamma(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    
    col_call_vega.write(str(call_vega(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    col_put_vega.write(str(put_vega(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)

    col_call_theta.write(str(call_theta(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    col_put_theta.write(str(put_theta(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)

    col_call_rho.write(str(call_rho(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    col_put_rho.write(str(put_rho(lcp, strike_price, t, uty,re, sigma)), unsafe_allow_html=True)
    
    
    with st.empty():
        for seconds in range(60):
            st.write(f"⏳ {seconds} seconds have passed")
            time.sleep(1)
        st.write("✔️ 1 minute over!")
    placeholder = st.empty()

def delta_call_consigne(s,k,c):
    #prob = " La probabilité de finir a l écheance dans la monnaie c'est "+c
    definition = "Si votre sous-jacent varie de 1 dinar alors votre prix d'option varie de "+ c + " dinar"
    final = ""
    if(abs(s-k) < 0.01):
        ### a la monnaie
        final = " Si vous avez 2 options alors vous devez vendre une action pour couvrir votre portefeuille "
    if( abs(s -k )>  0.01) :
       ###  en dehors de la monnaie
       final = ""
    if( abs(k - s )>  0.01 ) :
       ## dans la monnaie
       final = ""
        
    
def dataset_page() -> None:
    st.title('Indicateurs')
    col1, col2 = st.columns(2)
    st.write('yooo sig   ',sig1)
    st.write('yooo sig   ',T1)
    st.write('yooo sig   ',R1)
    st.write('yooo sig   ',R_e1)

def graphs_page() -> None:
    return 'hello world'


if __name__ == '__main__':

        # Test/Title
    
    stock = st.sidebar.selectbox( 'Vous pouvez choisir un taux de change ',('EURUSD=X','EURGBP=X','CADUSD=X','EURTND=X','TNDUSD=X','TNDGBP=X'))
    strike_price = st.sidebar.number_input('Entrer un strike price',value = 1.13 ,  format="%.5f")
    expiry = st.sidebar.date_input("Echéance " , value = datetime.now().replace(day=datetime.now().day+3))
    uty = st.sidebar.number_input('Entrer le taux sans risque',value = 0)
    re = st.sidebar.number_input('Entrer le taux étranger',value=0)


      


    home_page()
    
    
