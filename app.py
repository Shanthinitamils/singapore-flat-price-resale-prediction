import streamlit as st
import pickle
import numpy as np
import os
from streamlit_option_menu import option_menu
import joblib

st.set_page_config(page_title='Singapore Flat Price Resale Prediction',initial_sidebar_state='expanded',layout='wide')

class option():
    # Value get from data
    town_name=['TAMPINES','YISHUN','JURONG WEST','BEDOK','WOODLANDS','HOUGANG','ANG MO KIO','BUKIT BATOK',
               'CHOA CHU KANG','PASIR RIS','SENGKANG','BUKIT MERAH','TOA PAYOH','BUKIT PANJANG','CLEMENTI',
               'GEYLANG','KALLANG/WHAMPOA','QUEENSTOWN','JURONG EAST','SERANGOON','BISHAN','PUNGGOL','SEMBAWANG',
               'MARINE PARADE','CENTRAL AREA','BUKIT TIMAH','LIM CHU KANG']
    town_name_dict={'TAMPINES':23.0,'YISHUN':26.0,'JURONG WEST':13.0,'BEDOK':1.0,'WOODLANDS':25.0,'HOUGANG':11.0,'ANG MO KIO':0.0,'BUKIT BATOK':3.0,
               'CHOA CHU KANG':8.0,'PASIR RIS':17.0,'SENGKANG':21.0,'BUKIT MERAH':4.0,'TOA PAYOH':24.0,'BUKIT PANJANG':5.0,'CLEMENTI':9.0,
               'GEYLANG':10.0,'KALLANG/WHAMPOA':14.0,'QUEENSTOWN':19.0,'JURONG EAST':12.0,'SERANGOON':22.0,'BISHAN':2.0,'PUNGGOL':18.0,'SEMBAWANG':20.0,
               'MARINE PARADE':16.0,'CENTRAL AREA':7.0,'BUKIT TIMAH':6.0,'LIM CHU KANG':15.0}

    flat_model=['MODEL A','IMPROVED','NEW GENERATION','SIMPLIFIED','PREMIUM APARTMENT','STANDARD','APARTMENT',
                'MAISONETTE','MODEL A2','DBSS','MODEL A-MAISONETTE','ADJOINED FLAT','TERRACE','MULTI GENERATION',
                'TYPE S1','2-ROOM','IMPROVED-MAISONETTE','TYPE S2','PREMIUM APARTMENT LOFT','PREMIUM MAISONETTE',
                '3GEN']
    flat_model_dict={'MODEL A':8,'IMPROVED':5,'NEW GENERATION':12,'SIMPLIFIED':16,'PREMIUM APARTMENT':13,'STANDARD':17,'APARTMENT':3,
                'MAISONETTE':7,'MODEL A2':10,'DBSS':4,'MODEL A-MAISONETTE':9,'ADJOINED FLAT':2,'TERRACE':18,'MULTI GENERATION':11,
                'TYPE S1':19,'2-ROOM':0,'IMPROVED-MAISONETTE':6,'TYPE S2':20,'PREMIUM APARTMENT LOFT':14,'PREMIUM MAISONETTE':15,
                '3GEN':1}
    flat_type=['4 ROOM','3 ROOM','5 ROOM','EXECUTIVE','2 ROOM','1 ROOM','MULTI GENERATION','MULTI-GENERATION' ]

    flat_type_dict={'4 ROOM':3,'3 ROOM':2,'5 ROOM':4,'EXECUTIVE':5,'2 ROOM':1,'1 ROOM':0,'MULTI GENERATION':6,'MULTI-GENERATION':7}

st.title(":red[Singapore Flat Price Resale Prediction]")
st.subheader(':green[Introduction]')
st.markdown('''The objective of this project is to develop a machine learning model and deploy it as a user-friendly 
            web application that predicts the resale prices of flats in Singapore. 
            This predictive model will be based on historical data of resale flat transactions.''')

with st.form('prediction'):
    col2,col1=st.columns(2)
    with col2:
        town=st.selectbox(label='Town', options=option.town_name)

        flat_type=st.selectbox(label='Flat_Type', options=option.flat_type)

        floor_area=st.number_input(label='Floor_area_sqm',min_value=1.0,max_value=200.0,value=1.0)

        flat_model=st.selectbox(label='Flat_Model', options=option.flat_model)

        lease_commence=st.number_input(label='Lease_commence_date',min_value=1960,value=1960)

        remaining_lease=st.number_input(label='Remaining_Lease',min_value=1.0,max_value=200.0,value=1.0)

        year=st.number_input(label='Year',min_value=1990,value=1990)

    with col1:

        year_hold=st.number_input(label='Years_Holding',min_value=0.0,value=1.0)

        current_remaining_lease=st.number_input(label='Current_remaning_lease',min_value=0.0,value=1.0)

        property_age=st.number_input(label='Age of Property',min_value=0.0,value=1.0)

        lower_storey=st.number_input(label='Lower Storey',min_value=0.0,value=1.0)

        upper_storey=st.number_input(label='upper Storey',min_value=0.0,value=1.0)

        price_per_sqm=st.number_input(label='Price per Sqm',min_value=100.0,value=100.0)

        button=st.form_submit_button('PREDICT',use_container_width=True)

    if button:
            # Not fillung all the columns Command
        if not all([town, flat_type,floor_area,flat_model, lease_commence,remaining_lease,
                        year, year_hold, current_remaining_lease, property_age,lower_storey,
                        upper_storey,price_per_sqm]):
            st.error("Please fill in all required fields.")
        else:
            decision_tree_model = joblib.load('decision_tree_model.pkl')

            town=option.town_name_dict[town]
            flat_type=option.flat_type_dict[flat_type]
            flat_model=option.flat_model_dict[flat_model]

            lower_storey=np.log1p(lower_storey)
            upper_storey=np.log1p(upper_storey)
            price_per_sqm=np.log1p(price_per_sqm)

            user_data=np.array([[town, flat_type,floor_area,flat_model, lease_commence,remaining_lease,
                        year, year_hold, current_remaining_lease, property_age,lower_storey,
                        upper_storey,price_per_sqm ]]) # Giving input to predict data using picle file
                
            pred= decision_tree_model.predict(user_data)

            resale_price=np.expm1(pred[0]) # selling price is in log for form we use exp to retransform the Data

            st.subheader(f":green[Predicted Reselling Price :] {resale_price:.2f}")








            