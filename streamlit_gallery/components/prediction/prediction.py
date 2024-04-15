import gc
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import time
from joblib import load

import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder,
    LabelEncoder, 
    StandardScaler, 
    PolynomialFeatures
)

from loguru import logger

from ...utils.tweak_hotel_reservation import TweakFeatures

def load_lottie_url(url: str):

    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_pipeline():
    primary_path = 'models/pipeline.joblib'
    alternative_path = '../../../models/pipeline.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Pipeline not found in both primary and alternative directories!")
        
def load_model_data_leakage():
    primary_path = 'models/data_leakage.joblib'
    alternative_path = '../../../models/data_leakage.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

def load_model_soft():
    primary_path = 'models/best_model_voting_soft.joblib'
    alternative_path = '../../../models/best_model_voting_soft.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

def get_session_value(key, default_value):
    
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

def session_slider_int(label, min_value, max_value, key, default_value, step=1):

    value = get_session_value(key, default_value)
    new_value = st.slider(label, min_value, max_value, value, step=step)
    st.session_state[key] = new_value
    return new_value

def session_radio(label, options, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.radio(label, options, index=options.index(value))
    st.session_state[key] = new_value
    return new_value

def session_selectbox(label, options, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.selectbox(label, options, index=options.index(value))
    st.session_state[key] = new_value
    return new_value

def session_number_input(label, key, default_value, **kwargs):

    value = get_session_value(key, default_value)
    new_value = st.number_input(label, value=value, **kwargs)
    st.session_state[key] = new_value
    return new_value
    
def update_form_values(new_values):
    for key, value in new_values.items():
        st.session_state[key] = value

def transformed_leaked(input_data):
    
    label_encoder = LabelEncoder()
    
    original_data = pd.read_csv('data/raw/hotel_reservations.csv')
    booking_status = pd.DataFrame(label_encoder.fit_transform(original_data['booking_status']), columns=['booking_status'])
    
    original_data = original_data.drop(columns=['booking_status'])
    faulty_date_obs = TweakFeatures().get_faulty_date_index(pd.DataFrame(original_data))
    
    booking_status = booking_status.drop(index=faulty_date_obs)
    
    original_data_transformed = TweakFeatures().transform(original_data)
    
    input_data_transformed = TweakFeatures().transform(pd.DataFrame(input_data, index=[0]))
    
    ohe_categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_weekday']
    ohe  = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first').set_output(transform='pandas')

    orde_categorical_features = ['lead_time_category']
    orde = OrdinalEncoder().set_output(transform='pandas')

    standard_numerical_features = ['lead_time', 'avg_price_per_room']
    standard_scaler = StandardScaler().set_output(transform='pandas')
    
    ohe_result = ohe.fit_transform(original_data_transformed[ohe_categorical_features])
    orde_result = orde.fit_transform(original_data_transformed[orde_categorical_features])
    
    original_data_transformed = pd.concat([original_data_transformed.drop(columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_weekday', 'lead_time_category']), 
                                           ohe_result,
                                           orde_result],
                                           axis=1)
    
    ohe_result_input_data = ohe.transform(input_data_transformed[ohe_categorical_features])
    orde_result_input_data = orde.transform(input_data_transformed[orde_categorical_features])
    
    input_data_transformed = pd.concat([input_data_transformed.drop(columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_weekday', 'lead_time_category']), 
                                        ohe_result_input_data,
                                        orde_result_input_data],
                                        axis=1)
    
    original_data_scaled = standard_scaler.fit_transform(original_data_transformed[standard_numerical_features])
    original_data_transformed = original_data_transformed.drop(columns=['lead_time', 'avg_price_per_room'])
    original_data_scaled = pd.concat([original_data_transformed, original_data_scaled], axis=1)
    
    input_data_scaled = standard_scaler.transform(input_data_transformed[standard_numerical_features])
    input_data_transformed = input_data_transformed.drop(columns=['lead_time', 'avg_price_per_room'])
    input_data_scaled = pd.concat([input_data_transformed, input_data_scaled], axis=1)
    
    poly = PolynomialFeatures(degree=2, 
                              interaction_only=False, 
                              include_bias=True).set_output(transform='pandas')
    
    original_data_poly = poly.fit_transform(original_data_scaled)
    
    input_data_poly = poly.transform(input_data_scaled)
    
    return input_data_poly

def make_prediction(inputs,
                    analyst: str = "John"):
    
    optimal_threshold = 0.3
    
    if analyst == "John":
        transformed_dataset = transformed_leaked(pd.DataFrame(inputs, index=[0]))
        y_pred_data_leakage = load_model_data_leakage().predict(pd.DataFrame(transformed_dataset, index=[0]))
        return y_pred_data_leakage
    elif analyst == "Thomas":
        pipeline = load_pipeline()
        transformed_dataset = pipeline.transform(pd.DataFrame(inputs, index=[0]))
        y_prob_soft = load_model_soft().predict_proba(transformed_dataset)[:,1]
        y_pred_soft = (y_prob_soft >= optimal_threshold).astype(int)
        return y_pred_soft[0]
    
def main():
    
    gc.enable()

    logger.info(f'Start Prediction Session State: {st.session_state}')
    st.title("Is this customer likely to cancel their reservation?")
    st.subheader("Should we oversell or prepare for a no-show? 🤔")
    
    if st.session_state is None:
        st.session_state = {'analyst': 'John',
                            'client_name': 'Ryan',
                            'Booking_ID': 'stub',
                            'no_of_adults': 2,
                            'no_of_children': 2,
                            'no_of_weekend_nights': 2,
                            'no_of_week_nights': 2,
                            'type_of_meal_plan': 'Meal Plan 1',
                            'required_car_parking_space': 0,
                            'room_type_reserved': 'Room_Type 1',
                            'lead_time': 90,
                            'arrival_year': 2024,
                            'arrival_month': 12,
                            'arrival_date': 20,
                            'market_segment_type': 'Online',
                            'repeated_guest': 0,
                            'no_of_previous_cancellations': 0,
                            'no_of_previous_bookings_not_canceled': 0,
                            'avg_price_per_room': 100.0,
                            'no_of_special_requests': 0}
    
    # Initialize session state
    if 'form_content' not in st.session_state:
        
        st.session_state = {'analyst': 'John',
                            'client_name': 'Ryan',
                            'Booking_ID': 'stub',
                            'no_of_adults': 2,
                            'no_of_children': 2,
                            'no_of_weekend_nights': 2,
                            'no_of_week_nights': 2,
                            'type_of_meal_plan': 'Meal Plan 1',
                            'required_car_parking_space': '0',
                            'room_type_reserved': 'Room_Type 1',
                            'lead_time': 90,
                            'arrival_year': 2024,
                            'arrival_month': 12,
                            'arrival_date': 20,
                            'market_segment_type': 'Online',
                            'repeated_guest': '0',
                            'no_of_previous_cancellations': 0,
                            'no_of_previous_bookings_not_canceled': 0,
                            'avg_price_per_room': 100.0,
                            'no_of_special_requests': 0}
        
        st.session_state['form_content'] = {'analyst': 'John',
                                            'client_name': 'Ryan',
                                            'Booking_ID': 'stub',
                                            'no_of_adults': 2,
                                            'no_of_children': 2,
                                            'no_of_weekend_nights': 2,
                                            'no_of_week_nights': 2,
                                            'type_of_meal_plan': 'Meal Plan 1',
                                            'required_car_parking_space': '0',
                                            'room_type_reserved': 'Room_Type 1',
                                            'lead_time': 90,
                                            'arrival_year': 2024,
                                            'arrival_month': 12,
                                            'arrival_date': 20,
                                            'market_segment_type': 'Online',
                                            'repeated_guest': '0',
                                            'no_of_previous_cancellations': 0,
                                            'no_of_previous_bookings_not_canceled': 0,
                                            'avg_price_per_room': 100.0,
                                            'no_of_special_requests': 0}
    
    st.write("Try the model created by our two data scientists below!")
    st.subheader("Meet John and Thomas!")
    
    col_analyst_john, col_analyst_thomas = st.columns(2)

    with col_analyst_john:
        lottie_url = "https://lottie.host/0db51d3e-e84e-4e5a-8b1e-f73a89a77f65/i1GvROt5y3.json"
        lottie_animation = load_lottie_url(lottie_url)
        st_lottie(lottie_animation, speed=1, width=350, height=350)
        st.markdown(
            "<div style='text-align: center'><b>John</b></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center'>A fresh graduate with little to no experience in DSA, ChatGPT is his soul mate</div><br>", unsafe_allow_html=True)
        
        if st.button("Choose me!"):
            sampleA = {'analyst': 'John'}
        
            st.session_state.get('analyst', sampleA["analyst"])
            update_form_values(sampleA)
            logger.info("Analyst John has been chosen to make the prediction...")
            logger.info(f'After Choosing Analyst Session State: {st.session_state}')

    with col_analyst_thomas:
        lottie_url = "https://lottie.host/067bfd39-6ab6-484b-abd1-37451c842fd3/4OhK1ZCsaG.json"
        lottie_animation = load_lottie_url(lottie_url)
        st_lottie(lottie_animation, speed=1, width=350, height=350)
        st.markdown(
            "<div style='text-align: center'><b>Thomas</b></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center'>Trained in understanding and processing data, work collaboratively with LLM</div><br>", unsafe_allow_html=True)
        
        if st.button("Pick me!"):
            sampleB = {'analyst': 'Thomas'}
            
            st.session_state.get('analyst', sampleB["analyst"])
            update_form_values(sampleB)
            logger.info("Analyst Thomas has been chosen to make the prediction...")
            logger.info(f'After Choosing Analyst Session State: {st.session_state}')
    
    st.divider()
    
    client_name = st.text_input("Enter the client's name", st.session_state["client_name"])
    st.session_state["client_name"] = client_name
    
    with st.form('user_inputs'):

        st.header("Chapter 1: The Mysterious Client")
        st.write("Unveiling the Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            no_of_adults = session_slider_int(
                           label="How many adults will be staying? (0-4)", 
                           min_value=0,
                           max_value=4,
                           key='no_of_adults',
                           default_value=st.session_state.get('no_of_adults', 2)
                           )
            
            no_of_children = session_slider_int(
                             label="How many kids are part of the crew? (0-4)", 
                             min_value=0,
                             max_value=4,
                             key='no_of_children', 
                             default_value=st.session_state.get('no_of_children', 2)
                             )
        
        with col2:
            no_of_weekend_nights = session_number_input(
                                   label="How many weekend nights is the guest planning to stay?", 
                                   key='no_of_weekend_nights', 
                                   default_value=st.session_state.get('no_of_weekend_nights', 2)
                                   )
            
            no_of_week_nights = session_number_input(
                                label="How many weekday nights is the guest planning to stay?",
                                key='no_of_week_nights', 
                                default_value=st.session_state.get('no_of_week_nights', 2)
                                )
    
        st.header("Chapter 2: User Itinerary")
        st.write("Unravelling the Requests")
        
        col3, col4 = st.columns(2)
        
        with col3:
            type_of_meal_plan = session_selectbox(
                                label="Any breakfast plan for the guest?", 
                                options=["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"], 
                                key='type_of_meal_plan', 
                                default_value=st.session_state.get('type_of_meal_plan', "Meal Plan 1")
                                )
            
            required_car_parking_space = session_radio(
                                         label="Does the guest require car parking space", 
                                         options=['0', '1'], 
                                         key='required_car_parking_space', 
                                         default_value=st.session_state.get('required_car_parking_space', '0')
                                         )
            
            room_type_reserved = session_selectbox(
                                 label="Which room type is requested for this booking?", 
                                 options=["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"], 
                                 key='room_type_reserved', 
                                 default_value=st.session_state.get('room_type_reserved', "Room_Type 1")
                                 )
            
        with col4:
            market_segment_type = session_selectbox(
                                  label="Which market segment does this customer fall under?", 
                                  options=["Offline", "Online", "Corporate", "Aviation", "Complementary"], 
                                  key='market_segment_type', 
                                  default_value=st.session_state.get('market_segment_type', "Online")
                                  )
            
            no_of_special_requests = session_number_input(
                                     label="How many special requests are made by the guest?", 
                                     key='no_of_special_requests', 
                                     default_value=st.session_state.get('no_of_special_requests', 0)
                                     )
    
        st.header("Chapter 3: More details on the booking")
        st.write("When is the guest checking in?")
        
        col5, col6 = st.columns(2)
        
        with col5:
            arrival_year = session_slider_int(
                           label="Arrival Year", 
                           min_value=2024, 
                           max_value=2026, 
                           key='arrival_year', 
                           default_value=st.session_state.get('arrival_year', 2024)
                           )
            
            arrival_month = session_slider_int(
                            label="Arrival Month", 
                            min_value=1,
                            max_value=12, 
                            key='arrival_month', 
                            default_value=st.session_state.get('arrival_month', 12)
                            )
            
            arrival_date = session_slider_int(
                           label="Arrival Date", 
                           min_value=1,
                           max_value=31,
                           key='arrival_date', 
                           default_value=st.session_state.get('arrival_date', 20)
                           )
        
        with col6:
            lead_time = session_number_input(
                        label="From their booking time to arrival day, how many days in between?", 
                        key='lead_time', 
                        default_value=st.session_state.get('lead_time', 90)
                        )
            
            avg_price_per_room = session_number_input(
                                 label="What is the average price per room per night?", 
                                 key='avg_price_per_room', 
                                 default_value=st.session_state.get('avg_price_per_room', 100.0)
                                 )
    
        st.header("Chapter 4: Client's history")
        st.write("What's the client's history with us? A peek into the past...")
        
        col7, col8 = st.columns(2)
        
        with col7:
            repeated_guest = session_radio(
                             label="Has the guest stayed with us before?", 
                             options=["0", "1"], 
                             key='repeated_guest', 
                             default_value=st.session_state.get('repeated_guest', '0')
                             )
            
            no_of_previous_cancellations = session_number_input(
                                           label="How many times has the guest cancelled their reservation?", 
                                           key='no_of_previous_cancellations', 
                                           default_value=st.session_state.get('no_of_previous_cancellations', 0)
                                           )
            
        with col8:
            no_of_previous_bookings_not_canceled = session_number_input(
                                                   label="How many times has the guest not cancelled their reservation?", 
                                                   key='no_of_previous_bookings_not_canceled', 
                                                   default_value=st.session_state.get('no_of_previous_bookings_not_canceled', 0)
                                                   )
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(f"Dignosing {client_name}'s cancellation likelihood...")
            progress_bar = st.progress(0)
            
            for perc_completed in range(100):
                time.sleep(0.01)
                progress_bar.progress(perc_completed+1)
            
            inputs = {
                'Booking_ID': 'stub',
                'no_of_adults': no_of_adults,
                'no_of_children': no_of_children,
                'no_of_weekend_nights': no_of_weekend_nights,
                'no_of_week_nights': no_of_week_nights,
                'type_of_meal_plan': type_of_meal_plan,
                'required_car_parking_space': int(required_car_parking_space),
                'room_type_reserved': room_type_reserved,
                'lead_time': lead_time,
                'arrival_year': arrival_year,
                'arrival_month': arrival_month,
                'arrival_date': arrival_date,
                'market_segment_type': market_segment_type,
                'repeated_guest': int(repeated_guest),
                'no_of_previous_cancellations': no_of_previous_cancellations,
                'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
                'avg_price_per_room': avg_price_per_room,
                'no_of_special_requests': no_of_special_requests,
            }
            
            prediction = make_prediction(inputs=inputs,
                                         analyst=st.session_state.get('analyst')) 
            if prediction == 0:
                st.error(f"Our analyst {st.session_state.get('analyst')} suggests that {client_name} is likely to cancel his booking reservation.")
            else:
                st.success(f"Our analyst {st.session_state.get('analyst')} suggests that {client_name} is unlikely to cancel his booking reservation.")
                
            logger.info(f'After Making Prediction Session State: {st.session_state}')
            
            del(
                client_name,
                col1,
                col2,
                col3,
                col4,
                col5,
                col6,
                col7,
                col8,
                submitted,
                inputs,
                prediction
            )
            gc.collect()
            
if __name__ == "__main__":
    main()