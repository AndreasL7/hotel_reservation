import streamlit as st
import gc
from loguru import logger

def main():
    
    gc.enable()
    
    st.subheader("Welcome!")
    
    st.write("""
    Imagine you are a newly hired hotel manager, carefully juggling bookings to maximize your occupancy. Everything goes well until cancellations start rolling in, leaving you with empty rooms and lost revenue. How do you answer to the higher-ups? Wouldn't it be amazing to have a crystal ball that reveals which bookings are likely to be cancelled? This app is that crystal ball – it helps you predict cancellations before they happen, protect the bottom line, and get your promotion.
    """)
    
    image_url = "https://www.hospitalitynet.org/picture/xxl_153107378.jpg?t=1571895948"
    
    st.markdown(f'<a href="{image_url}"><img src="{image_url}" alt="description" width="700"/></a>', unsafe_allow_html=True)
        
    st.write("""
    
    Navigate to the **Prediction and Modelling** page to understand how our model works. 
    
    Grab your coffee and enjoy the investigation ahead! ☕️
    """)
    
    logger.info(st.session_state)
    
    gc.collect()

if __name__ == "__main__":
    main()