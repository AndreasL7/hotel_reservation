import sys
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from loguru import logger

from streamlit_gallery import components
from streamlit_gallery.utils.page import page_group

from streamlit_gallery.utils.tweak_hotel_reservation import TweakFeatures

@st.cache_data
def load_lottie_url(url: str):
    
    try:
        r = requests.get(url)
        # if r.status_code != 200:
        #     return None
    except:
        logger.info('Failed to load Lottie URL.')
        sys.exit(0)
    return r.json()

def main():

    # Custom CSS
    styles = """
        <style>
            body {
                background-color: #FAF3E0; 
                font-family: "Arial", sans-serif;
            }
            
            h1 {
                font-family: "Georgia", monospace; 
                color: #3E2723;
            }
            
            .stButton>button {
                background-color: #575735;
                color: white !important;
            }
        </style>
    """
    
    st.markdown(styles, unsafe_allow_html=True)
    
    #Lottie
    lottie_url = "https://lottie.host/3bb00abe-7ae2-4f27-8c34-c9533b28ab60/YpCYQYAyGu.json"  # Sample URL, replace with your desired animation
    lottie_animation = load_lottie_url(lottie_url)
    st_lottie(lottie_animation, speed=1, width=200, height=200)
    
    st.title('Hotel Booking Cancellation Prediction')
    st.markdown("""Due to resource constraint provided by Streamlit Sharing, only permitted users are allowed access. Please note that the app interface is not flawless, occasional state rollback may occur. Nevertheless, the app serves its purpose of demonstrating the model's performance.""")
    

    # password_guess = st.text_input('What is the Password?') 
    # if password_guess != st.secrets["password"]:
    #     st.stop()

    # Sidebar for navigation
    page = page_group("p")

    with st.sidebar:
        st.title("🕵️‍♂️ Hotel Reservation")
        st.caption("where Storytelling meets Modelling")
        st.write("")
        st.markdown('Made by <a href="https://www.linkedin.com/in/andreaslukita7/">Andreas Lukita</a>', unsafe_allow_html=True)

        with st.expander("⏳ COMPONENTS", True):
            page.item("Introduction", components.show_introduction, default=True)
            page.item("Prediction and Modelling⭐", components.show_prediction)
            page.item("Result", components.show_result)

    page.show()
    
if __name__ == "__main__":
    main()