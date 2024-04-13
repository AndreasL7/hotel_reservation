import streamlit as st
import streamlit.components.v1 as components
import gc
from joblib import load
from loguru import logger

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.model_selection import (
    train_test_split, 
)

from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder,
    LabelEncoder, 
    StandardScaler, 
    PolynomialFeatures
)

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
)

from ...utils.tweak_hotel_reservation import TweakFeatures

def ColourWidgetText(wgt_txt, wch_colour='#000000'):
    # Construct the HTML string
    htmlstr = f"""<script>
                     var elements = window.parent.document.querySelectorAll('*');
                     for (var i = 0; i < elements.length; i++) {{
                         if (elements[i].innerText === '{wgt_txt}') {{
                             elements[i].style.color = '{wch_colour}';
                         }}
                     }}
                  </script>"""
    
    # Render the HTML string using Streamlit components.html
    components.html(htmlstr, height=0, width=0)
        
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

def make_prediction(inputs, clf):
    
    if clf == "model_data_leakage":
        y_pred_data_leakage = load_model_data_leakage().predict(inputs)
        y_prob_data_leakage = load_model_data_leakage().predict_proba(inputs)[:,1]
        return y_pred_data_leakage, y_prob_data_leakage
    elif clf == "model_data_pipeline":
        optimal_threshold = 0.303
        y_pred_data_pipeline = load_model_soft().predict(inputs)
        y_prob_data_pipeline = load_model_soft().predict_proba(inputs)[:,1]
        y_pred_adj_threshold = (y_prob_data_pipeline >= optimal_threshold).astype(int)
        return y_pred_adj_threshold, y_prob_data_pipeline

def custom_confusion_matrix(y_true, y_pred):
    
    gc.enable()
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    display = metrics.ConfusionMatrixDisplay(cm, display_labels=['Canceled', 'Not_Canceled'])
    plot = display.plot()
    
    for row in plot.text_:
        for text in row:
            text.set_color('white')
        
    fig = plot.ax_.figure
    st.pyplot(fig)
    
    del(
        y_true,
        y_pred,
        fig
    )
    gc.collect()

def transformed_leaked(data):
    
    label_encoder = LabelEncoder()
    booking_status = pd.DataFrame(label_encoder.fit_transform(data['booking_status']), columns=['booking_status'])
    
    data = data.drop(columns=['booking_status'])
    faulty_date_obs = TweakFeatures().get_faulty_date_index(data)
    booking_status = booking_status.drop(index=faulty_date_obs)
    data = TweakFeatures().transform(data)
    
    ohe_categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_weekday']
    ohe  = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first').set_output(transform='pandas')

    orde_categorical_features = ['lead_time_category']
    orde = OrdinalEncoder().set_output(transform='pandas')

    standard_numerical_features = ['lead_time', 'avg_price_per_room']
    standard_scaler = StandardScaler().set_output(transform='pandas')
    
    ohe_result = ohe.fit_transform(data[ohe_categorical_features])
    orde_result = orde.fit_transform(data[orde_categorical_features])
    
    data_transformed = pd.concat([data.drop(columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_weekday', 'lead_time_category']), 
                                 ohe_result,
                                 orde_result],
                                 axis=1)
    
    data_scaled = standard_scaler.fit_transform(data_transformed[standard_numerical_features])
    data_transformed = data_transformed.drop(columns=['lead_time', 'avg_price_per_room'])
    data_scaled = pd.concat([data_transformed, data_scaled], axis=1)
    
    data_poly = PolynomialFeatures(degree=2, 
                                   interaction_only=False, 
                                   include_bias=True).set_output(transform='pandas').fit_transform(data_scaled)
    
    return data_poly, booking_status

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

def main():
    
    gc.enable()
    
    logger.info(f'Start Model Viz Session State: {st.session_state}')
    
    st.header("Model Performance Scorecard")
    
    data = pd.read_csv('data/raw/hotel_reservations.csv')
    
    if st.session_state['analyst'] == 'John':
        transformed_dataset, booking_status = transformed_leaked(data)
        
        X_train, X_test, y_train, y_test = train_test_split(transformed_dataset, 
                                                            booking_status, 
                                                            test_size=0.2, 
                                                            stratify=booking_status,
                                                            random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.2, 
                                                        stratify=y_train,
                                                        random_state=42)
        
        y_pred_data_leakage, y_prob_data_leakage = make_prediction(X_val, "model_data_leakage")
        
        accuracy_val = np.round(accuracy_score(y_val, y_pred_data_leakage), 2)
        precision_val = np.round(precision_score(y_val, y_pred_data_leakage), 2)
        recall_val = np.round(recall_score(y_val, y_pred_data_leakage), 2)
        f1_val = np.round(f1_score(y_val, y_pred_data_leakage), 2)
        auc_score_val = np.round(roc_auc_score(y_val, y_prob_data_leakage), 2)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Accuracy', accuracy_val)
        col2.metric('Precision', precision_val)
        col3.metric('Recall', recall_val)
        col4.metric('F1 Score', f1_val)
        col5.metric('ROC-AUC', auc_score_val)
        
        custom_confusion_matrix(y_val.values.ravel(), y_pred_data_leakage)
        
        st.divider()
        
        y_pred_data_leakage, y_prob_data_leakage = make_prediction(X_test, "model_data_leakage")
        
        accuracy_test = np.round(accuracy_score(y_test, y_pred_data_leakage), 2)
        precision_test = np.round(precision_score(y_test, y_pred_data_leakage), 2)
        recall_test = np.round(recall_score(y_test, y_pred_data_leakage), 2)
        f1_test = np.round(f1_score(y_test, y_pred_data_leakage), 2)
        auc_score_test = np.round(roc_auc_score(y_test, y_prob_data_leakage), 2)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Accuracy', accuracy_test, delta='-8.2%')
        col2.metric('Precision', precision_test, delta='-7.1%')
        col3.metric('Recall', recall_test, delta='-6.0%')
        col4.metric('F1 Score', f1_test, delta='-6.1%')
        col5.metric('ROC-AUC', auc_score_test, delta='-4.0%')
        
        custom_confusion_matrix(y_test.values.ravel(), y_pred_data_leakage)
        
        st.header("Peek into the Model...")
        col6, col7, col8 = st.columns(3)
        col6.metric('Number of Features', 903)
        col7.metric('Training Time', '~2h+')
        col8.metric('Explainability', 'Low')
        
    else:
        pipeline = load_pipeline()
        
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['booking_status']),
                                                            data['booking_status'], 
                                                            test_size=0.2, 
                                                            stratify=data['booking_status'],
                                                            random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size=0.2, 
                                                          stratify=y_train,
                                                          random_state=42)
        
        label_encoder = LabelEncoder()
        y_train = pd.DataFrame(label_encoder.fit_transform(y_train), columns=['booking_status'], index=y_train.index)
    
        faulty_date_obs_train = pipeline.named_steps['tweak_features'].get_faulty_date_index(X_train)
        X_train_transformed = pipeline.fit_transform(X_train, y_train.drop(faulty_date_obs_train).values.ravel())
        
        X_val_transformed = pipeline.transform(X_val)
        faulty_date_obs_val = pipeline.named_steps['tweak_features'].get_faulty_date_index(X_val)
        y_val = pd.DataFrame(label_encoder.transform(y_val), columns=['booking_status'], index=y_val.index)
        y_val = y_val.drop(faulty_date_obs_val)
        
        y_pred_data_pipeline, y_prob_data_pipeline = make_prediction(X_val_transformed, "model_data_pipeline")
        
        accuracy_val = np.round(accuracy_score(y_val, y_pred_data_pipeline), 2)
        precision_val = np.round(precision_score(y_val, y_pred_data_pipeline), 2)
        recall_val = np.round(recall_score(y_val, y_pred_data_pipeline), 2)
        f1_val = np.round(f1_score(y_val, y_pred_data_pipeline), 2)
        auc_score_val = np.round(roc_auc_score(y_val, y_prob_data_pipeline), 2)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Accuracy', accuracy_val)
        col2.metric('Precision', precision_val)
        col3.metric('Recall', recall_val)
        col4.metric('F1 Score', f1_val)
        col5.metric('ROC-AUC', auc_score_val)
        
        custom_confusion_matrix(y_val.values.ravel(), y_pred_data_pipeline)
        
        st.divider()
        
        X_test_transformed = pipeline.transform(X_test)
        faulty_date_obs_test = pipeline.named_steps['tweak_features'].get_faulty_date_index(X_test)
        y_test = pd.DataFrame(label_encoder.transform(y_test), columns=['booking_status'], index=y_test.index)
        y_test = y_test.drop(faulty_date_obs_test)
        
        y_pred_data_pipeline, y_prob_data_pipeline = make_prediction(X_test_transformed, "model_data_pipeline")
        
        accuracy_test = np.round(accuracy_score(y_test, y_pred_data_pipeline), 2)
        precision_test = np.round(precision_score(y_test, y_pred_data_pipeline), 2)
        recall_test = np.round(recall_score(y_test, y_pred_data_pipeline), 2)
        f1_test = np.round(f1_score(y_test, y_pred_data_pipeline), 2)
        auc_score_test = np.round(roc_auc_score(y_test, y_prob_data_pipeline), 2)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Accuracy', accuracy_test)
        col2.metric('Precision', precision_test)
        col3.metric('Recall', recall_test)
        col4.metric('F1 Score', f1_test)
        col5.metric('ROC-AUC', auc_score_test)
        
        custom_confusion_matrix(y_test.values.ravel(), y_pred_data_pipeline)
        
        st.header("Peek into the Model...")
        col6, col7, col8 = st.columns(3)
        col6.metric('Number of Features', '10')
        col7.metric('Training Time', '~5mins')
        col8.metric('Explainability', 'High')

    ColourWidgetText('903', '#ff0000')
    ColourWidgetText('~2h+', '#ff0000')
    ColourWidgetText('Low', '#ff0000')
    
    ColourWidgetText('10', '#00B050')
    ColourWidgetText('~5mins', '#00B050')
    ColourWidgetText('High', '#00B050')
    
    gc.collect()

if __name__ == "__main__":
    main()